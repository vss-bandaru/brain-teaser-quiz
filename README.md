# Medical Image Analysis POC
A proof-of-concept medical image analysis system for chest X-ray classification and pathology detection using deep learning. The project integrates image classification and object detection models to analyze radiographic images from the PadChest GR dataset and includes a Streamlit web app for interactive analysis and PDF report generation.

## Project Overview
The system performs two complementary tasks:

* Classification Predicts whether an X-ray is Normal or Abnormal.
* Detection For abnormal cases, identifies and localizes specific pathologies using bounding box detection.

A Streamlit frontend allows users to upload images, visualize results, and download structured reports.

## Project Structure
* 01_EDA.ipynb                      # Exploratory Data Analysis
* 02_Image_Classification.ipynb     # Classification model training (EfficientNet-B4)
* 03_Image_Preprocessing.ipynb      # Image normalization & bbox verification
* 04_detector_4_unbalanced.ipynb    # Detection model (unbalanced – 4 classes)
* 05_Medical_Analysis.ipynb         # Evaluation & medical interpretation
* app.py                            # Streamlit web application
* detector_5_unbalanced.ipynb       # Test detection model (5-class unbalanced)
* detector_8_balanced.ipynb         # Test detection model (8-class balanced)
* detector_13_unbalanced.ipynb      # Test detection model (13-class unbalanced)
* upload_dir.ipynb                  # S3 upload utilities

## Workflow
### Exploratory Data Analysis – 01_EDA.ipynb
* Basic dataset overview
* Class distribution
* Image quality and format checks
### Classification – 02_Image_Classification.ipynb
* Binary classification: Normal vs Abnormal
* Model: EfficientNet-B4
* Input size: 380×380
* Trained across balanced and imbalanced splits
### Image Preprocessing – 03_Image_Preprocessing.ipynb
* Intensity normalization
* Bounding box normalization
* Validating consistency between images and annotations
### Abnormality Detection – 04_detector_4_unbalanced.ipynb
* Model: Faster R-CNN (ResNet-50 FPN)
* Primary version trained on 4 pathology classes:
    * Aortic Atheromatosis
    * Aortic Elongation
    * Cardiomegaly
    * Scoliosis
* Additional variations:
    * 5-class unbalanced (detector_5_unbalanced.ipynb)
    * 8-class balanced (detector_8_balanced.ipynb)
    * 13-class unbalanced (detector_13_unbalanced.ipynb)
### 5. Medical Analysis – 05_Medical_Analysis.ipynb
* Evaluation metrics
* Detection mAP & classification metrics
* Clinical interpretation and comparison of detector variants

## Models
### Classification Model
* Architecture: EfficientNet-B4
* Task: Normal / Abnormal
* Output: Probability scores + final predicted label
* Checkpoint: efficientnet_b4_best.pth
* S3 Location: s3://padchest-gr-xray-data/gold/classification_output/

### Detection Model
* Architecture: Faster R-CNN (ResNet-50 FPN)
* Task: Multi-class pathology bounding box detection
* Output: Bounding boxes + labels + confidence scores
* Checkpoint: final_model.pth
* S3 Location: s3://padchest-gr-xray-data/gold/detection_output/

## Model Paths (Used in App)
CLS_CKPT_PATH = "s3://padchest-gr-xray-data/gold/classification_output/efficientnet_b4_best.pth"

DET_CKPT_PATH = "s3://padchest-gr-xray-data/gold/detection_output/final_model.pth"
