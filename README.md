# Traffic Sign Recognition Program
### Overview

This project aims to train a deep learning model so that it can decode traffic signs from images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, it is then tested on new images of traffic signs taken manually.

---
### Generate Additional Data and Data Preprocessing
Additional data are being generated from the original ones by randomly rotating, translating and shearing so that the model learns to classify images that are not necessarily taken at an ideal angle. Each images are being preprocessed by using histogram equalization.

### Models

#### LeNet
- Layer 1: Convolutional
  - 6 5x5 filters
  - relu activation
  - 2x2 max pooling
- Layer 2: Convolutional
  - 16 5x5 filters
  - relu activation
  - 2x2 max pooling
  - flatten
- Layer 3: Fully Connected
  - 120 neurons
  - relu activation
- Layer 4: Fully Connected
  - 84 neurons
  - relu activation
- Layer 5: Fully Connected
  - 43 neurons

Validation Accuracy = 0.986
Test Accuracy = 0.930

#### VGG16 Net
- Layer 1: Convolutional
  - 64 3x3 filters
  - relu activation
  - 2x2 max pooling
- Layer 2: Convolutional
  - 128 3x3 filters
  - relu activation
  - 2x2 max pooling
  - flatten
- Layer 3: Fully Connected
  - 4096 neurons
  - relu activation
- Layer 4: Fully Connected
  - 4096 neurons
  - relu activation
- Layer 5: Fully Connected
  - 43 neurons

Validation Accuracy = 0.997
Test Accuracy = 0.953

---
### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### Dataset

[Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset that has images resized to 32x32.
