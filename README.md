# SPOT: An Efficient Training-Free Task Similarity Quantification Method for Continual Learning

This repository hosts the code and experiments for **SPOT**, a training-free and efficient task similarity quantification method in continual learning. The project is currently **under review**.

---

## 🧠 Project Overview

SPOT leverages changes in empirical loss to measure task similarity without requiring any model fine-tuning. This repository includes:

- 🔍 Functional similarity estimation code for SplitMNIST and SplitCIFAR10
- 📊 Jupyter notebooks to replicate our accuracy/forgetting evaluation
- 📁 Lightweight design suitable for Colab execution

### ✅ Summary

> This work proposes SPOT, a lightweight method to quickly estimate task similarity using just one batch of new task data. It enables efficient prediction of catastrophic forgetting risk before training, offering a low-cost solution for continual learning in resource-constrained environments.

### 📐 Task Similarity Measure

We define task similarity based on the empirical loss change observed when parameters trained on one task are updated using a gradient step from another task. The SPOT method measures:

![Task Similarity Equation 1](eq3.png)

To obtain a positive, bounded similarity score, we apply exponential scaling:

![Task Similarity Equation 1](eq4.png)

These formulas allow efficient estimation of transferability between tasks. In practice, lower similarity corresponds to higher forgetting risk.

---

## 📦 Requirements 

This project uses a Colab-friendly environment.

```text
Python        3.9.16
PyTorch       1.13.1
Torchvision   0.14.1
scikit-learn  1.2.1
```

---

## 🚀 Usage

Please open the `.ipynb` example files in **Google Colab**, and upload the corresponding folders (e.g., [`func_simMNIST`](func_simMNIST) or [`func_simCIFAR10`](func_simCIFAR10)) into `/home` of the virtual environment.

⚠️ **Do not open the `.ipynb` files directly from your Google Drive** — this significantly slows down data generation due to inefficient I/O between Drive and Colab’s virtual file system.

---

## 📊 Reproducible Experiments

We provide ready-to-run notebooks:

- **SplitMNIST experiment**  
  <a target="_blank" href="https://colab.research.google.com/github/wang-xulong/Func_sim/blob/main/MNIST_sim_acc_fgt.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>  

- **SplitCIFAR10 experiment**  
  <a target="_blank" href="https://colab.research.google.com/github/wang-xulong/Func_sim/blob/main/CIFAR10_sim_acc_fgt.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>  

### 📈 Experimental Findings

The following figure illustrates the effects of task similarity on model accuracy and forgetting across SplitMNIST and SplitCIFAR10 datasets:

![Effects of Task Similarity](spot_similarity_effects.png)

---

## 📁 Repository Structure

```
├── func_simMNIST/           # Code for functional similarity on SplitMNIST
├── func_simCIFAR10/         # Code for functional similarity on SplitCIFAR10
├── func_simCIFAR100/        # Extension to CIFAR100 (WIP)
├── CIFAR10_sim_acc_fgt.ipynb
├── MNIST_sim_acc_fgt.ipynb
└── requirements.txt
```

---

## 🙏 Acknowledgment

Our dataset generation and continual learning framework are built upon [Avalanche](https://avalanche.continualai.org/), a leading open-source CL library.
