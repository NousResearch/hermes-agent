# Hyperparameters

Complete guide to SimPO hyperparameter selection and tuning.

## Overview

Key hyperparameters in SimPO:
1. **Learning Rate** - Most critical
2. **Beta (β)** - Reward scaling
3. **Gamma-Beta Ratio (γ/β)** - Target margin
4. **SFT Weight** - Regularization strength

## Learning Rate

### Recommended Ranges

**By model size**:
| Model Size | Learning Rate | Notes |
|------------|---------------|-------|
| 1B-3B | 5e-7 to 1e-6 | Higher end safe |
| 7B-8B | 3e-7 to 5e-7 | **Standard** |
| 13B-30B | 1e-7 to 3e-7 | Lower for stability |
| 70B+ | 5e-8 to 1e-7 | Very conservative |

**By task type**:
| Task | Learning Rate | Reason |
|------|---------------|--------|
| General chat | 5e-7 | Standard |
| Code generation | 3e-7 | **Precise reasoning** |
| Math reasoning | 3e-7 | **Careful optimization** |
| Creative writing | 1e-6 | More aggressive OK |

### Why Learning Rate Matters

**Too high** (> 1e-6 for 7B):
- Loss divergence
- Catastrophic forgetting
- Unstable training

**Too low** (< 1e-7 for 7B):
- Very slow convergence
- May not finish in time
- Undertraining

**Optimal** (3e-7 to 5e-7 for 7B):
- Stable convergence
- Good final performance
- Efficient training

### Config Examples

**Mistral 7B (general)**:
```yaml
learning_rate: 5e-7
num_train_epochs: 1
warmup_ratio: 0.1
lr_scheduler_type: cosine
```

**Llama 3 8B (reasoning)**:
```yaml
learning_rate: 3e-7
num_train_epochs: 1
warmup_ratio: 0.1
lr_scheduler_type: cosine
```

**Gemma 2 9B (creative)**:
```yaml
learning_rate: 1e-6
num_train_epochs: 1
warmup_ratio: 0.1
lr_scheduler_type: linear
```

## Beta (β)

### Recommended Values

**Range**: 2.0 to 10.0 (much higher than DPO's 0.01-0.1)

**By preference strength**:
| Beta | Preference Strength | Use Case |
|------|-------------------|----------|
| 1.0-2.0 | Weak | Subtle preferences |
| 2.0-5.0 | **Standard** | General alignment |
| 5.0-10.0 | Strong | Clear preferences |

**Default**: 2.0 to 2.5

### Why Beta Matters

**Low beta** (< 2.0):
- Weak reward signal
- Slow preference learning
- May underfit

**High beta** (> 10.0):
- Very strong reward signal
- Risk of overfitting
- May ignore weak preferences

**Optimal** (2.0-5.0):
- Balanced reward scaling
- Stable training
- Good generalization

### Interaction with Gamma

**Beta and gamma together**:
```
Target margin in reward space = gamma
Target margin in logit space = gamma / beta
```

**Example**:
```yaml
beta: 2.0
gamma_beta_ratio: 0.5
# Effective gamma = 2.0 * 0.5 = 1.0
```

### Config Examples

**Weak preferences**:
```yaml
beta: 2.0
gamma_beta_ratio: 0.3  # Small margin
```

**Standard**:
```yaml
beta: 2.5
gamma_beta_ratio: 0.5  # Default
```

**Strong preferences**:
```yaml
beta: 5.0
gamma_beta_ratio: 0.7  # Larger margin
```

## Gamma-Beta Ratio (γ/β)

### Recommended Values

**Range**: 0.0 to 1.0

**By scenario**:
| Ratio | Margin | Use Case |
|-------|--------|----------|
| 0.0-0.3 | Small | Weak preference data |
| 0.4-0.6 | **Standard** | General use |
| 0.7-1.0 | Large | Very clear preferences |

**Default**: 0.5

### Why Gamma Matters

**Low gamma** (< 0.3):
- Small target margin
- Less aggressive alignment
- More conservative

**High gamma** (> 0.7):
- Large target margin
- Stronger alignment
- More aggressive

**Optimal** (0.4-0.6):
- Balanced margin
- Stable training
- Good alignment

### Mathematical Meaning

**In loss function**:
```python
logits = pi_logratios - gamma_beta_ratio
loss = -log(sigmoid(beta * logits))
```

**Interpretation**:
- gamma_beta_ratio shifts the decision boundary
- Higher ratio = requires larger log prob difference
- Controls how "clear" preferences must be

### Config Examples

**Noisy preferences**:
```yaml
gamma_beta_ratio: 0.3  # Smaller margin, more tolerant
```

**Standard**:
```yaml
gamma_beta_ratio: 0.5  # Default
```

**High-quality preferences**:
```yaml
gamma_beta_ratio: 0.8  # Larger margin, stricter
```

## SFT Weight

### Recommended Values

**Range**: 0.0 to 1.0

**By model type**:
| Model Type | SFT Weight | Reason |
|------------|-----------|--------|
| Base model | 0.0 | No prior capabilities |
| **Instruct model** | 0.05-0.1 | Preserve instruction following |
| Chat model | 0.1-0.2 | Preserve conversational skills |

**Default**: 0.0 (no SFT regularization)

### Why SFT Weight Matters

**Zero SFT** (0.0):
- Pure preference optimization
- May forget capabilities
- Standard for base models

**Low SFT** (0.05-0.1):
- Balanced approach
- **Recommended for instruct models**
- Slight capability preservation

**High SFT** (> 0.2):
- Strong capability preservation
- Weaker preference alignment
- May reduce alignment gains

### Trade-off

```
Total Loss = SimPO Loss + (sft_weight * SFT Loss)
```

**Example**:
```yaml
sft_weight: 0.1
# 90% preference optimization + 10% capability preservation
```

### Config Examples

**Base model (no SFT)**:
```yaml
model_name_or_path: mistralai/Mistral-7B-v0.1
sft_weight: 0.0
```

**Instruct model (light SFT)**:
```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
sft_weight: 0.1
```

**Chat model (moderate SFT)**:
```yaml
model_name_or_path: HuggingFaceH4/zephyr-7b-beta
sft_weight: 0.2
```

## Model-Size-Specific Recommendations

### 7B Models (Mistral, Llama 3)

**Standard config**:
```yaml
learning_rate: 5e-7
beta: 2.0
gamma_beta_ratio: 0.5
sft_weight: 0.0  # 0.1 if instruct model
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
```

### 8B-13B Models

**Standard config**:
```yaml
learning_rate: 3e-7
beta: 2.5
gamma_beta_ratio: 0.5
sft_weight: 0.1  # If instruct
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

### 70B Models

**Standard config**:
```yaml
learning_rate: 1e-7
beta: 2.0
gamma_beta_ratio: 0.5
sft_weight: 0.05
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

## Batch Size & Gradient Accumulation

### Effective Batch Size

```
Effective Batch Size = per_device_batch_size * num_gpus * grad_accum_steps
```

**Recommended effective batch sizes**:
- 7B: 128-256
- 13B: 64-128
- 70B: 32-64

### Config Examples

**Single GPU (A100 40GB)**:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 128  # Effective batch = 128
```

**4 GPUs (A100 40GB)**:
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 16  # Effective batch = 2*4*16 = 128
```

**8 GPUs (A100 80GB)**:
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8  # Effective batch = 2*8*8 = 128
```

## Loss Type

### Sigmoid vs Hinge

**Sigmoid** (default, recommended):
```yaml
loss_type: sigmoid
label_smoothing: 0.0
```

**Hinge** (experimental):
```yaml
loss_type: hinge
# No label smoothing for hinge
```

**When to use hinge**:
- Margin-based tasks
- SVM-style optimization
- Experimental purposes

**Generally**: Stick with sigmoid

## Tuning Guide

### Step 1: Start with Defaults

```yaml
learning_rate: 5e-7  # For 7B
beta: 2.0
gamma_beta_ratio: 0.5
sft_weight: 0.0  # 0.1 if instruct
loss_type: sigmoid
```

### Step 2: Monitor Training

**Check every 100 steps**:
- Loss curve (should decrease smoothly)
- Reward margin (should increase)
- Chosen/rejected logps (should separate)

### Step 3: Adjust if Needed

**If loss diverges**:
```yaml
learning_rate: 3e-7  # Reduce from 5e-7
beta: 1.0           # Reduce from 2.0
```

**If loss plateaus early**:
```yaml
learning_rate: 1e-6  # Increase from 5e-7
beta: 5.0           # Increase from 2.0
```

**If model forgets**:
```yaml
sft_weight: 0.2  # Increase from 0.0
```

## Complete Example Configs

### Mistral 7B Base (Standard)

```yaml
model_name_or_path: mistralai/Mistral-7B-v0.1
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0

learning_rate: 5e-7
beta: 2.0
gamma_beta_ratio: 0.5
loss_type: sigmoid
sft_weight: 0.0

num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
warmup_ratio: 0.1
lr_scheduler_type: cosine

bf16: true
gradient_checkpointing: true
```

### Llama 3 8B Instruct (Reasoning)

```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
dataset_mixer:
  argilla/distilabel-math-preference-dpo: 1.0

learning_rate: 3e-7
beta: 5.0
gamma_beta_ratio: 0.7
loss_type: sigmoid
sft_weight: 0.1

num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
warmup_ratio: 0.1
lr_scheduler_type: cosine
```

## References

- SimPO paper: https://arxiv.org/abs/2405.14734
- Alignment Handbook: https://github.com/huggingface/alignment-handbook
