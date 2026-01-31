---
name: simpo-training
description: Simple Preference Optimization for LLM alignment. Reference-free alternative to DPO with better performance (+6.4 points on AlpacaEval 2.0). No reference model needed, more efficient than DPO. Use for preference alignment when want simpler, faster training than DPO/PPO.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Post-Training, SimPO, Preference Optimization, Alignment, DPO Alternative, Reference-Free, LLM Alignment, Efficient Training]
dependencies: [torch, transformers, datasets, trl, accelerate]
---

# SimPO - Simple Preference Optimization

## Quick start

SimPO is a reference-free preference optimization method that outperforms DPO without needing a reference model.

**Installation**:
```bash
# Create environment
conda create -n simpo python=3.10 && conda activate simpo

# Install PyTorch 2.2.2
# Visit: https://pytorch.org/get-started/locally/

# Install alignment-handbook
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
python -m pip install .

# Install Flash Attention 2
python -m pip install flash-attn --no-build-isolation
```

**Training** (Mistral 7B):
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py \
  training_configs/mistral-7b-base-simpo.yaml
```

## Common workflows

### Workflow 1: Train from base model (Mistral 7B)

**Config** (`mistral-7b-base-simpo.yaml`):
```yaml
# Model
model_name_or_path: mistralai/Mistral-7B-v0.1
torch_dtype: bfloat16

# Dataset
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
dataset_splits:
  - train_prefs
  - test_prefs

# SimPO hyperparameters
beta: 2.0                  # Reward scaling (2.0-10.0)
gamma_beta_ratio: 0.5       # Target margin (0-1)
loss_type: sigmoid          # sigmoid or hinge
sft_weight: 0.0             # Optional SFT regularization

# Training
learning_rate: 5e-7         # Critical: 3e-7 to 1e-6
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Output
output_dir: ./outputs/mistral-7b-simpo
```

**Launch training**:
```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/mistral-7b-base-simpo.yaml
```

### Workflow 2: Fine-tune instruct model (Llama 3 8B)

**Config** (`llama3-8b-instruct-simpo.yaml`):
```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

dataset_mixer:
  argilla/ultrafeedback-binarized-preferences-cleaned: 1.0

beta: 2.5
gamma_beta_ratio: 0.5
learning_rate: 5e-7
sft_weight: 0.1             # Add SFT loss to preserve capabilities

num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
output_dir: ./outputs/llama3-8b-simpo
```

**Launch**:
```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/llama3-8b-instruct-simpo.yaml
```

### Workflow 3: Reasoning-intensive tasks (lower LR)

**For math/code tasks**:
```yaml
model_name_or_path: deepseek-ai/deepseek-math-7b-base

dataset_mixer:
  argilla/distilabel-math-preference-dpo: 1.0

beta: 5.0                   # Higher for stronger signal
gamma_beta_ratio: 0.7       # Larger margin
learning_rate: 3e-7         # Lower LR for reasoning
sft_weight: 0.0

num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

## When to use vs alternatives

**Use SimPO when**:
- Want simpler training than DPO (no reference model)
- Have preference data (chosen/rejected pairs)
- Need better performance than DPO
- Limited compute resources
- Single-node training sufficient

**Algorithm selection**:
- **SimPO**: Simplest, best performance, no reference model
- **DPO**: Need reference model baseline, more conservative
- **PPO**: Maximum control, need reward model, complex setup
- **GRPO**: Memory-efficient RL, no critic

**Use alternatives instead**:
- **OpenRLHF**: Multi-node distributed training, PPO/GRPO
- **TRL**: Need multiple methods in one framework
- **DPO**: Established baseline comparison

## Common issues

**Issue: Loss divergence**

Reduce learning rate:
```yaml
learning_rate: 3e-7  # Reduce from 5e-7
```

Reduce beta:
```yaml
beta: 1.0  # Reduce from 2.0
```

**Issue: Model forgets capabilities**

Add SFT regularization:
```yaml
sft_weight: 0.1  # Add SFT loss component
```

**Issue: Poor preference separation**

Increase beta and margin:
```yaml
beta: 5.0            # Increase from 2.0
gamma_beta_ratio: 0.8  # Increase from 0.5
```

**Issue: OOM during training**

Reduce batch size:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Maintain effective batch
```

Enable gradient checkpointing:
```yaml
gradient_checkpointing: true
```

## Advanced topics

**Loss functions**: See [references/loss-functions.md](references/loss-functions.md) for sigmoid vs hinge loss, mathematical formulations, and when to use each.

**Hyperparameter tuning**: See [references/hyperparameters.md](references/hyperparameters.md) for beta, gamma, learning rate selection guide, and model-size-specific recommendations.

**Dataset preparation**: See [references/datasets.md](references/datasets.md) for preference data formats, quality filtering, and custom dataset creation.

## Hardware requirements

- **GPU**: NVIDIA A100/H100 recommended
- **VRAM**:
  - 7B model: 1× A100 40GB (DeepSpeed ZeRO-3)
  - 8B model: 2× A100 40GB
  - 70B model: 8× A100 80GB
- **Single-node**: DeepSpeed ZeRO-3 sufficient
- **Mixed precision**: BF16 recommended

**Memory optimization**:
- DeepSpeed ZeRO-3 (default config)
- Gradient checkpointing
- Flash Attention 2

## Resources

- Paper: https://arxiv.org/abs/2405.14734 (NeurIPS 2024)
- GitHub: https://github.com/princeton-nlp/SimPO
- Models: https://huggingface.co/princeton-nlp
- Alignment Handbook: https://github.com/huggingface/alignment-handbook



