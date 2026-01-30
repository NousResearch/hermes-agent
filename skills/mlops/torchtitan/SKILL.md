---
name: distributed-llm-pretraining-torchtitan
description: Provides PyTorch-native distributed LLM pretraining using torchtitan with 4D parallelism (FSDP2, TP, PP, CP). Use when pretraining Llama 3.1, DeepSeek V3, or custom models at scale from 8 to 512+ GPUs with Float8, torch.compile, and distributed checkpointing.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Model Architecture, Distributed Training, TorchTitan, FSDP2, Tensor Parallel, Pipeline Parallel, Context Parallel, Float8, Llama, Pretraining]
dependencies: [torch>=2.6.0, torchtitan>=0.2.0, torchao>=0.5.0]
---

# TorchTitan - PyTorch Native Distributed LLM Pretraining

## Quick start

TorchTitan is PyTorch's official platform for large-scale LLM pretraining with composable 4D parallelism (FSDP2, TP, PP, CP), achieving 65%+ speedups over baselines on H100 GPUs.

**Installation**:
```bash
# From PyPI (stable)
pip install torchtitan

# From source (latest features, requires PyTorch nightly)
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
```

**Download tokenizer**:
```bash
# Get HF token from https://huggingface.co/settings/tokens
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

**Start training on 8 GPUs**:
```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
```

## Common workflows

### Workflow 1: Pretrain Llama 3.1 8B on single node

Copy this checklist:

```
Single Node Pretraining:
- [ ] Step 1: Download tokenizer
- [ ] Step 2: Configure training
- [ ] Step 3: Launch training
- [ ] Step 4: Monitor and checkpoint
```

**Step 1: Download tokenizer**

```bash
python scripts/download_hf_assets.py \
  --repo_id meta-llama/Llama-3.1-8B \
  --assets tokenizer \
  --hf_token=YOUR_HF_TOKEN
```

**Step 2: Configure training**

Edit or create a TOML config file:

```toml
# llama3_8b_custom.toml
[job]
dump_folder = "./outputs"
description = "Llama 3.1 8B training"

[model]
name = "llama3"
flavor = "8B"
hf_assets_path = "./assets/hf/Llama-3.1-8B"

[optimizer]
name = "AdamW"
lr = 3e-4

[lr_scheduler]
warmup_steps = 200

[training]
local_batch_size = 2
seq_len = 8192
max_norm = 1.0
steps = 1000
dataset = "c4"

[parallelism]
data_parallel_shard_degree = -1  # Use all GPUs for FSDP

[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"

[checkpoint]
enable = true
folder = "checkpoint"
interval = 500
```

**Step 3: Launch training**

```bash
# 8 GPUs on single node
CONFIG_FILE="./llama3_8b_custom.toml" ./run_train.sh

# Or explicitly with torchrun
torchrun --nproc_per_node=8 \
  -m torchtitan.train \
  --job.config_file ./llama3_8b_custom.toml
```

**Step 4: Monitor and checkpoint**

TensorBoard logs are saved to `./outputs/tb/`:
```bash
tensorboard --logdir ./outputs/tb
```

### Workflow 2: Multi-node training with SLURM

```
Multi-Node Training:
- [ ] Step 1: Configure parallelism for scale
- [ ] Step 2: Set up SLURM script
- [ ] Step 3: Submit job
- [ ] Step 4: Resume from checkpoint
```

**Step 1: Configure parallelism for scale**

For 70B model on 256 GPUs (32 nodes):
```toml
[parallelism]
data_parallel_shard_degree = 32  # FSDP across 32 ranks
tensor_parallel_degree = 8        # TP within node
pipeline_parallel_degree = 1      # No PP for 70B
context_parallel_degree = 1       # Increase for long sequences
```

**Step 2: Set up SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=llama70b
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

srun torchrun \
  --nnodes=32 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m torchtitan.train \
  --job.config_file ./llama3_70b.toml
```

**Step 3: Submit job**

```bash
sbatch multinode_trainer.slurm
```

**Step 4: Resume from checkpoint**

Training auto-resumes if checkpoint exists in configured folder.

### Workflow 3: Enable Float8 training for H100s

Float8 provides 30-50% speedup on H100 GPUs.

```
Float8 Training:
- [ ] Step 1: Install torchao
- [ ] Step 2: Configure Float8
- [ ] Step 3: Launch with compile
```

**Step 1: Install torchao**

```bash
USE_CPP=0 pip install git+https://github.com/pytorch/ao.git
```

**Step 2: Configure Float8**

Add to your TOML config:
```toml
[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["output"]  # Exclude output layer

[compile]
enable = true
components = ["model", "loss"]
```

**Step 3: Launch with compile**

```bash
CONFIG_FILE="./llama3_8b.toml" ./run_train.sh \
  --model.converters="quantize.linear.float8" \
  --quantize.linear.float8.enable_fsdp_float8_all_gather \
  --compile.enable
```

### Workflow 4: 4D parallelism for 405B models

```
4D Parallelism (FSDP + TP + PP + CP):
- [ ] Step 1: Create seed checkpoint
- [ ] Step 2: Configure 4D parallelism
- [ ] Step 3: Launch on 512 GPUs
```

**Step 1: Create seed checkpoint**

Required for consistent initialization across PP stages:
```bash
NGPU=1 CONFIG_FILE=./llama3_405b.toml ./run_train.sh \
  --checkpoint.enable \
  --checkpoint.create_seed_checkpoint \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 1
```

**Step 2: Configure 4D parallelism**

```toml
[parallelism]
data_parallel_shard_degree = 8   # FSDP
tensor_parallel_degree = 8       # TP within node
pipeline_parallel_degree = 8     # PP across nodes
context_parallel_degree = 1      # CP for long sequences

[training]
local_batch_size = 32
seq_len = 8192
```

**Step 3: Launch on 512 GPUs**

```bash
# 64 nodes x 8 GPUs = 512 GPUs
srun torchrun --nnodes=64 --nproc_per_node=8 \
  -m torchtitan.train \
  --job.config_file ./llama3_405b.toml
```

## When to use vs alternatives

**Use TorchTitan when:**
- Pretraining LLMs from scratch (8B to 405B+)
- Need PyTorch-native solution without third-party dependencies
- Require composable 4D parallelism (FSDP2, TP, PP, CP)
- Training on H100s with Float8 support
- Want interoperable checkpoints with torchtune/HuggingFace

**Use alternatives instead:**
- **Megatron-LM**: Maximum performance for NVIDIA-only deployments
- **DeepSpeed**: Broader ZeRO optimization ecosystem, inference support
- **Axolotl/TRL**: Fine-tuning rather than pretraining
- **LitGPT**: Educational, smaller-scale training

## Common issues

**Issue: Out of memory on large models**

Enable activation checkpointing and reduce batch size:
```toml
[activation_checkpoint]
mode = "full"  # Instead of "selective"

[training]
local_batch_size = 1
```

Or use gradient accumulation:
```toml
[training]
local_batch_size = 1
global_batch_size = 32  # Accumulates gradients
```

**Issue: TP causes high memory with async collectives**

Set environment variable:
```bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
```

**Issue: Float8 training not faster**

Float8 only benefits large GEMMs. Filter small layers:
```toml
[quantize.linear.float8]
filter_fqns = ["attention.wk", "attention.wv", "output", "auto_filter_small_kn"]
```

**Issue: Checkpoint loading fails after parallelism change**

Use DCP's resharding capability:
```bash
# Convert sharded checkpoint to single file
python -m torch.distributed.checkpoint.format_utils \
  dcp_to_torch checkpoint/step-1000 checkpoint.pt
```

**Issue: Pipeline parallelism initialization**

Create seed checkpoint first (see Workflow 4, Step 1).

## Supported models

| Model | Sizes | Status |
|-------|-------|--------|
| Llama 3.1 | 8B, 70B, 405B | Production |
| Llama 4 | Various | Experimental |
| DeepSeek V3 | 16B, 236B, 671B (MoE) | Experimental |
| GPT-OSS | 20B, 120B (MoE) | Experimental |
| Qwen 3 | Various | Experimental |
| Flux | Diffusion | Experimental |

## Performance benchmarks (H100)

| Model | GPUs | Parallelism | TPS/GPU | Techniques |
|-------|------|-------------|---------|------------|
| Llama 8B | 8 | FSDP | 5,762 | Baseline |
| Llama 8B | 8 | FSDP+compile+FP8 | 8,532 | +48% |
| Llama 70B | 256 | FSDP+TP+AsyncTP | 876 | 2D parallel |
| Llama 405B | 512 | FSDP+TP+PP | 128 | 3D parallel |

## Advanced topics

**FSDP2 configuration**: See [references/fsdp.md](references/fsdp.md) for detailed FSDP2 vs FSDP1 comparison and ZeRO equivalents.

**Float8 training**: See [references/float8.md](references/float8.md) for tensorwise vs rowwise scaling recipes.

**Checkpointing**: See [references/checkpoint.md](references/checkpoint.md) for HuggingFace conversion and async checkpointing.

**Adding custom models**: See [references/custom-models.md](references/custom-models.md) for TrainSpec protocol.

## Resources

- GitHub: https://github.com/pytorch/torchtitan
- Paper: https://arxiv.org/abs/2410.06511
- ICLR 2025: https://iclr.cc/virtual/2025/poster/29620
- PyTorch Forum: https://discuss.pytorch.org/c/distributed/torchtitan/44

