# Megatron Integration with Accelerate

## Overview

Accelerate supports Megatron-LM for massive model training with tensor parallelism and pipeline parallelism.

**Megatron capabilities**:
- **Tensor Parallelism (TP)**: Split layers across GPUs
- **Pipeline Parallelism (PP)**: Split model depth across GPUs
- **Data Parallelism (DP)**: Replicate model across GPU groups
- **Sequence Parallelism**: Split sequences for long contexts

## Setup

### Install Megatron-LM

```bash
# Clone Megatron-LM repository
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .

# Install Apex (NVIDIA optimizations)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

### Accelerate Configuration

```bash
accelerate config
```

**Questions**:
```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> Multi-GPU

How many different machines will you use?
> 1

Do you want to use DeepSpeed/FSDP?
> No

Do you want to use Megatron-LM?
> Yes

What is the Tensor Parallelism degree? [1-8]
> 2

Do you want to enable Sequence Parallelism?
> No

What is the Pipeline Parallelism degree? [1-8]
> 2

What is the Data Parallelism degree? [1-8]
> 2

Where to perform activation checkpointing? ['SELECTIVE', 'FULL', 'NONE']
> SELECTIVE

Where to perform activation partitioning? ['SEQUENTIAL', 'UNIFORM']
> SEQUENTIAL
```

**Generated config** (`~/.cache/huggingface/accelerate/default_config.yaml`):
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MEGATRON_LM
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
megatron_lm_config:
  megatron_lm_gradient_clipping: 1.0
  megatron_lm_learning_rate_decay_iters: 320000
  megatron_lm_num_micro_batches: 1
  megatron_lm_pp_degree: 2
  megatron_lm_recompute_activations: true
  megatron_lm_sequence_parallelism: false
  megatron_lm_tp_degree: 2
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Parallelism Strategies

### Tensor Parallelism (TP)

**Splits each transformer layer across GPUs**:

```python
# Layer split across 2 GPUs
# GPU 0: First half of attention heads
# GPU 1: Second half of attention heads

# Each GPU computes partial outputs
# All-reduce combines results
```

**TP degree recommendations**:
- **TP=1**: No tensor parallelism (single GPU per layer)
- **TP=2**: 2 GPUs per layer (good for 7-13B models)
- **TP=4**: 4 GPUs per layer (good for 20-40B models)
- **TP=8**: 8 GPUs per layer (good for 70B+ models)

**Benefits**:
- Reduces memory per GPU
- All-reduce communication (fast)

**Drawbacks**:
- Requires fast inter-GPU bandwidth (NVLink)
- Communication overhead per layer

### Pipeline Parallelism (PP)

**Splits model depth across GPUs**:

```python
# 12-layer model, PP=4
# GPU 0: Layers 0-2
# GPU 1: Layers 3-5
# GPU 2: Layers 6-8
# GPU 3: Layers 9-11
```

**PP degree recommendations**:
- **PP=1**: No pipeline parallelism
- **PP=2**: 2 pipeline stages (good for 20-40B models)
- **PP=4**: 4 pipeline stages (good for 70B+ models)
- **PP=8**: 8 pipeline stages (good for 175B+ models)

**Benefits**:
- Linear memory reduction (4× PP = 4× less memory)
- Works across nodes (slower interconnect OK)

**Drawbacks**:
- Pipeline bubbles (idle time)
- Requires micro-batching

### Data Parallelism (DP)

**Replicates model across GPU groups**:

```python
# 8 GPUs, TP=2, PP=2, DP=2
# Group 0 (GPUs 0-3): Full model replica
# Group 1 (GPUs 4-7): Full model replica
```

**DP degree**:
- `DP = total_gpus / (TP × PP)`
- Example: 8 GPUs, TP=2, PP=2 → DP=2

**Benefits**:
- Increases throughput
- Scales batch size

### Sequence Parallelism

**Splits long sequences across GPUs** (extends TP):

```python
# 8K sequence, TP=2, Sequence Parallel=True
# GPU 0: Tokens 0-4095
# GPU 1: Tokens 4096-8191
```

**Benefits**:
- Enables very long sequences (100K+ tokens)
- Reduces activation memory

**Requirements**:
- Must use with TP > 1
- RoPE/ALiBi position encodings work best

## Accelerate Code Example

### Basic Setup

```python
from accelerate import Accelerator
from accelerate.utils import MegatronLMPlugin

# Configure Megatron
megatron_plugin = MegatronLMPlugin(
    tp_degree=2,              # Tensor parallelism degree
    pp_degree=2,              # Pipeline parallelism degree
    num_micro_batches=4,      # Micro-batches for pipeline
    gradient_clipping=1.0,    # Gradient clipping value
    sequence_parallelism=False,  # Enable sequence parallelism
    recompute_activations=True,  # Activation checkpointing
    use_distributed_optimizer=True,  # Distributed optimizer
    custom_prepare_model_function=None,  # Custom model prep
)

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision='bf16',
    megatron_lm_plugin=megatron_plugin
)

# Prepare model and optimizer
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# Training loop (same as DDP!)
for batch in train_dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
```

### Full Training Script

```python
import torch
from accelerate import Accelerator
from accelerate.utils import MegatronLMPlugin
from transformers import GPT2Config, GPT2LMHeadModel

def main():
    # Megatron configuration
    megatron_plugin = MegatronLMPlugin(
        tp_degree=2,
        pp_degree=2,
        num_micro_batches=4,
        gradient_clipping=1.0,
    )

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=8,
        megatron_lm_plugin=megatron_plugin
    )

    # Model
    config = GPT2Config(
        n_layer=24,
        n_head=16,
        n_embd=1024,
    )
    model = GPT2LMHeadModel(config)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

    # Prepare
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

        # Save checkpoint
        accelerator.wait_for_everyone()
        accelerator.save_state(f'checkpoint-epoch-{epoch}')

if __name__ == '__main__':
    main()
```

### Launch Command

```bash
# 8 GPUs, TP=2, PP=2, DP=2
accelerate launch --multi_gpu --num_processes 8 train.py

# Multi-node (2 nodes, 8 GPUs each)
# Node 0
accelerate launch --multi_gpu --num_processes 16 \
  --num_machines 2 --machine_rank 0 \
  --main_process_ip $MASTER_ADDR \
  --main_process_port 29500 \
  train.py

# Node 1
accelerate launch --multi_gpu --num_processes 16 \
  --num_machines 2 --machine_rank 1 \
  --main_process_ip $MASTER_ADDR \
  --main_process_port 29500 \
  train.py
```

## Activation Checkpointing

**Reduces memory by recomputing activations**:

```python
megatron_plugin = MegatronLMPlugin(
    recompute_activations=True,      # Enable checkpointing
    checkpoint_num_layers=1,         # Checkpoint every N layers
    distribute_checkpointed_activations=True,  # Distribute across TP
    partition_activations=True,      # Partition in PP
    check_for_nan_in_loss_and_grad=True,  # Stability check
)
```

**Strategies**:
- `SELECTIVE`: Checkpoint transformer blocks only
- `FULL`: Checkpoint all layers
- `NONE`: No checkpointing

**Memory savings**: 30-50% with 10-15% slowdown

## Distributed Optimizer

**Shards optimizer state across DP ranks**:

```python
megatron_plugin = MegatronLMPlugin(
    use_distributed_optimizer=True,  # Enable sharded optimizer
)
```

**Benefits**:
- Reduces optimizer memory by DP degree
- Example: DP=4 → 4× less optimizer memory per GPU

**Compatible with**:
- AdamW, Adam, SGD
- Mixed precision training

## Performance Tuning

### Micro-Batch Size

```python
# Pipeline parallelism requires micro-batching
megatron_plugin = MegatronLMPlugin(
    pp_degree=4,
    num_micro_batches=16,  # 16 micro-batches per pipeline
)

# Effective batch = num_micro_batches × micro_batch_size × DP
# Example: 16 × 2 × 4 = 128
```

**Recommendations**:
- More micro-batches → less pipeline bubble
- Typical: 4-16 micro-batches

### Sequence Length

```python
# For long sequences, enable sequence parallelism
megatron_plugin = MegatronLMPlugin(
    tp_degree=4,
    sequence_parallelism=True,  # Required: TP > 1
)

# Enables sequences up to TP × normal limit
# Example: TP=4, 8K normal → 32K with sequence parallel
```

### GPU Topology

**NVLink required for TP**:
```bash
# Check NVLink topology
nvidia-smi topo -m

# Good topology (NVLink between all GPUs)
# GPU0 - GPU1: NV12 (fast)
# GPU0 - GPU2: NV12 (fast)

# Bad topology (PCIe only)
# GPU0 - GPU4: PHB (slow, avoid TP across these)
```

**Recommendations**:
- **TP**: Within same node (NVLink)
- **PP**: Across nodes (slower interconnect OK)
- **DP**: Any topology

## Model Size Guidelines

| Model Size | GPUs | TP | PP | DP | Micro-Batches |
|------------|------|----|----|----|--------------|
| 7B | 8 | 1 | 1 | 8 | 1 |
| 13B | 8 | 2 | 1 | 4 | 1 |
| 20B | 16 | 4 | 1 | 4 | 1 |
| 40B | 32 | 4 | 2 | 4 | 4 |
| 70B | 64 | 8 | 2 | 4 | 8 |
| 175B | 128 | 8 | 4 | 4 | 16 |

**Assumptions**: BF16, 2K sequence length, A100 80GB

## Checkpointing

### Save Checkpoint

```python
# Save full model state
accelerator.save_state('checkpoint-1000')

# Megatron saves separate files per rank
# checkpoint-1000/
#   pytorch_model_tp_0_pp_0.bin
#   pytorch_model_tp_0_pp_1.bin
#   pytorch_model_tp_1_pp_0.bin
#   pytorch_model_tp_1_pp_1.bin
#   optimizer_tp_0_pp_0.bin
#   ...
```

### Load Checkpoint

```python
# Resume training
accelerator.load_state('checkpoint-1000')

# Automatically loads correct shard per rank
```

### Convert to Standard PyTorch

```bash
# Merge Megatron checkpoint to single file
python merge_megatron_checkpoint.py \
  --checkpoint-dir checkpoint-1000 \
  --output pytorch_model.bin
```

## Common Issues

### Issue: OOM with Pipeline Parallelism

**Solution**: Increase micro-batches
```python
megatron_plugin = MegatronLMPlugin(
    pp_degree=4,
    num_micro_batches=16,  # Increase from 4
)
```

### Issue: Slow Training

**Check 1**: Pipeline bubbles (PP too high)
```python
# Reduce PP, increase TP
tp_degree=4  # Increase
pp_degree=2  # Decrease
```

**Check 2**: Micro-batch size too small
```python
num_micro_batches=8  # Increase
```

### Issue: NVLink Not Detected

```bash
# Verify NVLink
nvidia-smi nvlink -s

# If no NVLink, avoid TP > 1
# Use PP or DP instead
```

## Resources

- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Accelerate Megatron docs: https://huggingface.co/docs/accelerate/usage_guides/megatron_lm
- Paper: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
- NVIDIA Apex: https://github.com/NVIDIA/apex
