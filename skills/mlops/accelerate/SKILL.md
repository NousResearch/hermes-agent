---
name: huggingface-accelerate
description: Simplest distributed training API. 4 lines to add distributed support to any PyTorch script. Unified API for DeepSpeed/FSDP/Megatron/DDP. Automatic device placement, mixed precision (FP16/BF16/FP8). Interactive config, single launch command. HuggingFace ecosystem standard.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Distributed Training, HuggingFace, Accelerate, DeepSpeed, FSDP, Mixed Precision, PyTorch, DDP, Unified API, Simple]
dependencies: [accelerate, torch, transformers]
---

# HuggingFace Accelerate - Unified Distributed Training

## Quick start

Accelerate simplifies distributed training to 4 lines of code.

**Installation**:
```bash
pip install accelerate
```

**Convert PyTorch script** (4 lines):
```python
import torch
+ from accelerate import Accelerator

+ accelerator = Accelerator()

  model = torch.nn.Transformer()
  optimizer = torch.optim.Adam(model.parameters())
  dataloader = torch.utils.data.DataLoader(dataset)

+ model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

  for batch in dataloader:
      optimizer.zero_grad()
      loss = model(batch)
-     loss.backward()
+     accelerator.backward(loss)
      optimizer.step()
```

**Run** (single command):
```bash
accelerate launch train.py
```

## Common workflows

### Workflow 1: From single GPU to multi-GPU

**Original script**:
```python
# train.py
import torch

model = torch.nn.Linear(10, 2).to('cuda')
optimizer = torch.optim.Adam(model.parameters())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for epoch in range(10):
    for batch in dataloader:
        batch = batch.to('cuda')
        optimizer.zero_grad()
        loss = model(batch).mean()
        loss.backward()
        optimizer.step()
```

**With Accelerate** (4 lines added):
```python
# train.py
import torch
from accelerate import Accelerator  # +1

accelerator = Accelerator()  # +2

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)  # +3

for epoch in range(10):
    for batch in dataloader:
        # No .to('cuda') needed - automatic!
        optimizer.zero_grad()
        loss = model(batch).mean()
        accelerator.backward(loss)  # +4
        optimizer.step()
```

**Configure** (interactive):
```bash
accelerate config
```

**Questions**:
- Which machine? (single/multi GPU/TPU/CPU)
- How many machines? (1)
- Mixed precision? (no/fp16/bf16/fp8)
- DeepSpeed? (no/yes)

**Launch** (works on any setup):
```bash
# Single GPU
accelerate launch train.py

# Multi-GPU (8 GPUs)
accelerate launch --multi_gpu --num_processes 8 train.py

# Multi-node
accelerate launch --multi_gpu --num_processes 16 \
  --num_machines 2 --machine_rank 0 \
  --main_process_ip $MASTER_ADDR \
  train.py
```

### Workflow 2: Mixed precision training

**Enable FP16/BF16**:
```python
from accelerate import Accelerator

# FP16 (with gradient scaling)
accelerator = Accelerator(mixed_precision='fp16')

# BF16 (no scaling, more stable)
accelerator = Accelerator(mixed_precision='bf16')

# FP8 (H100+)
accelerator = Accelerator(mixed_precision='fp8')

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Everything else is automatic!
for batch in dataloader:
    with accelerator.autocast():  # Optional, done automatically
        loss = model(batch)
    accelerator.backward(loss)
```

### Workflow 3: DeepSpeed ZeRO integration

**Enable DeepSpeed ZeRO-2**:
```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision='bf16',
    deepspeed_plugin={
        "zero_stage": 2,  # ZeRO-2
        "offload_optimizer": False,
        "gradient_accumulation_steps": 4
    }
)

# Same code as before!
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

**Or via config**:
```bash
accelerate config
# Select: DeepSpeed → ZeRO-2
```

**deepspeed_config.json**:
```json
{
    "fp16": {"enabled": false},
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    }
}
```

**Launch**:
```bash
accelerate launch --config_file deepspeed_config.json train.py
```

### Workflow 4: FSDP (Fully Sharded Data Parallel)

**Enable FSDP**:
```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",  # ZeRO-3 equivalent
    auto_wrap_policy="TRANSFORMER_AUTO_WRAP",
    cpu_offload=False
)

accelerator = Accelerator(
    mixed_precision='bf16',
    fsdp_plugin=fsdp_plugin
)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

**Or via config**:
```bash
accelerate config
# Select: FSDP → Full Shard → No CPU Offload
```

### Workflow 5: Gradient accumulation

**Accumulate gradients**:
```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=4)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):  # Handles accumulation
        optimizer.zero_grad()
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
```

**Effective batch size**: `batch_size * num_gpus * gradient_accumulation_steps`

## When to use vs alternatives

**Use Accelerate when**:
- Want simplest distributed training
- Need single script for any hardware
- Use HuggingFace ecosystem
- Want flexibility (DDP/DeepSpeed/FSDP/Megatron)
- Need quick prototyping

**Key advantages**:
- **4 lines**: Minimal code changes
- **Unified API**: Same code for DDP, DeepSpeed, FSDP, Megatron
- **Automatic**: Device placement, mixed precision, sharding
- **Interactive config**: No manual launcher setup
- **Single launch**: Works everywhere

**Use alternatives instead**:
- **PyTorch Lightning**: Need callbacks, high-level abstractions
- **Ray Train**: Multi-node orchestration, hyperparameter tuning
- **DeepSpeed**: Direct API control, advanced features
- **Raw DDP**: Maximum control, minimal abstraction

## Common issues

**Issue: Wrong device placement**

Don't manually move to device:
```python
# WRONG
batch = batch.to('cuda')

# CORRECT
# Accelerate handles it automatically after prepare()
```

**Issue: Gradient accumulation not working**

Use context manager:
```python
# CORRECT
with accelerator.accumulate(model):
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
```

**Issue: Checkpointing in distributed**

Use accelerator methods:
```python
# Save only on main process
if accelerator.is_main_process:
    accelerator.save_state('checkpoint/')

# Load on all processes
accelerator.load_state('checkpoint/')
```

**Issue: Different results with FSDP**

Ensure same random seed:
```python
from accelerate.utils import set_seed
set_seed(42)
```

## Advanced topics

**Megatron integration**: See [references/megatron-integration.md](references/megatron-integration.md) for tensor parallelism, pipeline parallelism, and sequence parallelism setup.

**Custom plugins**: See [references/custom-plugins.md](references/custom-plugins.md) for creating custom distributed plugins and advanced configuration.

**Performance tuning**: See [references/performance.md](references/performance.md) for profiling, memory optimization, and best practices.

## Hardware requirements

- **CPU**: Works (slow)
- **Single GPU**: Works
- **Multi-GPU**: DDP (default), DeepSpeed, or FSDP
- **Multi-node**: DDP, DeepSpeed, FSDP, Megatron
- **TPU**: Supported
- **Apple MPS**: Supported

**Launcher requirements**:
- **DDP**: `torch.distributed.run` (built-in)
- **DeepSpeed**: `deepspeed` (pip install deepspeed)
- **FSDP**: PyTorch 1.12+ (built-in)
- **Megatron**: Custom setup

## Resources

- Docs: https://huggingface.co/docs/accelerate
- GitHub: https://github.com/huggingface/accelerate
- Version: 1.11.0+
- Tutorial: "Accelerate your scripts"
- Examples: https://github.com/huggingface/accelerate/tree/main/examples
- Used by: HuggingFace Transformers, TRL, PEFT, all HF libraries



