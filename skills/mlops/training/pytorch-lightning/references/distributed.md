# PyTorch Lightning Distributed Training

## Distributed Strategies

Lightning supports multiple distributed strategies with a single parameter change.

### 1. DDP (DistributedDataParallel)

**Default strategy for multi-GPU**:

```python
# Automatic DDP on all available GPUs
trainer = L.Trainer(accelerator='gpu', devices=4, strategy='ddp')

# Or auto-detect
trainer = L.Trainer(accelerator='gpu', devices='auto')
```

**How DDP works**:
- Replicates model on each GPU
- Each GPU processes different batch
- Gradients all-reduced across GPUs
- Model weights synchronized

**Launch**:
```bash
# Lightning handles spawning processes automatically
python train.py
```

**DDP Configuration**:
```python
from lightning.pytorch.strategies import DDPStrategy

strategy = DDPStrategy(
    find_unused_parameters=False,  # Set True if model has unused params
    gradient_as_bucket_view=True,  # Memory optimization
    static_graph=False,  # Set True if graph doesn't change
)

trainer = L.Trainer(strategy=strategy)
```

### 2. FSDP (Fully Sharded Data Parallel)

**For large models (7B+ parameters)**:

```python
from lightning.pytorch.strategies import FSDPStrategy

strategy = FSDPStrategy(
    sharding_strategy="FULL_SHARD",  # ZeRO-3 equivalent
    activation_checkpointing=None,   # Or specify layer types
    cpu_offload=False,               # CPU offload for memory
)

trainer = L.Trainer(
    accelerator='gpu',
    devices=8,
    strategy=strategy,
    precision='bf16'  # Recommended with FSDP
)

trainer.fit(model, train_loader)
```

**FSDP Sharding Strategies**:
```python
# FULL_SHARD (most memory efficient, equivalent to ZeRO-3)
strategy = FSDPStrategy(sharding_strategy="FULL_SHARD")

# SHARD_GRAD_OP (less memory efficient, equivalent to ZeRO-2)
strategy = FSDPStrategy(sharding_strategy="SHARD_GRAD_OP")

# NO_SHARD (no sharding, like DDP)
strategy = FSDPStrategy(sharding_strategy="NO_SHARD")
```

**Auto-wrap policy** (wrap transformer blocks):
```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import functools

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={GPT2Block}
)

strategy = FSDPStrategy(
    auto_wrap_policy=auto_wrap_policy,
    activation_checkpointing_policy={GPT2Block}  # Checkpoint these blocks
)
```

### 3. DeepSpeed

**For massive models (70B+ parameters)**:

```python
from lightning.pytorch.strategies import DeepSpeedStrategy

# DeepSpeed ZeRO-3 with CPU offload
strategy = DeepSpeedStrategy(
    stage=3,                       # ZeRO-3
    offload_optimizer=True,        # CPU offload optimizer
    offload_parameters=True,       # CPU offload parameters
    cpu_checkpointing=True,        # Checkpoint to CPU
)

trainer = L.Trainer(
    accelerator='gpu',
    devices=8,
    strategy=strategy,
    precision='bf16'
)

trainer.fit(model, train_loader)
```

**DeepSpeed configuration file**:
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "bf16": {
    "enabled": true
  }
}
```

**Use config file**:
```python
strategy = DeepSpeedStrategy(config='deepspeed_config.json')
trainer = L.Trainer(strategy=strategy)
```

### 4. DDP Spawn

**Windows-compatible DDP**:

```python
# Use when DDP doesn't work (e.g., Windows, Jupyter)
trainer = L.Trainer(
    accelerator='gpu',
    devices=2,
    strategy='ddp_spawn'  # Spawns new processes
)
```

**Note**: Slower than DDP due to process spawning overhead

## Multi-Node Training

### Setup Multi-Node Cluster

**Node 0 (master)**:
```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=12355
export WORLD_SIZE=16  # 2 nodes × 8 GPUs
export NODE_RANK=0

python train.py
```

**Node 1 (worker)**:
```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=12355
export WORLD_SIZE=16
export NODE_RANK=1

python train.py
```

**Training script**:
```python
trainer = L.Trainer(
    accelerator='gpu',
    devices=8,              # GPUs per node
    num_nodes=2,            # Total nodes
    strategy='ddp'
)

trainer.fit(model, train_loader)
```

### SLURM Integration

**SLURM job script**:
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

# Lightning auto-detects SLURM environment
srun python train.py
```

**Training script** (no changes needed):
```python
# Lightning automatically reads SLURM environment variables
trainer = L.Trainer(
    accelerator='gpu',
    devices=8,
    num_nodes=4,  # From SBATCH --nodes
    strategy='ddp'
)
```

### Kubernetes (KubeFlow)

**Training script**:
```python
import os

# Lightning auto-detects Kubernetes
trainer = L.Trainer(
    accelerator='gpu',
    devices=int(os.getenv('WORLD_SIZE', 1)),
    strategy='ddp'
)
```

## Mixed Precision Training

### BF16 (A100/H100)

```python
trainer = L.Trainer(
    precision='bf16',  # Or 'bf16-mixed'
    accelerator='gpu'
)
```

**Advantages**:
- No gradient scaler needed
- Same dynamic range as FP32
- 2× speedup, 50% memory reduction

### FP16 (V100, older GPUs)

```python
trainer = L.Trainer(
    precision='16-mixed',  # Or just '16'
    accelerator='gpu'
)
```

**Automatic gradient scaling** handled by Lightning

### FP8 (H100)

```python
# Requires transformer_engine
# pip install transformer-engine[pytorch]

trainer = L.Trainer(
    precision='transformer-engine',
    accelerator='gpu'
)
```

**Benefits**: 2× faster than BF16 on H100

## Gradient Accumulation

**Simulate larger batch size**:

```python
trainer = L.Trainer(
    accumulate_grad_batches=4,  # Accumulate 4 batches
    precision='bf16'
)

# Effective batch = batch_size × accumulate_grad_batches × num_gpus
# Example: 32 × 4 × 8 = 1024
```

**Dynamic accumulation**:
```python
# Accumulate more early in training
trainer = L.Trainer(
    accumulate_grad_batches={
        0: 8,   # Epochs 0-4: accumulate 8
        5: 4,   # Epochs 5-9: accumulate 4
        10: 2   # Epochs 10+: accumulate 2
    }
)
```

## Checkpointing in Distributed

### Save Checkpoint

```python
from lightning.pytorch.callbacks import ModelCheckpoint

# Only rank 0 saves by default
checkpoint = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='model-{epoch:02d}',
    save_top_k=3
)

trainer = L.Trainer(callbacks=[checkpoint], strategy='ddp')
trainer.fit(model, train_loader)
```

**Manual save**:
```python
class MyModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        # Training...
        loss = ...

        # Save every 1000 steps (only rank 0)
        if batch_idx % 1000 == 0 and self.trainer.is_global_zero:
            self.trainer.save_checkpoint(f'checkpoint_step_{batch_idx}.ckpt')

        return loss
```

### Load Checkpoint

```python
# Resume training
trainer = L.Trainer(strategy='ddp')
trainer.fit(model, train_loader, ckpt_path='checkpoints/last.ckpt')

# Load for inference
model = MyModel.load_from_checkpoint('checkpoints/best.ckpt')
model.eval()
```

## Strategy Comparison

| Strategy | Memory Efficiency | Speed | Use Case |
|----------|------------------|-------|----------|
| DDP | Low | Fast | Small models (<7B), single node |
| FSDP | High | Medium | Large models (7-70B) |
| DeepSpeed ZeRO-2 | Medium | Fast | Medium models (1-13B) |
| DeepSpeed ZeRO-3 | Very High | Slower | Massive models (70B+) |
| DDP Spawn | Low | Slow | Windows, debugging |

## Best Practices

### 1. Choose Right Strategy

```python
# Model size guide
if model_params < 1e9:  # <1B
    strategy = 'ddp'
elif model_params < 7e9:  # 1-7B
    strategy = 'ddp' or DeepSpeedStrategy(stage=2)
elif model_params < 70e9:  # 7-70B
    strategy = FSDPStrategy(sharding_strategy="FULL_SHARD")
else:  # 70B+
    strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True)

trainer = L.Trainer(strategy=strategy)
```

### 2. Avoid Sync Issues

```python
class MyModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        # WRONG: This runs on all GPUs independently
        if batch_idx % 100 == 0:
            self.log_something()  # Logged 8 times on 8 GPUs!

        # CORRECT: Use is_global_zero
        if batch_idx % 100 == 0 and self.trainer.is_global_zero:
            self.log_something()  # Logged once

        loss = ...
        return loss
```

### 3. Efficient Data Loading

```python
from torch.utils.data import DataLoader, DistributedSampler

# Lightning handles DistributedSampler automatically
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 4 workers per GPU
    pin_memory=True,
    persistent_workers=True
)

# Lightning automatically wraps with DistributedSampler in DDP
trainer.fit(model, train_loader)
```

### 4. Reduce Communication Overhead

```python
from lightning.pytorch.strategies import DDPStrategy

strategy = DDPStrategy(
    gradient_as_bucket_view=True,  # Reduce memory copies
    static_graph=True,  # If model graph doesn't change (faster)
)

trainer = L.Trainer(strategy=strategy)
```

## Common Issues

### Issue: NCCL Timeout

**Symptom**: Training hangs with `NCCL timeout` error

**Solution 1**: Increase timeout
```bash
export NCCL_TIMEOUT=3600  # 1 hour
python train.py
```

**Solution 2**: Check network
```bash
# Test inter-node communication
nvidia-smi nvlink -s

# Verify all nodes can ping each other
ping <node-2-ip>
```

### Issue: OOM with FSDP

**Solution**: Enable CPU offload
```python
strategy = FSDPStrategy(
    sharding_strategy="FULL_SHARD",
    cpu_offload=True  # Offload to CPU
)
```

### Issue: Different Results with DDP

**Cause**: Different random seeds per GPU

**Solution**: Set seed in LightningModule
```python
class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        L.seed_everything(42, workers=True)  # Same seed everywhere
```

### Issue: DeepSpeed Config Errors

**Solution**: Use Lightning's auto config
```python
strategy = DeepSpeedStrategy(
    stage=3,
    # Don't specify config file, Lightning generates automatically
)
```

## Resources

- Distributed strategies: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html
- FSDP guide: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html
- DeepSpeed: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html
- Multi-node: https://lightning.ai/docs/pytorch/stable/clouds/cluster.html
