# Accelerate Performance Tuning

## Profiling

### Basic Profiling

```python
from accelerate import Accelerator
import time

accelerator = Accelerator()

# Warmup
for _ in range(10):
    batch = next(iter(dataloader))
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

# Profile training loop
start = time.time()
total_batches = 100

for i, batch in enumerate(dataloader):
    if i >= total_batches:
        break

    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

accelerator.wait_for_everyone()  # Sync all processes
elapsed = time.time() - start

# Metrics
batches_per_sec = total_batches / elapsed
samples_per_sec = (total_batches * batch_size * accelerator.num_processes) / elapsed

print(f"Throughput: {samples_per_sec:.2f} samples/sec")
print(f"Batches/sec: {batches_per_sec:.2f}")
```

### PyTorch Profiler Integration

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Profile first 10 batches
            break

        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# Print profiling results
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20
))

# Export to Chrome tracing
prof.export_chrome_trace("trace.json")
# View at chrome://tracing
```

## Memory Optimization

### 1. Gradient Accumulation

**Problem**: Large batch size causes OOM

**Solution**: Accumulate gradients across micro-batches

```python
accelerator = Accelerator(gradient_accumulation_steps=8)

# Effective batch = batch_size × accumulation_steps × num_gpus
# Example: 4 × 8 × 8 = 256

for batch in dataloader:
    with accelerator.accumulate(model):  # Handles accumulation logic
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

**Memory savings**: 8× less activation memory (with 8 accumulation steps)

### 2. Gradient Checkpointing

**Enable in model**:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    use_cache=False  # Required for gradient checkpointing
)

# Enable checkpointing
model.gradient_checkpointing_enable()

# Prepare with Accelerate
model = accelerator.prepare(model)
```

**Memory savings**: 30-50% with 10-15% slowdown

### 3. Mixed Precision

**BF16 (A100/H100)**:
```python
accelerator = Accelerator(mixed_precision='bf16')

# Automatic mixed precision
for batch in dataloader:
    outputs = model(**batch)  # Forward in BF16
    loss = outputs.loss
    accelerator.backward(loss)  # Backward in FP32
    optimizer.step()
```

**FP16 (V100, older GPUs)**:
```python
from accelerate.utils import GradScalerKwargs

scaler_kwargs = GradScalerKwargs(
    init_scale=2.**16,
    growth_interval=2000
)

accelerator = Accelerator(
    mixed_precision='fp16',
    kwargs_handlers=[scaler_kwargs]
)
```

**Memory savings**: 50% compared to FP32

### 4. CPU Offloading (DeepSpeed)

```python
from accelerate.utils import DeepSpeedPlugin

ds_plugin = DeepSpeedPlugin(
    zero_stage=3,
    offload_optimizer_device="cpu",  # Offload optimizer to CPU
    offload_param_device="cpu",      # Offload parameters to CPU
)

accelerator = Accelerator(
    deepspeed_plugin=ds_plugin,
    mixed_precision='bf16'
)
```

**Memory savings**: 10-20× for optimizer state, 5-10× for parameters

**Trade-off**: 20-30% slower due to CPU-GPU transfers

### 5. Flash Attention

```python
# Install flash-attn
# pip install flash-attn

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    attn_implementation="flash_attention_2"  # Enable Flash Attention 2
)

model = accelerator.prepare(model)
```

**Memory savings**: 50% for attention, 2× faster

**Requirements**: A100/H100, sequence length must be multiple of 128

## Communication Optimization

### 1. Gradient Bucketing (DDP)

```python
from accelerate.utils import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(
    bucket_cap_mb=25,  # Bucket size for gradient reduction
    gradient_as_bucket_view=True,  # Reduce memory copies
    static_graph=False  # Set True if model doesn't change
)

accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
```

**Recommended bucket sizes**:
- Small models (<1B): 25 MB
- Medium models (1-10B): 50-100 MB
- Large models (>10B): 100-200 MB

### 2. Find Unused Parameters

```python
# Only enable if model has unused parameters (slower!)
ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True
)
```

**Use case**: Models with conditional branches (e.g., mixture of experts)

**Cost**: 10-20% slower

### 3. NCCL Tuning

```bash
# Set environment variables before launch
export NCCL_DEBUG=INFO           # Debug info
export NCCL_IB_DISABLE=0         # Enable InfiniBand
export NCCL_SOCKET_IFNAME=eth0   # Network interface
export NCCL_P2P_LEVEL=NVL        # Use NVLink

accelerate launch train.py
```

**NCCL_P2P_LEVEL options**:
- `NVL`: NVLink (fastest, within node)
- `PIX`: PCIe (fast, within node)
- `PHB`: PCIe host bridge (slow, cross-node)

## Data Loading Optimization

### 1. DataLoader Workers

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Pin memory for faster GPU transfer
    prefetch_factor=2,  # Prefetch batches per worker
    persistent_workers=True  # Keep workers alive between epochs
)

train_loader = accelerator.prepare(train_loader)
```

**Recommendations**:
- `num_workers`: 2-4 per GPU (8 GPUs → 16-32 workers)
- `pin_memory`: Always True for GPU training
- `prefetch_factor`: 2-4 (higher for slow data loading)

### 2. Data Preprocessing

```python
from datasets import load_dataset

# Bad: Preprocess during training (slow)
dataset = load_dataset("openwebtext")

for batch in dataset:
    tokens = tokenizer(batch['text'])  # Slow!
    ...

# Good: Preprocess once, save
dataset = load_dataset("openwebtext")
tokenized = dataset.map(
    lambda x: tokenizer(x['text']),
    batched=True,
    num_proc=8,  # Parallel preprocessing
    remove_columns=['text']
)
tokenized.save_to_disk("preprocessed_data")

# Load preprocessed
dataset = load_from_disk("preprocessed_data")
```

### 3. Faster Tokenization

```python
import os

# Enable Rust-based tokenizers (10× faster)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    use_fast=True  # Use fast Rust tokenizer
)
```

## Compilation (PyTorch 2.0+)

### Compile Model

```python
import torch

# Compile model for faster execution
model = torch.compile(
    model,
    mode="reduce-overhead",  # Options: default, reduce-overhead, max-autotune
    fullgraph=False,         # Compile entire graph (stricter)
    dynamic=True             # Support dynamic shapes
)

model = accelerator.prepare(model)
```

**Speedup**: 10-50% depending on model

**Compilation modes**:
- `default`: Balanced (best for most cases)
- `reduce-overhead`: Min overhead (best for small batches)
- `max-autotune`: Max performance (slow compile, best for production)

### Compilation Best Practices

```python
# Bad: Compile after prepare (won't work)
model = accelerator.prepare(model)
model = torch.compile(model)  # Error!

# Good: Compile before prepare
model = torch.compile(model)
model = accelerator.prepare(model)

# Training loop
for batch in dataloader:
    # First iteration: slow (compilation)
    # Subsequent iterations: fast (compiled)
    outputs = model(**batch)
    ...
```

## Benchmarking Different Strategies

### Script Template

```python
import time
import torch
from accelerate import Accelerator

def benchmark_strategy(strategy_name, accelerator_kwargs):
    """Benchmark a specific training strategy."""
    accelerator = Accelerator(**accelerator_kwargs)

    # Setup
    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataloader = create_dataloader()

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark
    accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    start = time.time()

    num_batches = 100
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Metrics
    throughput = (num_batches * batch_size * accelerator.num_processes) / elapsed
    memory_used = torch.cuda.max_memory_allocated() / 1e9  # GB

    if accelerator.is_main_process:
        print(f"\n{strategy_name}:")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Memory: {memory_used:.2f} GB")
        print(f"  Time: {elapsed:.2f} sec")

    torch.cuda.reset_peak_memory_stats()

# Benchmark different strategies
strategies = [
    ("DDP + FP32", {}),
    ("DDP + BF16", {"mixed_precision": "bf16"}),
    ("DDP + BF16 + GradAccum", {"mixed_precision": "bf16", "gradient_accumulation_steps": 4}),
    ("FSDP", {"fsdp_plugin": fsdp_plugin}),
    ("DeepSpeed ZeRO-2", {"deepspeed_plugin": ds_plugin_stage2}),
    ("DeepSpeed ZeRO-3", {"deepspeed_plugin": ds_plugin_stage3}),
]

for name, kwargs in strategies:
    benchmark_strategy(name, kwargs)
```

## Performance Checklist

**Before training**:
- [ ] Use BF16/FP16 mixed precision
- [ ] Enable gradient checkpointing (if OOM)
- [ ] Set appropriate `num_workers` (2-4 per GPU)
- [ ] Enable `pin_memory=True`
- [ ] Preprocess data once, not during training
- [ ] Compile model with `torch.compile` (PyTorch 2.0+)

**For large models**:
- [ ] Use FSDP or DeepSpeed ZeRO-3
- [ ] Enable CPU offloading (if still OOM)
- [ ] Use Flash Attention
- [ ] Increase gradient accumulation

**For multi-node**:
- [ ] Check network topology (InfiniBand > Ethernet)
- [ ] Tune NCCL settings
- [ ] Use larger bucket sizes for DDP
- [ ] Verify NVLink for tensor parallelism

**Profiling**:
- [ ] Profile first 10-100 batches
- [ ] Check GPU utilization (`nvidia-smi dmon`)
- [ ] Check data loading time (should be <5% of iteration)
- [ ] Identify communication bottlenecks

## Common Performance Issues

### Issue: Low GPU Utilization (<80%)

**Cause 1**: Data loading bottleneck
```python
# Solution: Increase workers and prefetch
num_workers=8
prefetch_factor=4
```

**Cause 2**: Small batch size
```python
# Solution: Increase batch size or use gradient accumulation
batch_size=32  # Increase
gradient_accumulation_steps=4  # Or accumulate
```

### Issue: High Memory Usage

**Solution 1**: Gradient checkpointing
```python
model.gradient_checkpointing_enable()
```

**Solution 2**: Reduce batch size, increase accumulation
```python
batch_size=8  # Reduce from 32
gradient_accumulation_steps=16  # Maintain effective batch
```

**Solution 3**: Use FSDP or DeepSpeed ZeRO-3
```python
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

### Issue: Slow Multi-GPU Training

**Cause**: Communication bottleneck

**Check 1**: Gradient bucket size
```python
ddp_kwargs = DistributedDataParallelKwargs(bucket_cap_mb=100)
```

**Check 2**: NCCL settings
```bash
export NCCL_DEBUG=INFO
# Check for "Using NVLS" (good) vs "Using PHB" (bad)
```

**Check 3**: Network bandwidth
```bash
# Test inter-GPU bandwidth
nvidia-smi nvlink -s
```

## Resources

- Accelerate Performance: https://huggingface.co/docs/accelerate/usage_guides/performance
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- NCCL Tuning: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- Flash Attention: https://github.com/Dao-AILab/flash-attention
