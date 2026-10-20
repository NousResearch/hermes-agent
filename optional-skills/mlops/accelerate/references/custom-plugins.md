# Custom Plugins for Accelerate

## Overview

Accelerate allows creating **custom plugins** to extend distributed training strategies beyond built-in options (DDP, FSDP, DeepSpeed).

## Plugin Architecture

### Base Plugin Structure

```python
from accelerate.utils import DistributedDataParallelKwargs
from dataclasses import dataclass

@dataclass
class CustomPlugin:
    """Custom training plugin."""

    # Plugin configuration
    param1: int = 1
    param2: str = "default"

    def __post_init__(self):
        # Validation logic
        if self.param1 < 1:
            raise ValueError("param1 must be >= 1")
```

### Using Custom Plugin

```python
from accelerate import Accelerator

# Create plugin
custom_plugin = CustomPlugin(param1=4, param2="value")

# Pass to Accelerator
accelerator = Accelerator(
    custom_plugin=custom_plugin  # Not a real parameter, example only
)
```

## Built-In Plugin Examples

### 1. GradScalerKwargs (FP16 Configuration)

```python
from accelerate.utils import GradScalerKwargs

# Configure gradient scaler for FP16
scaler_kwargs = GradScalerKwargs(
    init_scale=2.**16,        # Initial loss scale
    growth_factor=2.0,        # Scale growth rate
    backoff_factor=0.5,       # Scale backoff rate
    growth_interval=2000,     # Steps between scale increases
    enabled=True              # Enable scaler
)

accelerator = Accelerator(
    mixed_precision='fp16',
    kwargs_handlers=[scaler_kwargs]  # Pass as kwargs handler
)
```

**Use case**: Fine-tune FP16 gradient scaling behavior

### 2. DistributedDataParallelKwargs

```python
from accelerate.utils import DistributedDataParallelKwargs

# Configure DDP behavior
ddp_kwargs = DistributedDataParallelKwargs(
    bucket_cap_mb=25,                 # Gradient bucketing size
    find_unused_parameters=False,     # Find unused params (slower)
    check_reduction=False,            # Check gradient reduction
    gradient_as_bucket_view=True,     # Memory optimization
    static_graph=False                # Static computation graph
)

accelerator = Accelerator(
    kwargs_handlers=[ddp_kwargs]
)
```

**Use case**: Optimize DDP performance for specific models

### 3. FP8RecipeKwargs (H100 FP8)

```python
from accelerate.utils import FP8RecipeKwargs

# Configure FP8 training (H100)
fp8_recipe = FP8RecipeKwargs(
    backend="te",              # TransformerEngine backend
    margin=0,                  # Scaling margin
    interval=1,                # Scaling interval
    fp8_format="HYBRID",       # E4M3 + E5M2 hybrid
    amax_history_len=1024,     # AMAX history length
    amax_compute_algo="max"    # AMAX computation algorithm
)

accelerator = Accelerator(
    mixed_precision='fp8',
    kwargs_handlers=[fp8_recipe]
)
```

**Use case**: Ultra-fast training on H100 GPUs

## Custom DeepSpeed Configuration

### ZeRO-3 with CPU Offload

```python
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

# Custom DeepSpeed config
ds_plugin = DeepSpeedPlugin(
    zero_stage=3,                     # ZeRO-3
    offload_optimizer_device="cpu",   # CPU offload optimizer
    offload_param_device="cpu",       # CPU offload parameters
    zero3_init_flag=True,             # ZeRO-3 initialization
    zero3_save_16bit_model=True,      # Save FP16 weights
)

accelerator = Accelerator(
    deepspeed_plugin=ds_plugin,
    mixed_precision='bf16'
)
```

### ZeRO-2 with NVMe Offload

```python
ds_plugin = DeepSpeedPlugin(
    zero_stage=2,
    offload_optimizer_device="nvme",  # NVMe offload
    offload_param_device="nvme",
    nvme_path="/local_nvme",          # NVMe mount path
)
```

### Custom JSON Config

```python
import json

# Load custom DeepSpeed config
with open('deepspeed_config.json', 'r') as f:
    ds_config = json.load(f)

ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)

accelerator = Accelerator(deepspeed_plugin=ds_plugin)
```

**Example config** (`deepspeed_config.json`):
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
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
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

## Custom FSDP Configuration

### FSDP with Custom Auto-Wrap Policy

```python
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

# Custom wrap policy (size-based)
wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1e6  # Wrap layers with 1M+ params
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch strategy
    mixed_precision_policy=None,  # Use Accelerator's mixed precision
    auto_wrap_policy=wrap_policy,  # Custom wrapping
    cpu_offload=False,
    ignored_modules=None,  # Modules to not wrap
    state_dict_type="FULL_STATE_DICT",  # Save format
    optim_state_dict_config=None,
    limit_all_gathers=False,
    use_orig_params=True,  # Use original param shapes
)

accelerator = Accelerator(
    fsdp_plugin=fsdp_plugin,
    mixed_precision='bf16'
)
```

### FSDP with Transformer Auto-Wrap

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# Wrap at transformer block level
wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={GPT2Block}  # Wrap GPT2Block layers
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    auto_wrap_policy=wrap_policy
)
```

## Creating Custom Training Strategy

### Example: Custom Gradient Accumulation

```python
from accelerate import Accelerator

class CustomGradientAccumulation:
    def __init__(self, steps=4, adaptive=False):
        self.steps = steps
        self.adaptive = adaptive
        self.current_step = 0

    def should_sync(self, loss):
        """Decide whether to sync gradients."""
        self.current_step += 1

        # Adaptive: sync on high loss
        if self.adaptive and loss > threshold:
            self.current_step = 0
            return True

        # Regular: sync every N steps
        if self.current_step >= self.steps:
            self.current_step = 0
            return True

        return False

# Usage
custom_accum = CustomGradientAccumulation(steps=8, adaptive=True)
accelerator = Accelerator()

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss

    # Scale loss
    loss = loss / custom_accum.steps
    accelerator.backward(loss)

    # Conditional sync
    if custom_accum.should_sync(loss.item()):
        optimizer.step()
        optimizer.zero_grad()
```

### Example: Custom Mixed Precision

```python
import torch

class CustomMixedPrecision:
    """Custom mixed precision with dynamic loss scaling."""

    def __init__(self, init_scale=2**16, scale_window=2000):
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=init_scale,
            growth_interval=scale_window
        )
        self.scale_history = []

    def scale_loss(self, loss):
        """Scale loss for backward."""
        return self.scaler.scale(loss)

    def unscale_and_clip(self, optimizer, max_norm=1.0):
        """Unscale gradients and clip."""
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            optimizer.param_groups[0]['params'],
            max_norm
        )

    def step(self, optimizer):
        """Optimizer step with scaler update."""
        scale_before = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()

        # Track scale changes
        if scale_before != scale_after:
            self.scale_history.append(scale_after)

# Usage
custom_mp = CustomMixedPrecision()

for batch in dataloader:
    with torch.cuda.amp.autocast(dtype=torch.float16):
        loss = model(**batch).loss

    scaled_loss = custom_mp.scale_loss(loss)
    scaled_loss.backward()

    custom_mp.unscale_and_clip(optimizer, max_norm=1.0)
    custom_mp.step(optimizer)
    optimizer.zero_grad()
```

## Advanced: Custom Distributed Backend

### Custom AllReduce Strategy

```python
import torch.distributed as dist

class CustomAllReduce:
    """Custom all-reduce with compression."""

    def __init__(self, compression_ratio=0.1):
        self.compression_ratio = compression_ratio

    def compress_gradients(self, tensor):
        """Top-k gradient compression."""
        k = int(tensor.numel() * self.compression_ratio)
        values, indices = torch.topk(tensor.abs().view(-1), k)
        return values, indices

    def all_reduce_compressed(self, tensor):
        """All-reduce with gradient compression."""
        # Compress
        values, indices = self.compress_gradients(tensor)

        # All-reduce compressed gradients
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

        # Decompress
        tensor_compressed = torch.zeros_like(tensor).view(-1)
        tensor_compressed[indices] = values / dist.get_world_size()

        return tensor_compressed.view_as(tensor)

# Usage in training loop
custom_ar = CustomAllReduce(compression_ratio=0.1)

for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()

    # Custom all-reduce
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = custom_ar.all_reduce_compressed(param.grad.data)

    optimizer.step()
    optimizer.zero_grad()
```

## Plugin Best Practices

### 1. Validation in `__post_init__`

```python
@dataclass
class CustomPlugin:
    learning_rate: float = 1e-3
    warmup_steps: int = 1000

    def __post_init__(self):
        # Validate parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        # Compute derived values
        self.min_lr = self.learning_rate * 0.1
```

### 2. Compatibility Checks

```python
@dataclass
class CustomPlugin:
    feature_enabled: bool = True

    def is_compatible(self, accelerator):
        """Check if plugin is compatible with accelerator config."""
        if self.feature_enabled and accelerator.mixed_precision == 'fp8':
            raise ValueError("Custom plugin not compatible with FP8")
        return True
```

### 3. State Management

```python
@dataclass
class CustomPlugin:
    counter: int = 0
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def update_state(self, value):
        """Update plugin state during training."""
        self.counter += 1
        self.history.append(value)
```

## Resources

- Accelerate Plugins: https://huggingface.co/docs/accelerate/package_reference/kwargs
- DeepSpeed Config: https://www.deepspeed.ai/docs/config-json/
- FSDP Guide: https://pytorch.org/docs/stable/fsdp.html
- Custom Training Loops: https://huggingface.co/docs/accelerate/usage_guides/training_tpu
