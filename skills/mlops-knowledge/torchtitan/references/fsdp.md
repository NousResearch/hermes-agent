# FSDP2 in TorchTitan

## Why FSDP2?

FSDP2 is a rewrite of PyTorch's Fully Sharded Data Parallel (FSDP) API, removing the `FlatParameter` abstraction for better composability and simpler implementation.

### Key improvements over FSDP1

- **DTensor-based sharding**: Sharded parameters are `DTensor`s on dim-0, enabling easy manipulation and communication-free sharded state dicts
- **Better memory management**: Deterministic and lower GPU memory (7% reduction) by avoiding `recordStream`
- **Simplified API**: Fewer arguments, no wrapper class

### Performance

On Llama-7B with 8x H100s, FSDP2 achieves higher MFU with 7% lower peak memory than FSDP1, matching the same loss curve.

## API Reference

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, OffloadPolicy

@contract(state_cls=FSDPState)
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
) -> nn.Module:
```

## Sharding Strategies (ZeRO Equivalents)

| FSDP2 Configuration | FSDP1 Equivalent | DeepSpeed |
|---------------------|------------------|-----------|
| 1D mesh + `reshard_after_forward=True` | FULL_SHARD | ZeRO-3 |
| 1D mesh + `reshard_after_forward=False` | SHARD_GRAD_OP | ZeRO-2 |
| 2D mesh + `reshard_after_forward=True` | HYBRID_SHARD | MiCS |
| 1D/2D mesh + `reshard_after_forward=8` (int) | - | ZeRO++ hpZ |

## Meta-Device Initialization

FSDP2 supports materializing tensors onto GPU _after_ sharding:

```python
# Initialize on meta device (no memory)
with torch.device("meta"):
    model = Transformer()

# Apply FSDP2 sharding
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module)
fully_shard(model)

# Parameters still on meta device
for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")

# Allocate sharded parameters on GPU
model.to_empty(device="cuda")

# Initialize weights
model.init_weights()
```

## State Dict Differences

| Operation | FSDP1 | FSDP2 |
|-----------|-------|-------|
| `model.state_dict()` | Full state dict | Sharded state dict (no communication) |
| `optim.state_dict()` | Local state dict | Sharded state dict (no communication) |
| `summon_full_params()` | Supported | Use `DTensor` APIs like `full_tensor()` |
| Gradient clipping | `FSDP.clip_grad_norm_()` | `nn.utils.clip_grad_norm_()` |

## Mixed Precision

```python
from torch.distributed._composable.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)

fully_shard(model, mp_policy=mp_policy)
```

## HSDP (Hybrid Sharded Data Parallel)

For 2D parallelism with replication + sharding:

```python
from torch.distributed.device_mesh import init_device_mesh

# Replicate across 4 groups, shard within 8 GPUs each
mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=("replicate", "shard"))

fully_shard(model, mesh=mesh)
```

## Configuration in TorchTitan

```toml
[parallelism]
# FSDP sharding degree (-1 = auto, use all available GPUs)
data_parallel_shard_degree = -1

# HSDP replication degree (1 = pure FSDP, >1 = HSDP)
data_parallel_replicate_degree = 1
```

## Removed Arguments from FSDP1

These FSDP1 arguments are no longer needed:

- `auto_wrap_policy`: Apply `fully_shard` directly to modules
- `backward_prefetch`: Always uses BACKWARD_PRE
- `param_init_fn`: Use meta-device initialization
- `device_id`: Uses mesh's device automatically
- `sync_module_states`: Not needed with DTensor
- `limit_all_gathers`: New memory management doesn't need it
- `use_orig_params`: Always true (no FlatParameter)
