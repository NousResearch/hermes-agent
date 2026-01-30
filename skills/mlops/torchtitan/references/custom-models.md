# Adding Custom Models to TorchTitan

This guide explains how to add a new model to TorchTitan following the established patterns.

## Directory Structure

```
torchtitan/models/your_model/
├── model/
│   ├── __init__.py
│   ├── args.py          # Model arguments
│   ├── model.py         # Model definition
│   └── state_dict_adapter.py  # HF conversion (optional)
├── infra/
│   ├── __init__.py
│   ├── parallelize.py   # TP, FSDP, compile application
│   └── pipeline.py      # PP application (optional)
├── train_configs/
│   ├── debug_model.toml
│   └── your_model_XB.toml
├── __init__.py          # TrainSpec registration
└── README.md
```

## Step 1: Define Model Arguments

Inherit from `BaseModelArgs`:

```python
# model/args.py
from torchtitan.protocols.model import BaseModelArgs
from dataclasses import dataclass

@dataclass
class YourModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 128256

    def get_nparams_and_flops(self, seq_len: int) -> tuple[int, int]:
        """Return (num_params, flops_per_token) for throughput calculation."""
        nparams = self.vocab_size * self.dim + ...  # Calculate params
        flops = 6 * nparams  # Approximate: 6 * params for forward+backward
        return nparams, flops

    def update_from_config(self, job_config) -> "YourModelArgs":
        """Update args from training config."""
        # Override specific args from job_config if needed
        return self
```

## Step 2: Define Model

Inherit from `ModelProtocol`:

```python
# model/model.py
import torch.nn as nn
from torchtitan.protocols.model import ModelProtocol
from .args import YourModelArgs

class YourModel(ModelProtocol):
    def __init__(self, args: YourModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleDict({
            str(i): TransformerBlock(args) for i in range(args.n_layers)
        })
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h)
        h = self.norm(h)
        return self.output(h)

    def init_weights(self):
        """Initialize weights recursively."""
        for module in self.modules():
            if hasattr(module, 'init_weights') and module is not self:
                module.init_weights()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
```

**Important guidelines**:
- Write single-device model code (parallelism applied externally)
- Use `nn.ModuleDict` for layers (preserves FQNs when deleting for PP)
- Make input/output layers optional for PP compatibility
- Define `init_weights()` recursively

## Step 3: Parallelize Function

```python
# infra/parallelize.py
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module

def parallelize_your_model(
    model: YourModel,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    # Apply in this order: TP -> AC -> compile -> FSDP

    # 1. Tensor Parallelism
    if parallel_dims.tp_enabled:
        apply_tp(model, world_mesh["tp"], job_config)

    # 2. Activation Checkpointing
    if job_config.activation_checkpoint.mode == "full":
        apply_ac(model, job_config)

    # 3. torch.compile
    if job_config.compile.enable:
        model = torch.compile(model)

    # 4. FSDP
    if parallel_dims.dp_enabled:
        apply_fsdp(model, world_mesh["dp"], job_config)

    return model
```

## Step 4: Create TrainSpec

```python
# __init__.py
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec
from .model.model import YourModel
from .model.args import YourModelArgs
from .infra.parallelize import parallelize_your_model

MODEL_CONFIGS = {
    "8B": YourModelArgs(dim=4096, n_layers=32, n_heads=32),
    "70B": YourModelArgs(dim=8192, n_layers=80, n_heads=64),
}

def get_train_spec(flavor: str) -> TrainSpec:
    return TrainSpec(
        model_cls=YourModel,
        model_args=MODEL_CONFIGS[flavor],
        parallelize_fn=parallelize_your_model,
        pipeline_fn=None,  # Or your_pipeline_fn for PP
        build_optimizer_fn=build_optimizer,  # Reuse existing
        build_lr_scheduler_fn=build_lr_scheduler,  # Reuse existing
        build_dataloader_fn=build_dataloader,  # Reuse existing
        build_tokenizer_fn=build_tokenizer,  # Reuse existing
        build_loss_fn=build_loss,  # Reuse existing
        state_dict_adapter=None,  # Or YourStateDictAdapter
    )

# Register so train.py can find it
register_train_spec("your_model", get_train_spec)
```

## Step 5: State Dict Adapter (Optional)

For HuggingFace checkpoint conversion:

```python
# model/state_dict_adapter.py
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

class YourStateDictAdapter(BaseStateDictAdapter):
    def to_hf(self, state_dict: dict) -> dict:
        """Convert torchtitan state dict to HF format."""
        hf_state_dict = {}
        for key, value in state_dict.items():
            hf_key = self._convert_key_to_hf(key)
            hf_state_dict[hf_key] = value
        return hf_state_dict

    def from_hf(self, state_dict: dict) -> dict:
        """Convert HF state dict to torchtitan format."""
        tt_state_dict = {}
        for key, value in state_dict.items():
            tt_key = self._convert_key_from_hf(key)
            tt_state_dict[tt_key] = value
        return tt_state_dict
```

## Step 6: Training Config

```toml
# train_configs/your_model_8b.toml
[job]
dump_folder = "./outputs"
description = "Your Model 8B training"

[model]
name = "your_model"
flavor = "8B"

[optimizer]
name = "AdamW"
lr = 3e-4

[training]
local_batch_size = 2
seq_len = 8192
steps = 1000
dataset = "c4"

[parallelism]
data_parallel_shard_degree = -1
tensor_parallel_degree = 1
```

## Step 7: Register Model

Add to `torchtitan/models/__init__.py`:

```python
from .your_model import get_train_spec as get_your_model_train_spec

MODEL_REGISTRY["your_model"] = get_your_model_train_spec
```

## Testing

### Numerics Test

Compare output with HuggingFace implementation:

```python
def test_numerics():
    # Load same checkpoint into both implementations
    tt_model = YourModel(args).load_checkpoint(...)
    hf_model = HFYourModel.from_pretrained(...)

    # Compare outputs
    input_ids = torch.randint(0, vocab_size, (1, 128))
    tt_output = tt_model(input_ids)
    hf_output = hf_model(input_ids).logits

    torch.testing.assert_close(tt_output, hf_output, atol=1e-4, rtol=1e-4)
```

### Loss Convergence

Compare loss curves with verified baseline (see `docs/converging.md`).

### Performance Benchmark

Add benchmark config to `benchmarks/` folder.

## Guiding Principles

1. **Readability over flexibility**: Don't over-abstract
2. **Minimal model changes**: Parallelism applied externally
3. **Clean, minimal codebase**: Reuse existing components where possible
4. **Single-device semantics**: Model code should work on single GPU
