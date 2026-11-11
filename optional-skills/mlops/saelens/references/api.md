# SAELens API Reference

## SAE Class

The core class representing a Sparse Autoencoder.

### Loading Pre-trained SAEs

```python
from sae_lens import SAE

# From official releases
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# From HuggingFace
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="username/repo-name",
    sae_id="path/to/sae",
    device="cuda"
)

# From local disk
sae = SAE.load_from_disk("/path/to/sae", device="cuda")
```

### SAE Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `W_enc` | [d_in, d_sae] | Encoder weights |
| `W_dec` | [d_sae, d_in] | Decoder weights |
| `b_enc` | [d_sae] | Encoder bias |
| `b_dec` | [d_in] | Decoder bias |
| `cfg` | SAEConfig | Configuration object |

### Core Methods

#### encode()

```python
# Encode activations to sparse features
features = sae.encode(activations)
# Input: [batch, pos, d_in]
# Output: [batch, pos, d_sae]
```

#### decode()

```python
# Reconstruct activations from features
reconstructed = sae.decode(features)
# Input: [batch, pos, d_sae]
# Output: [batch, pos, d_in]
```

#### forward()

```python
# Full forward pass (encode + decode)
reconstructed = sae(activations)
# Returns reconstructed activations
```

#### save_model()

```python
sae.save_model("/path/to/save")
```

---

## SAEConfig

Configuration class for SAE architecture and training context.

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_in` | int | Input dimension (model's d_model) |
| `d_sae` | int | SAE hidden dimension |
| `architecture` | str | "standard", "gated", "jumprelu", "topk" |
| `activation_fn_str` | str | Activation function name |
| `model_name` | str | Source model name |
| `hook_name` | str | Hook point in model |
| `normalize_activations` | str | Normalization method |
| `dtype` | str | Data type |
| `device` | str | Device |

### Accessing Config

```python
print(sae.cfg.d_in)      # 768 for GPT-2 small
print(sae.cfg.d_sae)     # e.g., 24576 (32x expansion)
print(sae.cfg.hook_name) # e.g., "blocks.8.hook_resid_pre"
```

---

## LanguageModelSAERunnerConfig

Comprehensive configuration for training SAEs.

### Example Configuration

```python
from sae_lens import LanguageModelSAERunnerConfig

cfg = LanguageModelSAERunnerConfig(
    # Model and hook
    model_name="gpt2-small",
    hook_name="blocks.8.hook_resid_pre",
    hook_layer=8,
    d_in=768,

    # SAE architecture
    architecture="standard",  # "standard", "gated", "jumprelu", "topk"
    d_sae=768 * 8,           # Expansion factor
    activation_fn="relu",

    # Training hyperparameters
    lr=4e-4,
    l1_coefficient=8e-5,
    lp_norm=1.0,
    lr_scheduler_name="constant",
    lr_warm_up_steps=500,

    # Sparsity control
    l1_warm_up_steps=1000,
    use_ghost_grads=True,
    feature_sampling_window=1000,
    dead_feature_window=5000,
    dead_feature_threshold=1e-8,

    # Data
    dataset_path="monology/pile-uncopyrighted",
    streaming=True,
    context_size=128,

    # Batch sizes
    train_batch_size_tokens=4096,
    store_batch_size_prompts=16,
    n_batches_in_buffer=64,

    # Training duration
    training_tokens=100_000_000,

    # Logging
    log_to_wandb=True,
    wandb_project="sae-training",
    wandb_log_frequency=100,

    # Checkpointing
    checkpoint_path="checkpoints",
    n_checkpoints=5,

    # Hardware
    device="cuda",
    dtype="float32",
)
```

### Key Parameters Explained

#### Architecture Parameters

| Parameter | Description |
|-----------|-------------|
| `architecture` | SAE type: "standard", "gated", "jumprelu", "topk" |
| `d_sae` | Hidden dimension (or use `expansion_factor`) |
| `expansion_factor` | Alternative to d_sae: d_sae = d_in × expansion_factor |
| `activation_fn` | "relu", "topk", etc. |
| `activation_fn_kwargs` | Dict for activation params (e.g., {"k": 50} for topk) |

#### Sparsity Parameters

| Parameter | Description |
|-----------|-------------|
| `l1_coefficient` | L1 penalty weight (higher = sparser) |
| `l1_warm_up_steps` | Steps to ramp up L1 penalty |
| `use_ghost_grads` | Apply gradients to dead features |
| `dead_feature_threshold` | Activation threshold for "dead" |
| `dead_feature_window` | Steps to check for dead features |

#### Learning Rate Parameters

| Parameter | Description |
|-----------|-------------|
| `lr` | Base learning rate |
| `lr_scheduler_name` | "constant", "cosineannealing", etc. |
| `lr_warm_up_steps` | LR warmup steps |
| `lr_decay_steps` | Steps for LR decay |

---

## SAETrainingRunner

Main class for executing training.

### Basic Training

```python
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

cfg = LanguageModelSAERunnerConfig(...)
runner = SAETrainingRunner(cfg)
sae = runner.run()
```

### Accessing Training Metrics

```python
# During training, metrics logged to W&B include:
# - l0: Average active features
# - ce_loss_score: Cross-entropy recovery
# - mse_loss: Reconstruction loss
# - l1_loss: Sparsity loss
# - dead_features: Count of dead features
```

---

## ActivationsStore

Manages activation collection and batching.

### Basic Usage

```python
from sae_lens import ActivationsStore

store = ActivationsStore.from_sae(
    model=model,
    sae=sae,
    store_batch_size_prompts=8,
    train_batch_size_tokens=4096,
    n_batches_in_buffer=32,
    device="cuda",
)

# Get batch of activations
activations = store.get_batch_tokens()
```

---

## HookedSAETransformer

Integration of SAEs with TransformerLens models.

### Basic Usage

```python
from sae_lens import HookedSAETransformer

# Load model with SAE
model = HookedSAETransformer.from_pretrained("gpt2-small")
model.add_sae(sae)

# Run with SAE in the loop
output = model.run_with_saes(tokens, saes=[sae])

# Cache with SAE activations
output, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
```

---

## SAE Architectures

### Standard (ReLU + L1)

```python
cfg = LanguageModelSAERunnerConfig(
    architecture="standard",
    activation_fn="relu",
    l1_coefficient=8e-5,
)
```

### Gated

```python
cfg = LanguageModelSAERunnerConfig(
    architecture="gated",
)
```

### TopK

```python
cfg = LanguageModelSAERunnerConfig(
    architecture="topk",
    activation_fn="topk",
    activation_fn_kwargs={"k": 50},  # Exactly 50 active features
)
```

### JumpReLU (State-of-the-art)

```python
cfg = LanguageModelSAERunnerConfig(
    architecture="jumprelu",
)
```

---

## Utility Functions

### Upload to HuggingFace

```python
from sae_lens import upload_saes_to_huggingface

upload_saes_to_huggingface(
    saes=[sae],
    repo_id="username/my-saes",
    token="hf_token",
)
```

### Neuronpedia Integration

```python
# Features can be viewed on Neuronpedia
# URL format: neuronpedia.org/{model}/{layer}-{sae_type}/{feature_id}
# Example: neuronpedia.org/gpt2-small/8-res-jb/1234
```
