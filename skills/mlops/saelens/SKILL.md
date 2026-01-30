---
name: sparse-autoencoder-training
description: Provides guidance for training and analyzing Sparse Autoencoders (SAEs) using SAELens to decompose neural network activations into interpretable features. Use when discovering interpretable features, analyzing superposition, or studying monosemantic representations in language models.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Sparse Autoencoders, SAE, Mechanistic Interpretability, Feature Discovery, Superposition]
dependencies: [sae-lens>=6.0.0, transformer-lens>=2.0.0, torch>=2.0.0]
---

# SAELens: Sparse Autoencoders for Mechanistic Interpretability

SAELens is the primary library for training and analyzing Sparse Autoencoders (SAEs) - a technique for decomposing polysemantic neural network activations into sparse, interpretable features. Based on Anthropic's groundbreaking research on monosemanticity.

**GitHub**: [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) (1,100+ stars)

## The Problem: Polysemanticity & Superposition

Individual neurons in neural networks are **polysemantic** - they activate in multiple, semantically distinct contexts. This happens because models use **superposition** to represent more features than they have neurons, making interpretability difficult.

**SAEs solve this** by decomposing dense activations into sparse, monosemantic features - typically only a small number of features activate for any given input, and each feature corresponds to an interpretable concept.

## When to Use SAELens

**Use SAELens when you need to:**
- Discover interpretable features in model activations
- Understand what concepts a model has learned
- Study superposition and feature geometry
- Perform feature-based steering or ablation
- Analyze safety-relevant features (deception, bias, harmful content)

**Consider alternatives when:**
- You need basic activation analysis → Use **TransformerLens** directly
- You want causal intervention experiments → Use **pyvene** or **TransformerLens**
- You need production steering → Consider direct activation engineering

## Installation

```bash
pip install sae-lens
```

Requirements: Python 3.10+, transformer-lens>=2.0.0

## Core Concepts

### What SAEs Learn

SAEs are trained to reconstruct model activations through a sparse bottleneck:

```
Input Activation → Encoder → Sparse Features → Decoder → Reconstructed Activation
    (d_model)       ↓        (d_sae >> d_model)    ↓         (d_model)
                 sparsity                      reconstruction
                 penalty                          loss
```

**Loss Function**: `MSE(original, reconstructed) + L1_coefficient × L1(features)`

### Key Validation (Anthropic Research)

In "Towards Monosemanticity", human evaluators found **70% of SAE features genuinely interpretable**. Features discovered include:
- DNA sequences, legal language, HTTP requests
- Hebrew text, nutrition statements, code syntax
- Sentiment, named entities, grammatical structures

## Workflow 1: Loading and Analyzing Pre-trained SAEs

### Step-by-Step

```python
from transformer_lens import HookedTransformer
from sae_lens import SAE

# 1. Load model and pre-trained SAE
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# 2. Get model activations
tokens = model.to_tokens("The capital of France is Paris")
_, cache = model.run_with_cache(tokens)
activations = cache["resid_pre", 8]  # [batch, pos, d_model]

# 3. Encode to SAE features
sae_features = sae.encode(activations)  # [batch, pos, d_sae]
print(f"Active features: {(sae_features > 0).sum()}")

# 4. Find top features for each position
for pos in range(tokens.shape[1]):
    top_features = sae_features[0, pos].topk(5)
    token = model.to_str_tokens(tokens[0, pos:pos+1])[0]
    print(f"Token '{token}': features {top_features.indices.tolist()}")

# 5. Reconstruct activations
reconstructed = sae.decode(sae_features)
reconstruction_error = (activations - reconstructed).norm()
```

### Available Pre-trained SAEs

| Release | Model | Layers |
|---------|-------|--------|
| `gpt2-small-res-jb` | GPT-2 Small | Multiple residual streams |
| `gemma-2b-res` | Gemma 2B | Residual streams |
| Various on HuggingFace | Search tag `saelens` | Various |

### Checklist
- [ ] Load model with TransformerLens
- [ ] Load matching SAE for target layer
- [ ] Encode activations to sparse features
- [ ] Identify top-activating features per token
- [ ] Validate reconstruction quality

## Workflow 2: Training a Custom SAE

### Step-by-Step

```python
from sae_lens import SAE, LanguageModelSAERunnerConfig, SAETrainingRunner

# 1. Configure training
cfg = LanguageModelSAERunnerConfig(
    # Model
    model_name="gpt2-small",
    hook_name="blocks.8.hook_resid_pre",
    hook_layer=8,
    d_in=768,  # Model dimension

    # SAE architecture
    architecture="standard",  # or "gated", "topk"
    d_sae=768 * 8,  # Expansion factor of 8
    activation_fn="relu",

    # Training
    lr=4e-4,
    l1_coefficient=8e-5,  # Sparsity penalty
    l1_warm_up_steps=1000,
    train_batch_size_tokens=4096,
    training_tokens=100_000_000,

    # Data
    dataset_path="monology/pile-uncopyrighted",
    context_size=128,

    # Logging
    log_to_wandb=True,
    wandb_project="sae-training",

    # Checkpointing
    checkpoint_path="checkpoints",
    n_checkpoints=5,
)

# 2. Train
trainer = SAETrainingRunner(cfg)
sae = trainer.run()

# 3. Evaluate
print(f"L0 (avg active features): {trainer.metrics['l0']}")
print(f"CE Loss Recovered: {trainer.metrics['ce_loss_score']}")
```

### Key Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `d_sae` | 4-16× d_model | More features, higher capacity |
| `l1_coefficient` | 5e-5 to 1e-4 | Higher = sparser, less accurate |
| `lr` | 1e-4 to 1e-3 | Standard optimizer LR |
| `l1_warm_up_steps` | 500-2000 | Prevents early feature death |

### Evaluation Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| **L0** | 50-200 | Average active features per token |
| **CE Loss Score** | 80-95% | Cross-entropy recovered vs original |
| **Dead Features** | <5% | Features that never activate |
| **Explained Variance** | >90% | Reconstruction quality |

### Checklist
- [ ] Choose target layer and hook point
- [ ] Set expansion factor (d_sae = 4-16× d_model)
- [ ] Tune L1 coefficient for desired sparsity
- [ ] Enable L1 warm-up to prevent dead features
- [ ] Monitor metrics during training (W&B)
- [ ] Validate L0 and CE loss recovery
- [ ] Check dead feature ratio

## Workflow 3: Feature Analysis and Steering

### Analyzing Individual Features

```python
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae, _, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

# Find what activates a specific feature
feature_idx = 1234
test_texts = [
    "The scientist conducted an experiment",
    "I love chocolate cake",
    "The code compiles successfully",
    "Paris is beautiful in spring",
]

for text in test_texts:
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    features = sae.encode(cache["resid_pre", 8])
    activation = features[0, :, feature_idx].max().item()
    print(f"{activation:.3f}: {text}")
```

### Feature Steering

```python
def steer_with_feature(model, sae, prompt, feature_idx, strength=5.0):
    """Add SAE feature direction to residual stream."""
    tokens = model.to_tokens(prompt)

    # Get feature direction from decoder
    feature_direction = sae.W_dec[feature_idx]  # [d_model]

    def steering_hook(activation, hook):
        # Add scaled feature direction at all positions
        activation += strength * feature_direction
        return activation

    # Generate with steering
    output = model.generate(
        tokens,
        max_new_tokens=50,
        fwd_hooks=[("blocks.8.hook_resid_pre", steering_hook)]
    )
    return model.to_string(output[0])
```

### Feature Attribution

```python
# Which features most affect a specific output?
tokens = model.to_tokens("The capital of France is")
_, cache = model.run_with_cache(tokens)

# Get features at final position
features = sae.encode(cache["resid_pre", 8])[0, -1]  # [d_sae]

# Get logit attribution per feature
# Feature contribution = feature_activation × decoder_weight × unembedding
W_dec = sae.W_dec  # [d_sae, d_model]
W_U = model.W_U    # [d_model, vocab]

# Contribution to "Paris" logit
paris_token = model.to_single_token(" Paris")
feature_contributions = features * (W_dec @ W_U[:, paris_token])

top_features = feature_contributions.topk(10)
print("Top features for 'Paris' prediction:")
for idx, val in zip(top_features.indices, top_features.values):
    print(f"  Feature {idx.item()}: {val.item():.3f}")
```

## Common Issues & Solutions

### Issue: High dead feature ratio
```python
# WRONG: No warm-up, features die early
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=1e-4,
    l1_warm_up_steps=0,  # Bad!
)

# RIGHT: Warm-up L1 penalty
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=8e-5,
    l1_warm_up_steps=1000,  # Gradually increase
    use_ghost_grads=True,   # Revive dead features
)
```

### Issue: Poor reconstruction (low CE recovery)
```python
# Reduce sparsity penalty
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=5e-5,  # Lower = better reconstruction
    d_sae=768 * 16,       # More capacity
)
```

### Issue: Features not interpretable
```python
# Increase sparsity (higher L1)
cfg = LanguageModelSAERunnerConfig(
    l1_coefficient=1e-4,  # Higher = sparser, more interpretable
)
# Or use TopK architecture
cfg = LanguageModelSAERunnerConfig(
    architecture="topk",
    activation_fn_kwargs={"k": 50},  # Exactly 50 active features
)
```

### Issue: Memory errors during training
```python
cfg = LanguageModelSAERunnerConfig(
    train_batch_size_tokens=2048,  # Reduce batch size
    store_batch_size_prompts=4,    # Fewer prompts in buffer
    n_batches_in_buffer=8,         # Smaller activation buffer
)
```

## Integration with Neuronpedia

Browse pre-trained SAE features at [neuronpedia.org](https://neuronpedia.org):

```python
# Features are indexed by SAE ID
# Example: gpt2-small layer 8 feature 1234
# → neuronpedia.org/gpt2-small/8-res-jb/1234
```

## Key Classes Reference

| Class | Purpose |
|-------|---------|
| `SAE` | Sparse Autoencoder model |
| `LanguageModelSAERunnerConfig` | Training configuration |
| `SAETrainingRunner` | Training loop manager |
| `ActivationsStore` | Activation collection and batching |
| `HookedSAETransformer` | TransformerLens + SAE integration |

## Reference Documentation

For detailed API documentation, tutorials, and advanced usage, see the `references/` folder:

| File | Contents |
|------|----------|
| [references/README.md](references/README.md) | Overview and quick start guide |
| [references/api.md](references/api.md) | Complete API reference for SAE, TrainingSAE, configurations |
| [references/tutorials.md](references/tutorials.md) | Step-by-step tutorials for training, analysis, steering |

## External Resources

### Tutorials
- [Basic Loading & Analysis](https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb)
- [Training a Sparse Autoencoder](https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)
- [ARENA SAE Curriculum](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab)

### Papers
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) - Anthropic (2023)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) - Anthropic (2024)
- [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600) - Cunningham et al. (ICLR 2024)

### Official Documentation
- [SAELens Docs](https://jbloomaus.github.io/SAELens/)
- [Neuronpedia](https://neuronpedia.org) - Feature browser

## SAE Architectures

| Architecture | Description | Use Case |
|--------------|-------------|----------|
| **Standard** | ReLU + L1 penalty | General purpose |
| **Gated** | Learned gating mechanism | Better sparsity control |
| **TopK** | Exactly K active features | Consistent sparsity |

```python
# TopK SAE (exactly 50 features active)
cfg = LanguageModelSAERunnerConfig(
    architecture="topk",
    activation_fn="topk",
    activation_fn_kwargs={"k": 50},
)
```
