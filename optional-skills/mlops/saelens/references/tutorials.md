# SAELens Tutorials

## Tutorial 1: Loading and Analyzing Pre-trained SAEs

### Goal
Load a pre-trained SAE and analyze which features activate on specific inputs.

### Step-by-Step

```python
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

# 1. Load model and SAE
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda"
)

print(f"SAE input dim: {sae.cfg.d_in}")
print(f"SAE hidden dim: {sae.cfg.d_sae}")
print(f"Expansion factor: {sae.cfg.d_sae / sae.cfg.d_in:.1f}x")

# 2. Get model activations
prompt = "The capital of France is Paris"
tokens = model.to_tokens(prompt)
_, cache = model.run_with_cache(tokens)
activations = cache["resid_pre", 8]  # [1, seq_len, 768]

# 3. Encode to SAE features
features = sae.encode(activations)  # [1, seq_len, d_sae]

# 4. Analyze sparsity
active_per_token = (features > 0).sum(dim=-1)
print(f"Average active features per token: {active_per_token.float().mean():.1f}")

# 5. Find top features for each token
str_tokens = model.to_str_tokens(prompt)
for pos in range(len(str_tokens)):
    top_features = features[0, pos].topk(5)
    print(f"\nToken '{str_tokens[pos]}':")
    for feat_idx, feat_val in zip(top_features.indices, top_features.values):
        print(f"  Feature {feat_idx.item()}: {feat_val.item():.3f}")

# 6. Check reconstruction quality
reconstructed = sae.decode(features)
mse = ((activations - reconstructed) ** 2).mean()
print(f"\nReconstruction MSE: {mse.item():.6f}")
```

---

## Tutorial 2: Training a Custom SAE

### Goal
Train a Sparse Autoencoder on GPT-2 activations.

### Step-by-Step

```python
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

# 1. Configure training
cfg = LanguageModelSAERunnerConfig(
    # Model
    model_name="gpt2-small",
    hook_name="blocks.6.hook_resid_pre",
    hook_layer=6,
    d_in=768,

    # SAE architecture
    architecture="standard",
    d_sae=768 * 8,  # 8x expansion
    activation_fn="relu",

    # Training
    lr=4e-4,
    l1_coefficient=8e-5,
    l1_warm_up_steps=1000,
    train_batch_size_tokens=4096,
    training_tokens=10_000_000,  # Small run for demo

    # Data
    dataset_path="monology/pile-uncopyrighted",
    streaming=True,
    context_size=128,

    # Dead feature prevention
    use_ghost_grads=True,
    dead_feature_window=5000,

    # Logging
    log_to_wandb=True,
    wandb_project="sae-training-demo",

    # Hardware
    device="cuda",
    dtype="float32",
)

# 2. Train
runner = SAETrainingRunner(cfg)
sae = runner.run()

# 3. Save
sae.save_model("./my_trained_sae")
```

### Hyperparameter Tuning Guide

| If you see... | Try... |
|---------------|--------|
| High L0 (>200) | Increase `l1_coefficient` |
| Low CE recovery (<80%) | Decrease `l1_coefficient`, increase `d_sae` |
| Many dead features (>5%) | Enable `use_ghost_grads`, increase `l1_warm_up_steps` |
| Training instability | Lower `lr`, increase `lr_warm_up_steps` |

---

## Tutorial 3: Feature Attribution and Steering

### Goal
Identify which SAE features contribute to specific predictions and use them for steering.

### Step-by-Step

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

# 1. Feature attribution for a specific prediction
prompt = "The capital of France is"
tokens = model.to_tokens(prompt)
_, cache = model.run_with_cache(tokens)
activations = cache["resid_pre", 8]
features = sae.encode(activations)

# Target token
target_token = model.to_single_token(" Paris")

# Compute feature contributions to target logit
# contribution = feature_activation * decoder_weight * unembedding
W_dec = sae.W_dec  # [d_sae, d_model]
W_U = model.W_U    # [d_model, d_vocab]

# Feature direction projected to vocabulary
feature_to_logit = W_dec @ W_U  # [d_sae, d_vocab]

# Contribution of each feature to "Paris" at final position
feature_acts = features[0, -1]  # [d_sae]
contributions = feature_acts * feature_to_logit[:, target_token]

# Top contributing features
top_features = contributions.topk(10)
print("Top features contributing to 'Paris':")
for idx, val in zip(top_features.indices, top_features.values):
    print(f"  Feature {idx.item()}: {val.item():.3f}")

# 2. Feature steering
def steer_with_feature(feature_idx, strength=5.0):
    """Add a feature direction to the residual stream."""
    feature_direction = sae.W_dec[feature_idx]  # [d_model]

    def hook(activation, hook_obj):
        activation[:, -1, :] += strength * feature_direction
        return activation

    output = model.generate(
        tokens,
        max_new_tokens=10,
        fwd_hooks=[("blocks.8.hook_resid_pre", hook)]
    )
    return model.to_string(output[0])

# Try steering with top feature
top_feature_idx = top_features.indices[0].item()
print(f"\nSteering with feature {top_feature_idx}:")
print(steer_with_feature(top_feature_idx, strength=10.0))
```

---

## Tutorial 4: Feature Ablation

### Goal
Test the causal importance of features by ablating them.

### Step-by-Step

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

prompt = "The capital of France is"
tokens = model.to_tokens(prompt)

# Baseline prediction
baseline_logits = model(tokens)
target_token = model.to_single_token(" Paris")
baseline_prob = torch.softmax(baseline_logits[0, -1], dim=-1)[target_token].item()
print(f"Baseline P(Paris): {baseline_prob:.4f}")

# Get features to ablate
_, cache = model.run_with_cache(tokens)
activations = cache["resid_pre", 8]
features = sae.encode(activations)
top_features = features[0, -1].topk(10).indices

# Ablate top features one by one
for feat_idx in top_features:
    def ablation_hook(activation, hook, feat_idx=feat_idx):
        # Encode → zero feature → decode
        feats = sae.encode(activation)
        feats[:, :, feat_idx] = 0
        return sae.decode(feats)

    ablated_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[("blocks.8.hook_resid_pre", ablation_hook)]
    )
    ablated_prob = torch.softmax(ablated_logits[0, -1], dim=-1)[target_token].item()
    change = (ablated_prob - baseline_prob) / baseline_prob * 100
    print(f"Ablate feature {feat_idx.item()}: P(Paris)={ablated_prob:.4f} ({change:+.1f}%)")
```

---

## Tutorial 5: Comparing Features Across Prompts

### Goal
Find which features activate consistently for a concept.

### Step-by-Step

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

# Test prompts about the same concept
prompts = [
    "The Eiffel Tower is located in",
    "Paris is the capital of",
    "France's largest city is",
    "The Louvre museum is in",
]

# Collect feature activations
all_features = []
for prompt in prompts:
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    activations = cache["resid_pre", 8]
    features = sae.encode(activations)
    # Take max activation across positions
    max_features = features[0].max(dim=0).values
    all_features.append(max_features)

all_features = torch.stack(all_features)  # [n_prompts, d_sae]

# Find features that activate consistently
mean_activation = all_features.mean(dim=0)
min_activation = all_features.min(dim=0).values

# Features active in ALL prompts
consistent_features = (min_activation > 0.5).nonzero().squeeze(-1)
print(f"Features active in all prompts: {len(consistent_features)}")

# Top consistent features
top_consistent = mean_activation[consistent_features].topk(min(10, len(consistent_features)))
print("\nTop consistent features (possibly 'France/Paris' related):")
for idx, val in zip(top_consistent.indices, top_consistent.values):
    feat_idx = consistent_features[idx].item()
    print(f"  Feature {feat_idx}: mean activation {val.item():.3f}")
```

---

## External Resources

### Official Tutorials
- [Basic Loading & Analysis](https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb)
- [Training SAEs](https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)
- [Logits Lens with Features](https://github.com/jbloomAus/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb)

### ARENA Curriculum
Comprehensive SAE course: https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab

### Key Papers
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) - Anthropic (2023)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) - Anthropic (2024)
- [Sparse Autoencoders Find Interpretable Features](https://arxiv.org/abs/2309.08600) - ICLR 2024
