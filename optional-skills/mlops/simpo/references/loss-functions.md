# Loss Functions

Complete guide to SimPO loss functions and mathematical formulations.

## Overview

SimPO supports two loss types:
- **Sigmoid** (default) - Smooth, differentiable loss
- **Hinge** - Margin-based, sparse loss

Both are reference-free (no reference model needed).

## SimPO Loss Formula

### Core Calculation

**Step 1: Log probability ratio**:
```
pi_logratios = log P_θ(y_chosen|x) - log P_θ(y_rejected|x)
```

**Step 2: Apply target margin**:
```
logits = pi_logratios - γ/β
```
Where:
- γ/β = `gamma_beta_ratio` (target margin)

**Step 3: Compute loss** (depends on loss type)

### Sigmoid Loss (Default)

**Formula**:
```
L = -log σ(β * logits) * (1 - ε) - log σ(-β * logits) * ε
```

Where:
- β = `beta` (reward scaling)
- σ = sigmoid function
- ε = `label_smoothing` (default 0.0)

**Implementation**:
```python
losses = (
    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
)
```

**Characteristics**:
- Smooth, continuous gradients
- Probabilistic interpretation
- Standard choice for most tasks
- Works well with higher beta values

### Hinge Loss

**Formula**:
```
L = max(0, 1 - β * logits)
```

**Implementation**:
```python
losses = torch.relu(1 - self.beta * logits)
```

**Characteristics**:
- Non-smooth (has kink at logits = 1/β)
- Margin-based (SVM-style)
- Can lead to sparser solutions
- Less commonly used

## Comparison to DPO

### DPO Loss (Reference Model Required)

**Formula**:
```
L_DPO = -E[log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Key features**:
- Requires reference model π_ref
- Normalizes by reference log probabilities
- More conservative (stays close to reference)

### SimPO Loss (Reference-Free)

**Formula**:
```
L_SimPO = -log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - γ/β))
```

**Key features**:
- No reference model needed
- Direct preference optimization
- Target margin γ/β controls preference strength
- More efficient (fewer model forward passes)

**Visual comparison**:
```
DPO:    [Policy] - [Reference] → Loss
SimPO:  [Policy]               → Loss
```

## Average Log Probability Reward

### Calculation

**Per-token log probabilities**:
```python
# Get log probs for each token
per_token_logps = log_softmax(logits).gather(dim=-1, index=labels)

# Create mask to ignore padding
loss_mask = (labels != label_pad_token_id)
```

**Average log probability** (if `average_log_prob=True`):
```python
avg_logp = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
```

**Sum log probability** (if `average_log_prob=False`):
```python
sum_logp = (per_token_logps * loss_mask).sum(-1)
```

**Why average?**
- Normalizes for sequence length
- Prevents bias toward shorter/longer responses
- Standard practice in SimPO

### Reward Metrics

**Chosen reward**:
```python
chosen_rewards = beta * policy_chosen_logps.detach()
```

**Rejected reward**:
```python
rejected_rewards = beta * policy_rejected_logps.detach()
```

**Reward margin**:
```python
reward_margin = chosen_rewards.mean() - rejected_rewards.mean()
```

## Label Smoothing

### Formula with Smoothing

**Sigmoid loss**:
```
L = -log σ(β * logits) * (1 - ε) - log σ(-β * logits) * ε
```

**Effect**:
- ε = 0.0: No smoothing (default)
- ε = 0.1: 10% smoothing (soft labels)
- ε = 0.5: Maximum smoothing

**When to use**:
- Noisy preference labels
- Uncertain preferences
- Prevent overconfidence

**Config**:
```yaml
label_smoothing: 0.1  # 10% smoothing
```

## SFT Regularization

### Combined Loss

**With SFT component**:
```
L_total = L_SimPO + λ * L_SFT
```

Where:
- L_SFT = cross-entropy loss on chosen responses
- λ = `sft_weight` (0.0 to 1.0)

**Implementation**:
```python
if self.sft_weight > 0:
    sft_loss = -policy_chosen_logps
    total_loss = simpo_loss + self.sft_weight * sft_loss
```

**When to use**:
- Preserve model capabilities
- Prevent catastrophic forgetting
- Fine-tuning instruct models

**Trade-off**:
- Higher sft_weight: Preserve capabilities, less alignment
- Lower sft_weight: Stronger alignment, may forget capabilities

**Config**:
```yaml
sft_weight: 0.1  # 10% SFT regularization
```

## Loss Type Selection

### Sigmoid vs Hinge

| Aspect | Sigmoid | Hinge |
|--------|---------|-------|
| Smoothness | Smooth | Non-smooth |
| Gradients | Continuous | Discontinuous at margin |
| Sparsity | Dense solutions | Sparse solutions |
| Interpretability | Probabilistic | Geometric margin |
| Use case | **General purpose** | Margin-based tasks |
| Recommendation | **Default choice** | Experimental |

**Config**:
```yaml
# Sigmoid (default)
loss_type: sigmoid

# Hinge (alternative)
loss_type: hinge
```

## Mathematical Properties

### Gradient Analysis

**Sigmoid loss gradient**:
```
∂L/∂logits = -β * σ(-β * logits) * (1 - ε) + β * σ(β * logits) * ε
```

**Hinge loss gradient**:
```
∂L/∂logits = -β   if logits < 1/β
             0     otherwise
```

**Implications**:
- Sigmoid: Always provides gradient signal
- Hinge: No gradient when margin satisfied

### Convergence Behavior

**Sigmoid**:
- Asymptotically approaches zero loss
- Continues optimizing even with large margins
- Smoother training curves

**Hinge**:
- Reaches zero loss at margin
- Stops optimizing once margin satisfied
- May have training plateaus

## Complete Loss Examples

### Example 1: Basic SimPO (Sigmoid)

**Config**:
```yaml
beta: 2.0
gamma_beta_ratio: 0.5
loss_type: sigmoid
label_smoothing: 0.0
sft_weight: 0.0
```

**Loss calculation**:
```python
# Step 1: Compute log probs
chosen_logps = avg_log_prob(policy(chosen))    # e.g., -1.2
rejected_logps = avg_log_prob(policy(rejected)) # e.g., -2.5

# Step 2: Log ratio and margin
pi_logratios = -1.2 - (-2.5) = 1.3
logits = 1.3 - 0.5 = 0.8

# Step 3: Sigmoid loss
loss = -log(sigmoid(2.0 * 0.8))
     = -log(sigmoid(1.6))
     = -log(0.832)
     = 0.184
```

### Example 2: SimPO with SFT

**Config**:
```yaml
beta: 2.5
gamma_beta_ratio: 0.5
loss_type: sigmoid
sft_weight: 0.1
```

**Loss calculation**:
```python
# SimPO loss (as above)
simpo_loss = 0.184

# SFT loss
sft_loss = -chosen_logps = -(-1.2) = 1.2

# Total loss
total_loss = simpo_loss + 0.1 * sft_loss
           = 0.184 + 0.12
           = 0.304
```

## Debugging

### Check Reward Margins

**Low margin (< 0.5)**:
- Preferences not being learned
- Increase beta or gamma_beta_ratio

**High margin (> 5.0)**:
- May be overfitting
- Reduce beta or learning rate

**Monitor**:
```python
reward_margin = chosen_rewards.mean() - rejected_rewards.mean()
print(f"Reward margin: {reward_margin:.2f}")
```

### Check Log Probabilities

**Typical values**:
- Chosen: -1.0 to -2.0 (higher is better)
- Rejected: -2.0 to -4.0 (lower is worse)

**Warning signs**:
- Both very negative (< -10): Model not learning
- Both very positive (> 0): Numerical instability

## References

- SimPO paper: https://arxiv.org/abs/2405.14734
- DPO paper: https://arxiv.org/abs/2305.18290
- Implementation: https://github.com/princeton-nlp/SimPO
