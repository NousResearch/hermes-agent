# OBLITERATUS Methods — Detailed Guide

## How Abliteration Works (Theory)

When a model is trained with RLHF/DPO/CAI, it learns to represent "should I refuse?"
as a direction in its internal activation space. When processing a "harmful" prompt,
activations shift in this direction, causing the model to generate refusal text.

Abliteration works by:
1. Measuring this direction (the difference between harmful and harmless activations)
2. Removing it from the model's weight matrices via orthogonal projection
3. The model can no longer "point toward" refusal, so it responds normally

Mathematically: `W_new = W_old - (W_old @ d @ d.T)` where `d` is the refusal direction.

## Method Details

### basic
**Technique:** Single refusal direction via diff-in-means
**Based on:** Arditi et al. 2024 ("Refusal in Language Models Is Mediated by a Single Direction")
**Speed:** Fast (~5-10 min for 8B)
**Quality:** Moderate — works for simple refusal patterns
**Best for:** Quick tests, models with clean single-direction refusal
**Limitation:** Misses complex multi-direction refusal patterns

### advanced (DEFAULT)
**Technique:** Multiple SVD directions with norm-preserving projection
**Speed:** Medium (~10-20 min for 8B)
**Quality:** Good — handles multi-direction refusal
**Best for:** Dense models (Llama, Qwen, Mistral) as a reliable default
**Key improvement:** Norm preservation prevents weight magnitude drift

### informed (RECOMMENDED)
**Technique:** Analysis-guided auto-configuration
**Speed:** Slow (~20-40 min for 8B, runs 4 analysis modules first)
**Quality:** Best — adapts to each model's specific refusal implementation
**Best for:** Any model when quality matters more than speed

The informed pipeline runs these analysis modules during abliteration:
1. **AlignmentImprintDetector** — Detects DPO/RLHF/CAI/SFT → sets regularization
2. **ConceptConeAnalyzer** — Polyhedral vs linear refusal → sets n_directions
3. **CrossLayerAlignmentAnalyzer** — Cluster-aware → selects target layers
4. **DefenseRobustnessEvaluator** — Self-repair risk → sets refinement passes
5. **Ouroboros loop** — Re-probes after excision, re-excises if refusal persists

### aggressive
**Technique:** Whitened SVD + jailbreak-contrastive activations + attention head surgery
**Speed:** Slow (~30-60 min for 8B)
**Quality:** High but higher risk of coherence damage
**Best for:** Models that resist gentler methods
**Key feature:** Whitened SVD separates refusal signal from natural activation variance

### surgical
**Technique:** SAE features + neuron masking + head surgery + per-expert directions
**Speed:** Very slow (~1-2 hrs for 8B, needs SAE)
**Quality:** Highest precision
**Best for:** Reasoning models (R1 distills) where you must preserve CoT
**Key feature:** CoT-Aware — explicitly protects reasoning-critical directions

### nuclear
**Technique:** Everything combined — expert transplant + steering + per-expert directions
**Speed:** Very slow
**Quality:** Most thorough removal, highest risk of side effects
**Best for:** Stubborn MoE models (DeepSeek, Mixtral, DBRX) that resist other methods
**Key feature:** Expert-granular abliteration decomposes signals per MoE expert

### optimized
**Technique:** Bayesian hyperparameter search via Optuna TPE
**Speed:** Very slow (runs many trials)
**Quality:** Finds optimal configuration automatically
**Best for:** Research, when you want the mathematically best parameters
**Requires:** optuna package

### spectral_cascade
**Technique:** DCT frequency-domain decomposition of refusal signal
**Speed:** Medium-slow
**Quality:** Novel approach, less battle-tested
**Best for:** Research, exploring alternative decomposition strategies

### inverted
**Technique:** Reflects (inverts) the refusal direction instead of removing it
**Speed:** Fast (same as basic)
**Quality:** Aggressive — model becomes actively willing, not just neutral
**Best for:** When you want the model to be maximally helpful
**Warning:** Can make the model too eager; may reduce safety-adjacent reasoning

### failspy / gabliteration / heretic / rdo
**Technique:** Faithful reproductions of prior community/academic work
**Speed:** Varies
**Quality:** Known baselines
**Best for:** Reproducing published results, comparing methods

## Method Selection Flowchart

```
Is this a quick test?
├─ YES → basic
└─ NO → Is the model MoE (DeepSeek, Mixtral)?
         ├─ YES → nuclear
         └─ NO → Is it a reasoning model (R1 distill)?
                  ├─ YES → surgical
                  └─ NO → Do you care about speed?
                           ├─ YES → advanced
                           └─ NO → informed
```

## Key Parameters

| Parameter           | Range    | Default | Effect                                      |
|:--------------------|:---------|:--------|:--------------------------------------------|
| n_directions        | 1-32     | auto    | More = more thorough but riskier             |
| regularization      | 0.0-1.0  | 0.0     | Higher preserves more original behavior      |
| refinement_passes   | 1-5      | 1       | More catches self-repair (Ouroboros effect)   |
| quantization        | 4/8 bit  | none    | Saves VRAM, slight quality tradeoff          |

## Troubleshooting

| Problem                    | Solution                                          |
|:---------------------------|:--------------------------------------------------|
| Refusal rate still > 10%   | Try aggressive/nuclear, add refinement passes     |
| Perplexity up > 20%        | Reduce n_directions, increase regularization       |
| Model generates nonsense   | Regularization too low, try 0.2-0.3               |
| OOM on GPU                 | Use 4-bit quantization, or try smaller model       |
| MoE model barely changes   | Use nuclear method (expert-granular)               |
| CoT reasoning broken       | Use surgical method (CoT-aware)                    |
