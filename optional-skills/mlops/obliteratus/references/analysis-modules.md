# OBLITERATUS analysis modules

Public README documents **15** analysis modules. Load this only when the user wants to map refusal geometry *before* (or instead of) weight surgery. The `informed` CLI method already runs a subset during obliteration.

Do not `import` these from Hermes. Analysis studies go through `obliteratus run <config.yaml> --preset ...` via the `terminal` tool.

## The 15 modules (README)

| Module | Question |
|---|---|
| Cross-layer alignment | How does the refusal direction evolve across layers? |
| Refusal logit lens | At which layer does the model decide to refuse? |
| Whitened SVD | Principal refusal directions after whitening |
| Activation probing | How much refusal signal exists at each layer? |
| Defense robustness | Will guardrails self-repair (Ouroboros)? |
| Concept cone geometry | One mechanism or many? Shared across categories? |
| Alignment imprint detection | DPO vs RLHF vs CAI vs SFT fingerprint |
| Multi-token position | Where in the sequence the refusal signal concentrates |
| Sparse surgery | Which weight rows carry the most refusal |
| Causal tracing | Which components are causally necessary |
| Residual stream decomposition | Attention vs MLP contribution |
| Linear probing classifiers | Learned probe vs analytical direction |
| Cross-model transfer | Universality index |
| Steering vectors | Inference-time disable without touching weights |
| Evaluation suite | Refusal rate, perplexity, coherence, KL, CKA, effective rank |

## `informed` auto-config

The ANALYZE stage runs four modules and wires their outputs:

| Module | Configures |
|---|---|
| Alignment imprint | Regularization / projection aggressiveness |
| Concept cone geometry | Direction count (1 if linear, up to 8 if polyhedral) |
| Cross-layer alignment | Layer selection |
| Defense robustness | Refinement passes; skip entangled layers |

VERIFY then fires extra passes if Ouroboros self-repair is detected.

## Running analysis via CLI

```bash
obliteratus run analysis-study.yaml --preset quick
obliteratus presets
obliteratus strategies
```

Presets mentioned in the CLI help include `quick`, `full`, `attention`, `jailbreak`, `guardrail`.

Ablation strategies (structural, not direction projection): `layer_removal`, `head_pruning`, `ffn_ablation`, `embedding_ablation`.

See `templates/analysis-study.yaml` for a YAML skeleton.
