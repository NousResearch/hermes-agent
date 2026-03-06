---
name: obliteratus
description: Remove refusal behaviors from open-weight LLMs using OBLITERATUS — mechanistic interpretability techniques (diff-in-means, SVD, whitened SVD, SAE decomposition, etc.) to excise guardrails while preserving reasoning. 9 CLI methods (+ 4 Python-API-only), 15 analysis modules, 116 model presets across 5 compute tiers. Use when a user wants to uncensor, abliterate, or remove refusal from an LLM.
version: 1.0.0
author: Hermes Agent
license: MIT
dependencies: [obliteratus, torch, transformers, bitsandbytes, accelerate, safetensors]
metadata:
  hermes:
    tags: [Abliteration, Uncensoring, Refusal-Removal, LLM, Weight-Projection, SVD, Mechanistic-Interpretability, HuggingFace, Model-Surgery]

---

# OBLITERATUS Skill

Remove refusal behaviors (guardrails) from open-weight LLMs without retraining or fine-tuning. Uses mechanistic interpretability techniques — including diff-in-means, SVD, whitened SVD, SAE decomposition, Bayesian kernel projection, and more — to identify and surgically excise refusal directions from model weights while preserving reasoning capabilities.

**License warning:** OBLITERATUS is AGPL-3.0. NEVER import it as a Python library. Always invoke via CLI (`obliteratus` command) or subprocess. This keeps Hermes Agent's MIT license clean.

## When to Use This Skill

Trigger when the user:
- Wants to "uncensor" or "abliterate" an LLM
- Asks about removing refusal/guardrails from a model
- Wants to create an uncensored version of Llama, Qwen, Mistral, etc.
- Mentions "refusal removal", "abliteration", "weight projection"
- Wants to analyze how a model's refusal mechanism works
- References OBLITERATUS, FailSpy, abliterator, or refusal directions

## Step 1: Installation

Check if already installed:
```bash
obliteratus --version 2>/dev/null && echo "INSTALLED" || echo "NOT INSTALLED"
```

If not installed, clone and install from GitHub:
```
Repository: https://github.com/elder-plinius/OBLITERATUS
Install: pip install -e . (from the cloned directory)
For Gradio UI: pip install -e ".[spaces]"
```

**IMPORTANT:** Confirm with user before installing. This pulls in ~5-10GB of dependencies (PyTorch, Transformers, bitsandbytes, etc.).

## Step 2: Check Hardware

Before anything, check what GPU is available:
```bash
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'GPU: {gpu}')
    print(f'VRAM: {vram:.1f} GB')
    if vram < 4: print('TIER: tiny (models under 1B)')
    elif vram < 8: print('TIER: small (models 1-4B)')
    elif vram < 16: print('TIER: medium (models 4-9B with 4bit quant)')
    elif vram < 32: print('TIER: large (models 8-32B with 4bit quant)')
    else: print('TIER: frontier (models 32B+)')
else:
    print('NO GPU - only tiny models (under 1B) on CPU')
"
```

### VRAM Requirements (with 4-bit quantization)

| VRAM     | Max Model Size  | Example Models                              |
|:---------|:----------------|:--------------------------------------------|
| CPU only | ~1B params      | GPT-2, TinyLlama, SmolLM                    |
| 4-8 GB   | ~4B params      | Qwen2.5-1.5B, Phi-3.5 mini, Llama 3.2 3B   |
| 8-16 GB  | ~9B params      | Llama 3.1 8B, Mistral 7B, Gemma 2 9B       |
| 24 GB    | ~32B params     | Qwen3-32B, Llama 3.1 70B (tight), Command-R |
| 48 GB+   | ~72B+ params    | Qwen2.5-72B, DeepSeek-R1                    |
| Multi-GPU| 200B+ params    | Llama 3.1 405B, DeepSeek-V3 (685B MoE)      |

## Step 3: Browse Available Models

```bash
# List models for your compute tier
obliteratus models --tier medium

# Get architecture info for a specific model
obliteratus info meta-llama/Llama-3.1-8B-Instruct
```

## Step 4: Choose a Method

### Method Selection Guide

**First time / unsure? Use `informed`.** It auto-configures everything.

| Situation                         | Recommended Method | Why                                      |
|:----------------------------------|:-------------------|:-----------------------------------------|
| First attempt, any model          | `informed`         | Auto-detects alignment type, auto-tunes  |
| Quick test / prototyping          | `basic`            | Fast, simple, good enough to evaluate    |
| Dense model (Llama, Mistral)      | `advanced`         | Multi-direction, norm-preserving         |
| MoE model (DeepSeek, Mixtral)     | `nuclear`          | Expert-granular, handles MoE complexity  |
| Reasoning model (R1 distills)     | `surgical`         | CoT-aware, preserves chain-of-thought    |
| Stubborn refusals persist         | `aggressive`       | Whitened SVD + head surgery + jailbreak   |
| Want reversible changes           | Use steering vectors (see Analysis section) |
| Maximum quality, time no object   | `optimized`        | Bayesian search for best parameters      |

### 9 CLI Methods

These can be passed to `--method` on the command line:

- **basic** — Single refusal direction via diff-in-means. Fastest, simplest. (Arditi et al. 2024)
- **advanced** — Multiple SVD directions, norm-preserving projection. Good default.
- **aggressive** — Whitened SVD + jailbreak contrast + attention head surgery
- **spectral_cascade** — DCT frequency-domain decomposition
- **informed** — Runs analysis DURING abliteration to auto-configure. Detects DPO/RLHF/CAI, maps refusal geometry, compensates for self-repair. Best quality.
- **surgical** — SAE features + neuron masking + head surgery + per-expert. Maximum precision.
- **optimized** — Bayesian hyperparameter search (Optuna TPE). Slowest but optimal.
- **inverted** — Flips the refusal direction (model becomes eager to help, not just neutral)
- **nuclear** — Maximum force combo for stubborn MoE models.

### 4 Python-API-Only Methods

These reproduce prior community/academic work but are NOT available via CLI — only via the Python API (`from obliteratus.abliterate import AbliterationPipeline`). **Do not use these in CLI commands.**

- **failspy** — FailSpy/abliterator reproduction
- **gabliteration** — Gabliteration reproduction
- **heretic** — Heretic/p-e-w reproduction
- **rdo** — Refusal Direction Optimization (ICML 2025)

## Step 5: Run Abliteration

### Basic Usage

```bash
# Default (advanced method)
obliteratus obliterate meta-llama/Llama-3.1-8B-Instruct

# With the informed pipeline (recommended)
obliteratus obliterate meta-llama/Llama-3.1-8B-Instruct --method informed

# With 4-bit quantization to save VRAM
obliteratus obliterate meta-llama/Llama-3.1-8B-Instruct \
  --method informed \
  --quantization 4bit \
  --output-dir ./abliterated-models

# For large models (120B+), use conservative settings
obliteratus obliterate Qwen/Qwen2.5-72B-Instruct \
  --method advanced \
  --quantization 4bit \
  --large-model \
  --output-dir ./abliterated-models
```

### Fine-Tuning Parameters

```bash
obliteratus obliterate <model> \
  --method advanced \
  --n-directions 8 \
  --regularization 0.1 \
  --refinement-passes 3 \
  --dtype bfloat16 \
  --device auto \
  --output-dir ./output
```

Parameter explanations:
- `--n-directions N` — How many refusal directions to remove (default: auto-detected)
- `--regularization 0.0-1.0` — Fraction of original weights to preserve (higher = safer but less complete removal)
- `--refinement-passes N` — Iterative passes to catch self-repair (Ouroboros effect)
- `--dtype` — float16, bfloat16, or float32
- `--quantization` — 4bit or 8bit (saves VRAM, slight quality tradeoff)
- `--large-model` — Conservative defaults for 120B+ models (fewer directions, fewer passes)

### Interactive Mode (Guided)

For users unsure about options:
```bash
obliteratus interactive
```

### Web UI (Gradio)

```bash
obliteratus ui --port 7860
```

## Step 6: Verify Results

After abliteration, check the output report for:

| Metric         | Good Value          | Concerning Value        | Meaning                                    |
|:---------------|:--------------------|:------------------------|:-------------------------------------------|
| Refusal rate   | Near 0%             | > 10%                   | Refusals still present, try harder method  |
| Perplexity     | Within 10% of orig  | > 20% increase          | Model coherence damaged, too aggressive    |
| KL divergence  | < 0.1               | > 0.5                   | Large output distribution shift            |
| Coherence      | High                | Low                     | Model generating nonsense                  |

### If perplexity spiked (too aggressive):
1. Increase `--regularization` (e.g., 0.2 or 0.3)
2. Decrease `--n-directions` (e.g., 4 instead of 8)
3. Use a less aggressive method (`advanced` instead of `aggressive`)

### If refusal persists (not aggressive enough):
1. Use `--method aggressive` or `--method nuclear`
2. Add `--refinement-passes 3` to catch self-repair
3. Use `--method informed` which auto-compensates

## Step 7: Use the Abliterated Model

The output is a standard HuggingFace model directory. Use it like any other model:

### Quick test
```bash
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./abliterated-models/model-name")
tokenizer = AutoTokenizer.from_pretrained("./abliterated-models/model-name")
inputs = tokenizer("Write a story about:", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
EOF
```

### Upload to HuggingFace Hub
```bash
huggingface-cli login  # if not already logged in
huggingface-cli upload your-username/model-name-abliterated ./abliterated-models/model-name
```

### Serve with vLLM
```bash
vllm serve ./abliterated-models/model-name --port 8000
```

## Analysis Modules (15 Modules, Pre-Abliteration, Optional)

For understanding refusal geometry before committing to abliteration.

### Run a Study

```bash
obliteratus run study-config.yaml --preset jailbreak
```

### Study Presets

| Preset       | Purpose                              | Time   |
|:-------------|:-------------------------------------|:-------|
| `quick`      | Sanity check, basic metrics          | ~5 min |
| `jailbreak`  | Refusal circuit localization         | ~20 min|
| `guardrail`  | Guardrail robustness evaluation      | ~30 min|
| `attention`  | Attention head contributions         | ~30 min|
| `knowledge`  | FFN importance mapping               | ~30 min|
| `full`       | Complete analysis, all strategies    | ~1 hr  |

### Key Analysis Modules

- **Alignment Imprint Detection** — Fingerprints DPO vs RLHF vs CAI vs SFT from subspace geometry
- **Concept Cone Geometry** — Is refusal one linear direction or a polyhedral cone (many directions)?
- **Refusal Logit Lens** — Which transformer layer makes the refusal decision?
- **Ouroboros Detection** — Will the model self-repair its refusal after removal?
- **Causal Tracing** — Which attention heads and MLP layers are causally necessary for refusal?
- **Cross-Model Transfer** — Can refusal directions from one model architecture work on another?
- **Residual Stream Decomposition** — Attention vs MLP contribution to refusal behavior
- **SAE-based Analysis** — Sparse Autoencoder feature decomposition of refusal circuits

## Steering Vectors (Reversible Alternative)

For testing refusal removal without permanent weight changes:

Steering vectors apply activation hooks at inference time. Model weights stay unchanged.
Generated during the PROBE/DISTILL stages and can be saved/applied/removed at will.
Useful for A/B testing before committing to permanent abliteration.

## YAML Config for Reproducible Studies

For complex or reproducible workflows, use YAML configs. See templates/ for examples:
```bash
obliteratus run my_study.yaml
```

## Telemetry Notice

- **CLI usage (local installs)**: Telemetry is OFF by default. Must explicitly opt in via `OBLITERATUS_TELEMETRY=1` env var or `--contribute` flag.
- **HuggingFace Spaces**: Telemetry is ON by default (auto-enabled when `SPACE_ID` env var is detected).
- Collected: model ID, method, benchmark scores, hardware info, timing (anonymous)
- NOT collected: IP addresses, user identity, prompt content
- Force off: `export OBLITERATUS_TELEMETRY=0`

## Common Pitfalls

1. **OOM (Out of Memory)** — Use `--quantization 4bit` and `--large-model` for big models
2. **Perplexity spike** — Too aggressive. Increase `--regularization` or reduce `--n-directions`
3. **Refusal persists** — Try `--method aggressive` or `--refinement-passes 3`
4. **MoE models resist** — Use `--method nuclear` for DeepSeek, Mixtral, DBRX
5. **Gated models fail** — Run `huggingface-cli login` and accept model terms on HF website first
6. **Self-repair (Ouroboros)** — Some models reconstruct refusal. Use `--method informed` which auto-compensates
7. **CoT damage** — Reasoning models lose chain-of-thought. Use `--method surgical` (CoT-aware)
8. **Disk space** — Output is full model copy. 8B fp16 = ~16GB, 70B fp16 = ~140GB
9. **Slow on CPU** — CPU-only is viable only for tiny models (<1B). Anything bigger needs GPU.

## Complementary Hermes Skills

After abliteration:
- **axolotl** / **unsloth** — Fine-tune the abliterated model further
- **serving-llms-vllm** — Serve the model as an OpenAI-compatible API
- **sparse-autoencoder-training** — Train SAEs for deeper interpretability work

## Resources

- [OBLITERATUS GitHub](https://github.com/elder-plinius/OBLITERATUS) (AGPL-3.0)
- [HuggingFace Spaces Demo](https://huggingface.co/spaces/pliny-the-prompter/obliteratus)
- [Arditi et al. 2024 — Refusal in LMs Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Refusal Direction Optimization — ICML 2025](https://arxiv.org/abs/2411.14793)
