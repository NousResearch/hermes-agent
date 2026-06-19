---
name: mlops-model-operations
description: "MLOps umbrella for model discovery, local/served inference, evaluation, experiment tracking, fine-tuned artifact handling, and specialized model tools."
---

# MLOps Model Operations

Use this skill for machine-learning model discovery, serving, evaluation, tracking, and specialized model workflows.

## Universal workflow

1. Clarify task type: find/download model, run local inference, serve an API, evaluate, track experiments, generate media, or segment images.
2. Check hardware/software constraints: OS, Python/CUDA/Metal, GPU/VRAM, disk, model license, and credentials.
3. Prefer small smoke tests before large downloads, long evaluations, or server launches.
4. Record exact model IDs, revisions, quantization, prompts/datasets, commands, and metrics.
5. Verify with a real inference/evaluation/logging result.

## Hugging Face Hub

- Use `hf` CLI/API to authenticate, search, inspect metadata, download/upload models and datasets.
- Pin revisions for reproducibility and respect gated model requirements.

## Local and server inference

- **llama.cpp/GGUF**: choose quantization for local CPU/GPU constraints; verify tokenizer/chat template and context length.
- **vLLM**: use for high-throughput OpenAI-compatible serving; configure tensor parallelism, memory utilization, and health checks.

## Evaluation and tracking

- **lm-evaluation-harness**: run standard or custom benchmarks with explicit model args, task list, batch size, and output path.
- **Weights & Biases**: log configs, metrics, artifacts, sweeps, and model registry entries; never expose API keys.

## Specialized models

- **AudioCraft/MusicGen/AudioGen**: check Python/CUDA constraints, generate a short sample first, then scale duration/batch.
- **Segment Anything (SAM)**: choose checkpoint/model variant, supply points/boxes/masks, and save visual + mask outputs for verification.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
