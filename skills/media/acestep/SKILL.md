---
name: acestep
description: Set up and run ACE-Step 1.5 XL, the state-of-the-art open-source music generation foundation model. Generates full songs from lyrics + tags with 50+ language support, cover generation, repainting, and vocal-to-BGM conversion.
version: 1.0.0
metadata:
  hermes:
    tags: [music, audio, generation, ai, acestep, ace-step, lyrics, songs]
    related_skills: [heartmula, audiocraft, songwriting-and-ai-music]
---

# ACE-Step 1.5 XL - Open-Source Music Generation

## Overview
ACE-Step 1.5 XL is an open-source music foundation model (MIT license) by StepFun that generates commercial-grade music from lyrics and style tags. It features a hybrid architecture with a Language Model (LM) planner and a 4B-parameter Diffusion Transformer (DiT) decoder. This is the recommended music generation skill; see also `heartmula` as an alternative.

Key capabilities:
- **Text-to-Music** - Generate full songs from lyrics + style/genre tags
- **Cover Generation** - Create covers from existing audio
- **Repainting** - Selective local audio editing and regeneration
- **Vocal-to-BGM** - Auto-generate accompaniment for vocal tracks
- **Track Separation** - Separate audio into individual stems
- **LoRA Training** - Personalize with as few as 8 songs (1 hour on RTX 3090)
- **50+ Languages** - Multilingual lyrics support with structural tags

## When to Use
- User wants to generate music/songs from text descriptions
- User wants an open-source alternative to Suno (quality between Suno v4.5 and v5)
- User wants local/offline music generation on any platform
- User asks about ACE-Step, ACEMusic, or AI music generation
- User needs fast generation (under 2s on A100, under 10s on RTX 3090)
- User wants cover generation, repainting, or vocal-to-BGM conversion
- User wants to fine-tune a music model with LoRA

## Hardware Requirements

### GPU VRAM Guide
| GPU VRAM | Recommended DiT | Recommended LM Model | Notes |
|----------|----------------|----------------------|-------|
| **6-8GB** | 2B turbo | `acestep-5Hz-lm-0.6B` | Lightweight config |
| **8-16GB** | 2B turbo/sft | `acestep-5Hz-lm-0.6B` / `1.7B` | Good balance |
| **16-20GB** | 2B sft or XL turbo | `acestep-5Hz-lm-1.7B` | XL requires CPU offload below 20GB |
| **20-24GB** | XL turbo/sft | `acestep-5Hz-lm-1.7B` | XL fits without offload |
| **24GB+** | XL sft | `acestep-5Hz-lm-4B` | Best quality, all models fit |

- **XL models**: ~9GB bf16 for weights. Minimum 12GB VRAM (with offload + quantization), 20GB+ recommended.
- **2B models**: ~4.7GB for weights. Runs on as little as 4GB VRAM with INT8 + CPU offload.

### Supported Platforms
- **NVIDIA CUDA** - Full support, recommended
- **Apple MPS** - macOS Apple Silicon support
- **AMD ROCm** - Linux and Windows (Python 3.12 required on Windows)
- **Intel XPU** - Intel GPU support
- **CPU** - Supported but slow; use cloud GPU or https://acemusic.ai for free online generation

## Installation Steps

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

### 2. Clone Repository
```bash
cd ~/  # or desired directory
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
```

### 3. Install Dependencies
```bash
uv sync
```

Models are auto-downloaded on first run. No manual checkpoint downloads needed.

### 4. Platform-Specific Notes
- **Python 3.11-3.12** required
- **macOS**: Portable package available at https://files.acemusic.ai/acemusic/mac/ACE-Step-1.5.zip
- **Windows**: Portable package available at https://files.acemusic.ai/acemusic/win/ACE-Step-1.5.7z
- **ROCm on Windows**: Requires Python 3.12 specifically

## Usage

### Method 1: Gradio Web UI (Recommended for interactive use)
```bash
cd ACE-Step-1.5
uv run acestep
```
Opens at http://localhost:7860. The UI auto-selects the best configuration for your GPU. Supports all features including LoRA training.

### Method 2: REST API Server
```bash
cd ACE-Step-1.5
uv run acestep-api
```
Starts async HTTP API at http://localhost:8001. Suitable for integration with other services.

### Method 3: Python API
```python
from acestep import ACEStepPipeline

pipe = ACEStepPipeline()
result = pipe.generate(
    tags="pop,female-vocal,piano,emotional",
    lyrics="""[Verse]
Your lyrics here...

[Chorus]
Chorus lyrics here...
""",
    duration=120,  # seconds
)
# result contains generated audio
```

### Method 4: CLI (Interactive Wizard)
```bash
cd ACE-Step-1.5
uv run acestep-cli
```
Interactive wizard mode for guided music generation.

### Platform Launch Scripts
| Platform | Gradio UI | API Server |
|----------|-----------|------------|
| **Linux** | `./start_gradio_ui.sh` | `./start_api_server.sh` |
| **macOS** | `./start_gradio_ui_macos.sh` | `./start_api_server_macos.sh` |
| **Windows** | `start_gradio_ui.bat` | `start_api_server.bat` |
| **Windows (ROCm)** | `start_gradio_ui_rocm.bat` | `start_api_server_rocm.bat` |

## Model Variants

### XL (4B) DiT Models - Recommended
| Model | Steps | Quality | Diversity | Use Case |
|-------|-------|---------|-----------|----------|
| `acestep-v15-xl-base` | 50 | High | High | Best for extract/lego/complete tasks |
| `acestep-v15-xl-sft` | 50 | Very High | Medium | Best overall quality |
| `acestep-v15-xl-turbo` | 8 | Very High | Medium | Fastest, great quality |

### 2B DiT Models - Lower VRAM
| Model | Steps | Quality | Diversity | Use Case |
|-------|-------|---------|-----------|----------|
| `acestep-v15-base` | 50 | Medium | High | Fine-tuning friendly |
| `acestep-v15-sft` | 50 | High | Medium | Good quality |
| `acestep-v15-turbo` | 8 | Very High | Medium | Fast generation |

### LM Models (Planner)
| Model | Base | Audio Understanding | Composition |
|-------|------|---------------------|-------------|
| `acestep-5Hz-lm-0.6B` | Qwen3-0.6B | Medium | Medium |
| `acestep-5Hz-lm-1.7B` | Qwen3-1.7B | Medium | Medium |
| `acestep-5Hz-lm-4B` | Qwen3-4B | Strong | Strong |

## Input Formatting

**Style/Genre Tags** (comma-separated):
```
pop,female-vocal,piano,emotional,ballad
```
or
```
rock,energetic,electric-guitar,drums,male-vocal
```

Supports 1000+ instruments and styles with fine-grained timbre description.

**Lyrics** (use bracketed structural tags):
```
[Verse]
Your lyrics here...

[Chorus]
Chorus lyrics...

[Bridge]
Bridge lyrics...

[Outro]
```

## Key Parameters
| Parameter | Description |
|-----------|-------------|
| Duration | 10 seconds to 10 minutes (600s) |
| Diffusion Steps | 8 (turbo) or 50 (base/sft) |
| CFG Scale | Classifier-free guidance (base/sft models) |
| Reference Audio | Optional audio input to guide style |
| Batch Size | Up to 8 songs simultaneously |
| BPM / Key / Time Signature | Metadata control for precise output |

## Performance
- **A100**: Under 2 seconds per full song
- **RTX 3090**: Under 10 seconds per full song
- **Output**: High-fidelity stereo audio
- **Duration Range**: 10 seconds to 10 minutes

## Configuration
Create a `.env` file in the repo root to persist settings across updates:
```bash
cp .env.example .env
```

Key settings:
```
ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
PORT=7860
LANGUAGE=en
```

## Pitfalls
1. **XL models need 12GB+ VRAM** - Use 2B variants for GPUs with less memory, or enable CPU offload + INT8 quantization.
2. **ROCm on Windows requires Python 3.12** - AMD officially provides Python 3.12 wheels only.
3. **CPU mode is very slow** - Use https://acemusic.ai for free cloud generation if no GPU is available.
4. **First run downloads models** - Expect a one-time download of several GB on first launch.

## Links
- Repo: https://github.com/ACE-Step/ACE-Step-1.5
- Models: https://huggingface.co/ACE-Step/Ace-Step1.5
- Project Page: https://ace-step.github.io/ace-step-v1.5.github.io/
- Technical Report: https://arxiv.org/abs/2602.00744
- Online Demo: https://acemusic.ai
- Discord: https://discord.gg/PeWDxrkdj7
- License: MIT
