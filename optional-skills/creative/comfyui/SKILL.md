---
name: comfyui
description: "Generate images, video, and audio with ComfyUI — install, launch, manage nodes/models, run workflows with parameter injection. Uses the official comfy-cli for lifecycle and direct REST API for execution."
version: 4.0.0
requires: ComfyUI (local or Comfy Cloud); comfy-cli (pip install comfy-cli)
author: [kshitijk4poor, alt-glitch]
license: MIT
platforms: [macos, linux, windows]
prerequisites:
  commands: ["python3"]
setup:
  help: "pip install comfy-cli && comfy install. Cloud: get API key at platform.comfy.org"
metadata:
  hermes:
    tags:
      - comfyui
      - image-generation
      - stable-diffusion
      - flux
      - creative
      - generative-ai
      - video-generation
    related_skills: [stable-diffusion-image-generation, image_gen]
    category: creative
---

# ComfyUI

Generate images, video, and audio through ComfyUI using the official `comfy-cli` for
setup/management and direct REST API calls for workflow execution.

**Reference files in this skill:**

- `references/official-cli.md` — comfy-cli command reference (install, launch, nodes, models)
- `references/rest-api.md` — ComfyUI REST API endpoints (local + cloud)
- `references/workflow-format.md` — workflow JSON format, common node types, parameter mapping

**Scripts in this skill:**

- `scripts/comfyui_setup.sh` — full setup automation (install + launch + verify)
- `scripts/extract_schema.py` — reads workflow JSON, outputs which parameters are controllable
- `scripts/run_workflow.py` — injects user args, submits workflow, monitors progress, downloads outputs
- `scripts/check_deps.py` — checks if required custom nodes and models are installed

## When to Use

- User asks to generate images with Stable Diffusion, SDXL, Flux, or other diffusion models
- User wants to run a specific ComfyUI workflow
- User wants to chain generative steps (txt2img → upscale → face restore)
- User needs ControlNet, inpainting, img2img, or other advanced pipelines
- User asks to manage ComfyUI queue, check models, or install custom nodes
- User wants video/audio generation via AnimateDiff, Hunyuan, AudioCraft, etc.

## Architecture: Two Layers

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: comfy-cli (official)                       │
│   Setup, lifecycle, nodes, models                   │
│   comfy install / launch / stop / node / model      │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│ Layer 2: REST API + skill scripts                   │
│   Workflow execution, param injection, monitoring   │
│   POST /api/prompt, GET /api/view, WebSocket        │
│   scripts/run_workflow.py, extract_schema.py        │
└─────────────────────────────────────────────────────┘
```

**Why two layers?** The official CLI handles installation and server management excellently
but has minimal workflow execution support (just raw file submission, no param injection,
no structured output). The REST API fills that gap — the scripts in this skill handle the
param injection, execution monitoring, and output download that the CLI doesn't do.

## Quick Start

### Detect Environment

```bash
# What's available?
command -v comfy >/dev/null 2>&1 && echo "comfy-cli: installed"
curl -s http://127.0.0.1:8188/system_stats 2>/dev/null && echo "server: running"
```

### Local Setup (from scratch)

```bash
pip install comfy-cli
comfy --skip-prompt tracking disable
comfy install                          # downloads ComfyUI + Manager
comfy launch --background              # starts server on :8188
```

### Cloud Setup (no local GPU)

No installation needed. Get an API key at https://platform.comfy.org/login.

```bash
export COMFY_CLOUD_API_KEY="comfyui-xxxxxxxxxxxx"
# All execution uses https://cloud.comfy.org as base URL
```

## Core Workflow

### Step 1: Get a Workflow

Users provide workflow JSON files. These come from:
- ComfyUI web editor → "Save (API Format)" button
- Community downloads (civitai, Reddit, Discord)
- The `scripts/` directory of this skill (example workflows)

**The workflow must be in API format** (node IDs as keys with `class_type`).
If user has editor format (has `nodes[]` and `links[]` at top level), they
need to re-export using "Save (API Format)" in the ComfyUI web editor.

### Step 2: Understand What's Controllable

```bash
python3 scripts/extract_schema.py workflow_api.json
```

Output (JSON):
```json
{
  "parameters": {
    "prompt": {"node_id": "6", "field": "text", "type": "string", "value": "a cat"},
    "negative_prompt": {"node_id": "7", "field": "text", "type": "string", "value": "bad quality"},
    "seed": {"node_id": "3", "field": "seed", "type": "int", "value": 42},
    "steps": {"node_id": "3", "field": "steps", "type": "int", "value": 20},
    "width": {"node_id": "5", "field": "width", "type": "int", "value": 512},
    "height": {"node_id": "5", "field": "height", "type": "int", "value": 512}
  }
}
```

### Step 3: Run with Parameters

**Local:**
```bash
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "a beautiful sunset over mountains", "seed": 123, "steps": 30}' \
  --output-dir ./outputs
```

**Cloud:**
```bash
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "a beautiful sunset", "seed": 123}' \
  --host https://cloud.comfy.org \
  --api-key "$COMFY_CLOUD_API_KEY" \
  --output-dir ./outputs
```

### Step 4: Present Results

The script outputs JSON with file paths:
```json
{
  "status": "success",
  "outputs": [
    {"file": "./outputs/ComfyUI_00001_.png", "node_id": "9", "type": "image"}
  ]
}
```

Show images to the user via `vision_analyze` or return the file path directly.

## Decision Tree

| User says | Tool | Command |
|-----------|------|---------|
| "install ComfyUI" | comfy-cli | `comfy install` |
| "start ComfyUI" | comfy-cli | `comfy launch --background` |
| "stop ComfyUI" | comfy-cli | `comfy stop` |
| "install X node" | comfy-cli | `comfy node install <name>` |
| "download X model" | comfy-cli | `comfy model download --url <url>` |
| "list installed models" | comfy-cli | `comfy model list` |
| "list installed nodes" | comfy-cli | `comfy node show installed` |
| "generate an image" | script | `run_workflow.py --args '{"prompt": "..."}'` |
| "use this image" (img2img) | REST | upload image, then run_workflow.py |
| "what can I change in this workflow?" | script | `extract_schema.py workflow.json` |
| "check if workflow deps are met" | script | `check_deps.py workflow.json` |
| "what's in the queue?" | REST | `curl http://HOST:8188/queue` |
| "cancel that" | REST | `curl -X POST http://HOST:8188/interrupt` |
| "free GPU memory" | REST | `curl -X POST http://HOST:8188/free` |

## Setup & Onboarding

### 1. Install ComfyUI

```bash
pip install comfy-cli
comfy --skip-prompt tracking disable   # disable analytics
comfy install                          # interactive: picks GPU backend
```

For non-interactive install:
```bash
comfy install --nvidia                 # NVIDIA GPU
comfy install --amd                    # AMD GPU (ROCm)
comfy install --m-series               # Apple Silicon
comfy install --cpu                    # CPU only
```

See https://docs.comfy.org/installation for full options.
If user asks for help, read the docs and assist them.

### 2. Launch Server

```bash
comfy launch --background              # starts on 127.0.0.1:8188
comfy launch -- --listen 0.0.0.0       # listen on all interfaces
comfy launch -- --port 8190            # custom port
```

Verify:
```bash
curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool
```

### 3. Install Custom Nodes

```bash
comfy node install comfyui-impact-pack
comfy node install comfyui-animatediff-evolved
comfy node update all                  # update everything
```

### 4. Download Models

```bash
# From CivitAI
comfy model download --url "https://civitai.com/api/download/models/128713" \
  --relative-path models/checkpoints

# From HuggingFace
comfy model download --url "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
  --relative-path models/checkpoints
```

### 5. Verify Everything

```bash
python3 scripts/check_deps.py workflow_api.json --host 127.0.0.1 --port 8188
```

## Image Upload (img2img / Inpainting)

Upload files directly via REST:

```bash
# Upload input image
curl -X POST "http://127.0.0.1:8188/upload/image" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
# Returns: {"name": "photo.png", "subfolder": "", "type": "input"}

# Upload mask for inpainting
curl -X POST "http://127.0.0.1:8188/upload/mask" \
  -F "image=@mask.png" -F "type=input" \
  -F 'original_ref={"filename":"photo.png","subfolder":"","type":"input"}'
```

Then reference the uploaded filename in workflow args:
```bash
python3 scripts/run_workflow.py --workflow inpaint.json \
  --args '{"image": "photo.png", "mask": "mask.png", "prompt": "fill with flowers"}'
```

## Cloud Execution

Base URL: `https://cloud.comfy.org`
Auth: `X-API-Key` header

```bash
# Submit workflow
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "cyberpunk city"}' \
  --host https://cloud.comfy.org \
  --api-key "$COMFY_CLOUD_API_KEY" \
  --output-dir ./outputs \
  --timeout 300

# Upload image for cloud workflows
curl -X POST "https://cloud.comfy.org/api/upload/image" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -F "image=@input.png" -F "type=input" -F "overwrite=true"
```

Concurrent job limits:
| Tier | Concurrent Jobs |
|------|----------------|
| Free/Standard | 1 |
| Creator | 3 |
| Pro | 5 |

Extra submissions queue automatically.

## Queue & System Management

```bash
# Check queue
curl -s http://127.0.0.1:8188/queue | python3 -m json.tool

# Clear pending queue
curl -X POST http://127.0.0.1:8188/queue -d '{"clear": true}'

# Cancel running job
curl -X POST http://127.0.0.1:8188/interrupt

# Free GPU memory (unload all models)
curl -X POST http://127.0.0.1:8188/free -H "Content-Type: application/json" \
  -d '{"unload_models": true, "free_memory": true}'

# System stats (VRAM, RAM, GPU info)
curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool
```

## Pitfalls

1. **API format required** — `comfy run` and the scripts only accept API-format workflow JSON.
   If the user has editor format (from "Save" not "Save (API Format)"), they need to
   re-export. Check: API format has `class_type` in each node object, editor format has
   top-level `nodes` and `links` arrays.

2. **Server must be running** — All execution requires a live server. `comfy launch --background`
   starts one. Check with `curl http://127.0.0.1:8188/system_stats`.

3. **Model names are exact** — Case-sensitive, includes file extension. Use
   `comfy model list` to discover what's installed.

4. **Missing custom nodes** — "class_type not found" means a required node isn't installed.
   Run `check_deps.py` to find what's missing, then `comfy node install <name>`.

5. **Working directory** — `comfy-cli` auto-detects the ComfyUI workspace. If commands
   fail with "no workspace found", use `comfy --workspace /path/to/ComfyUI <command>`
   or `comfy set-default /path/to/ComfyUI`.

6. **Cloud vs local output download** — Cloud `/api/view` returns a 302 redirect to a
   signed URL. Always follow redirects (`curl -L`). The `run_workflow.py` script handles
   this automatically.

7. **Timeout for video/audio** — Long generations (video, high step counts) can take
   minutes. Pass `--timeout 600` to `run_workflow.py`. Default is 120 seconds.

8. **tracking prompt** — First run of `comfy` may prompt for analytics tracking consent.
   Use `comfy --skip-prompt tracking disable` to skip it non-interactively.

9. **comfy-cli invocation via uvx** — If comfy-cli is not installed globally, invoke with
   `uvx --from comfy-cli comfy <command>`. All examples in this skill use bare `comfy`
   but prepend `uvx --from comfy-cli` if needed.

## Verification Checklist

- [ ] `comfy` available on PATH (or `uvx --from comfy-cli comfy --help` works)
- [ ] `curl http://127.0.0.1:8188/system_stats` returns JSON
- [ ] `comfy model list` shows at least one checkpoint
- [ ] Workflow JSON is in API format (has `class_type` keys)
- [ ] `check_deps.py` reports no missing nodes/models
- [ ] Test run completes and outputs are saved
