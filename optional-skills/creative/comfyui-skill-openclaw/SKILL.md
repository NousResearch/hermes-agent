---
name: comfyui-skill-openclaw
description: "Generate images, video, and audio through ComfyUI workflows using the comfyui-skill CLI. Import workflows, manage dependencies, execute across multiple servers, track history, and optionally manage via Web UI. Use when the user asks to generate images, run ComfyUI workflows, or manage ComfyUI resources."
version: 1.0.0
requires: "ComfyUI server running locally or via Comfy Cloud; CLI auto-installed via uvx or pip"
author: HuangYuChuh
license: Apache-2.0
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [comfyui, image-generation, stable-diffusion, flux, creative, generative-ai, ai-art, workflow]
    related_skills: [stable-diffusion-image-generation, image_gen]
    category: creative
---

# ComfyUI Agent Skill (OpenClaw)

Generate images, video, and audio through ComfyUI using the `comfyui-skill` CLI.
Workflows become callable "skills" with named parameters — the agent never touches raw node graphs.

**Reference files:**
- `references/cli-reference.md` — complete command reference (27 commands)
- `references/api-notes.md` — underlying REST API (for debugging)
- `scripts/comfyui_setup.sh` — workspace initialization

## When to Use

- User asks to generate images with Stable Diffusion, SDXL, Flux, or other diffusion models
- User wants to run a specific ComfyUI workflow
- User wants to chain generative steps (txt2img → upscale → face restore)
- User needs ControlNet, inpainting, img2img, or other advanced pipelines
- User asks to manage ComfyUI queue, check models, or install custom nodes
- User wants video/audio generation via AnimateDiff, Hunyuan, AudioCraft, etc.
- User wants to import, organize, or configure workflows visually (Web UI)

## How It Works

1. **Import** a workflow JSON (editor or API format) → CLI extracts a parameter schema
2. **Run** with friendly args (`--args '{"prompt": "a cat"}'`) → CLI injects values into nodes
3. **Retrieve** outputs → CLI downloads generated files locally

The agent never sees node IDs or graph wiring. The CLI handles:
- Editor → API format conversion (resolves reroutes, widget ordering via `/object_info`)
- Auto-upload of local images referenced in args
- Dependency checking (missing custom nodes, models)
- WebSocket streaming with polling fallback
- Multi-server routing (`server_id/workflow_id`)
- Idempotent execution via `--job-id`

## CLI Invocation

**Zero-install (recommended):**

```bash
uvx --from comfyui-skill-cli comfyui-skill [OPTIONS] COMMAND [ARGS]
```

**If installed via pip/pipx:**

```bash
comfyui-skill [OPTIONS] COMMAND [ARGS]
```

For brevity, examples below use an alias:

```bash
COMFY="uvx --from comfyui-skill-cli comfyui-skill"
# Or if pip-installed: COMFY="comfyui-skill"
```

**Always pass `--json` for structured output** the agent can parse:

```bash
$COMFY --json list
$COMFY --json run my-workflow --args '{"prompt": "a cat"}'
```

## Setup & Onboarding

### 1. ComfyUI Must Be Running

The CLI talks to a running ComfyUI server. If the user doesn't have one:

- Point them to https://docs.comfy.org/installation
- Supports: NVIDIA (CUDA), AMD (ROCm), Intel Arc, Apple Silicon (MPS), CPU-only
- Desktop app available for Windows/macOS; manual install for Linux
- Comfy Cloud available for users without a GPU (https://platform.comfy.org)

### 2. Initialize a Workspace

The CLI reads `config.json` and `data/` from its working directory:

```bash
bash scripts/comfyui_setup.sh
```

Or manually:

```bash
mkdir -p ~/.hermes/comfyui && cd ~/.hermes/comfyui
$COMFY --json server add --id local --url http://127.0.0.1:8188 --name "Local ComfyUI"
```

For Comfy Cloud:

```bash
$COMFY --json server add --id cloud --url https://cloud.comfy.org \
  --name "Comfy Cloud" --api-key "comfyui-xxxxxxxxxxxx"
```

### 3. Verify Connection

```bash
$COMFY --json server status
```

Should return `"status": "online"`. If offline, user needs to start ComfyUI.

### 4. Import a Workflow

```bash
$COMFY --json workflow import /path/to/workflow.json --name my-workflow
```

Auto-detects format (editor or API), converts if needed, extracts parameter schema.

To import from the ComfyUI server's saved workflows:

```bash
$COMFY --json workflow import --from-server
```

## Core Workflow

### Step 1: List Available Skills

```bash
$COMFY --json list
```

Returns all imported workflows with parameter schemas and `param_count`.
- `required: true` → ask the user if not provided
- `required: false` → infer from context or omit
- Never expose node IDs; only use business parameter names

### Step 2: Check Dependencies (First Run)

```bash
$COMFY --json deps check my-workflow
```

If `is_ready` is false:

```bash
$COMFY --json deps install my-workflow --all
```

Missing models must be downloaded manually — CLI reports which folder to place them in.

### Step 3: Execute

**Blocking (recommended for most use):**

```bash
$COMFY --json run my-workflow --args '{"prompt": "a beautiful sunset", "seed": 42}'
```

Blocks until done, streams progress, downloads outputs.

**Non-blocking (for long jobs):**

```bash
# Submit
$COMFY --json submit my-workflow --args '{"prompt": "..."}'
# Returns: {"prompt_id": "abc-123"}

# Poll — each call is a SEPARATE tool invocation, do NOT loop in shell
$COMFY --json status abc-123
# Returns: {"status": "running"} or {"status": "success", "outputs": [...]}
```

**Polling pattern (critical):** Each `status` call must be a separate bash command.
Do NOT write a shell loop. Read the JSON, report progress to the user, then call again.

### Step 4: Present Results

On success, `outputs` contains file paths. Show them to the user via image preview or file reference.

## Quick Decision Tree

| User says | Command |
|-----------|---------|
| "generate an image" / "draw" | `run <skill> --args '{"prompt": "..."}'` |
| "import this workflow" | `workflow import <path>` |
| "use this image" (img2img) | `upload <image>` then `run` with the reference |
| "inpaint this" | `upload <mask> --mask` then `run` |
| "what workflows do I have" | `list` |
| "what models are available" | `models list checkpoints` |
| "check if everything's installed" | `deps check <skill>` |
| "what failed" / "show history" | `history list <skill>` |
| "cancel that" | `cancel <prompt_id>` |
| "free up GPU memory" | `free` |
| "which nodes exist for X" | `nodes search <query>` |
| "manage workflows visually" | `python3 ./ui/open_ui.py` (Web UI) |

## Web UI (Optional)

The project ships a dedicated Web UI for visual workflow management:

```bash
python3 ./ui/open_ui.py
```

The Web UI provides:
- Visual workflow import and parameter schema editing
- Drag-and-drop workflow ordering
- Multi-server configuration and health monitoring
- One-click dependency checking and installation
- Execution history browser
- i18n support (English, 简体中文, 繁體中文, 日本語, 한국어, Español)

The Web UI is a companion to the CLI, not a replacement. Agents use the CLI; humans use the Web UI for setup and configuration.

**Source:** https://github.com/HuangYuChuh/ComfyUI_Skills_OpenClaw

## Multi-Server

Skills are addressed as `server_id/workflow_id`:

```bash
$COMFY --json list                              # all servers
$COMFY --json run local/txt2img --args '{...}'  # specific server
$COMFY --json run cloud/flux --args '{...}'     # different server
$COMFY --json server stats --all                # VRAM/RAM across all servers
```

If `server_id` is omitted, the default server is used.

## Image Upload (img2img / Inpainting)

```bash
# Upload input image
$COMFY --json upload /path/to/photo.png

# Upload mask for inpainting
$COMFY --json upload /path/to/mask.png --mask --original photo.png

# Auto-upload: if a param has type "image" and value starts with /, ./, ../, ~,
# the CLI uploads it automatically
$COMFY --json run inpaint --args '{"image": "./photo.png", "mask": "./mask.png", "prompt": "fill with flowers"}'
```

## Model Discovery

```bash
$COMFY --json models list                  # all folder types
$COMFY --json models list checkpoints      # checkpoint files
$COMFY --json models list loras            # LoRA files
$COMFY --json models list controlnet       # ControlNet models
```

Folders: `checkpoints`, `loras`, `vae`, `controlnet`, `clip`, `clip_vision`,
`upscale_models`, `embeddings`, `unet`, `diffusion_models`.

## Node Discovery

```bash
$COMFY --json nodes list                   # all nodes, grouped by category
$COMFY --json nodes list -c sampling       # filter by category
$COMFY --json nodes info KSampler          # full details of one node
$COMFY --json nodes search "upscale"       # fuzzy search
```

## Queue & System

```bash
$COMFY --json queue list                   # running + pending jobs
$COMFY --json queue clear                  # clear pending
$COMFY --json cancel <prompt_id>           # cancel specific job
$COMFY --json free                         # unload models + free VRAM
$COMFY --json server stats                 # system info (VRAM, RAM, GPU)
```

## Workflow Management

```bash
$COMFY --json workflow import <path> --name <id>    # import from file
$COMFY --json workflow import --from-server          # import from ComfyUI server
$COMFY --json workflow enable <skill_id>             # enable
$COMFY --json workflow disable <skill_id>            # disable
$COMFY --json workflow delete <skill_id>             # delete
$COMFY --json info <skill_id>                        # show schema + details
```

## Idempotent Execution

For retries that shouldn't burn extra GPU:

```bash
$COMFY --json run my-workflow --args '{"prompt": "..."}' --job-id "unique-key-123"
```

If `unique-key-123` was already executed, returns the cached result instantly (O(1) file check).

## Pitfalls

1. **Working directory matters** — The CLI reads `config.json` and `data/` from CWD.
   Always `cd` to the workspace. If `list` returns empty, you're in the wrong directory.

2. **Editor format needs a live server** — Importing editor-format workflows calls
   `/object_info` to resolve widget ordering. API-format imports work offline.

3. **Missing custom nodes** — Always `deps check` before first run. "class_type not found"
   means missing nodes.

4. **JSON args quoting** — Wrap `--args` in single quotes: `--args '{"prompt": "a cat"}'`.

5. **Comfy Cloud differences** — Cloud uses `/api/` prefix and `X-API-Key` auth.
   The CLI handles this transparently when configured with `--api-key`.

6. **Model names are exact** — Case-sensitive, includes extension. Use
   `models list checkpoints` to discover installed models.

7. **Long generations** — Video and high-step workflows can take minutes. Use `run`
   for blocking or `submit` + `status` for non-blocking.

8. **Concurrent limits (Cloud)** — Free/Standard: 1 job. Creator: 3. Pro: 5.

9. **Config portability** — Use `config export` / `config import` to transfer setups.

10. **Cloud API nodes unauthorized** — Workflows using Kling, Sora, or other paid API nodes
    need a Comfy Cloud API Key. Generate one at https://platform.comfy.org and configure
    via `server add --api-key` or the Web UI's server settings.

## Verification Checklist

- [ ] `uvx --from comfyui-skill-cli comfyui-skill --version` runs successfully
- [ ] `server status` returns online
- [ ] Workspace dir has `config.json` and `data/`
- [ ] At least one workflow imported (`list` returns non-empty)
- [ ] `deps check` passes for imported workflows
- [ ] Test run completes and outputs are saved
