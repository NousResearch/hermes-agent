---
name: comfyui
description: "Use when generating images/video/audio with ComfyUI — import workflows, run them with friendly parameters, manage models and dependencies. Uses the comfyui-skill CLI over the REST API."
version: 3.0.0
requires: ComfyUI running locally or via Comfy Cloud; comfyui-skill CLI (auto-installed via uvx)
author: kshitijk4poor
license: MIT
platforms: [macos, linux, windows]
prerequisites:
  commands: ["uv"]
setup:
  help: "CLI auto-runs via uvx. ComfyUI install: https://docs.comfy.org/installation"
metadata:
  hermes:
    tags:
      [
        comfyui,
        image-generation,
        stable-diffusion,
        flux,
        creative,
        generative-ai,
        video-generation,
      ]
    related_skills: [stable-diffusion-image-generation, image_gen]
    category: creative
---

# ComfyUI

Generate images, video, and audio through ComfyUI using the `comfyui-skill` CLI.
The CLI wraps ComfyUI's REST API into an agent-friendly interface — workflows become
"skills" with named parameters (e.g., `prompt`, `seed`) instead of raw node graphs.

**Reference files in this skill:**

- `references/cli-reference.md` — complete command reference with all subcommands and options
- `references/api-notes.md` — underlying REST API routes (for debugging / advanced use)
- `scripts/comfyui_setup.sh` — workspace initialization script

## When to Use

- User asks to generate images with Stable Diffusion, SDXL, Flux, or other diffusion models
- User wants to run a specific ComfyUI workflow
- User wants to chain generative steps (txt2img → upscale → face restore)
- User needs ControlNet, inpainting, img2img, or other advanced pipelines
- User asks to manage ComfyUI queue, check models, or install custom nodes
- User wants video/audio generation via AnimateDiff, Hunyuan, AudioCraft, etc.

## How It Works

The `comfyui-skill` CLI turns ComfyUI workflows into callable "skills":

1. **Import** a workflow JSON (from editor or API format) → CLI extracts a parameter schema
2. **Run** with friendly args (`--args '{"prompt": "a cat"}'`) → CLI injects values into the right nodes
3. **Retrieve** outputs → CLI downloads generated files locally

The agent never sees raw node IDs or graph wiring. The CLI handles:

- Editor-format → API-format conversion (resolves reroutes, widget ordering via `/object_info`)
- Auto-upload of local images referenced in args
- Dependency checking (missing custom nodes, models)
- WebSocket streaming with polling fallback
- Multi-server routing
- Idempotent execution via `--job-id`

## CLI Invocation

The CLI is invoked via `uvx` (no persistent install needed):

```bash
uvx --from comfyui-skill-cli comfyui-skill [OPTIONS] COMMAND [ARGS]
```

For brevity in all examples below, we alias this:

```bash
# In execute_code / terminal, always use the full uvx form:
COMFY="uvx --from comfyui-skill-cli comfyui-skill"
```

**Always pass `--json` for structured output** the agent can parse:

```bash
$COMFY --json list
$COMFY --json run my-workflow --args '{"prompt": "a cat"}'
```

If `comfyui-skill` is already installed as a `uv tool` (`uv tool install comfyui-skill-cli`),
it's on PATH directly and `uvx` is not needed.

## Setup & Onboarding

### 1. ComfyUI Must Be Running

The CLI talks to a running ComfyUI server. If the user doesn't have one:

- Point them to https://docs.comfy.org/installation. If they ask for help in onboarding, read the docs and help them set things up.
- Supports: NVIDIA (CUDA), AMD (ROCm), Intel Arc, Apple Silicon (MPS), CPU-only
- Desktop app available for Windows/macOS; manual install for Linux
- Comfy Cloud available for users without a GPU (https://platform.comfy.org)

### 2. Initialize a Workspace

The CLI reads `config.json` and `data/` from its working directory. Run the
setup script or initialize manually:

```bash
bash scripts/comfyui_setup.sh
```

Or manually:

```bash
mkdir -p ~/.hermes/comfyui && cd ~/.hermes/comfyui
```

Then add a server:

```bash
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

Should return `{"status": "online", ...}`. If offline, user needs to start ComfyUI.

### 4. Import a Workflow

Users typically have workflow JSON files from the ComfyUI editor:

```bash
$COMFY --json workflow import /path/to/workflow.json --name my-workflow
```

The CLI auto-detects format (editor or API), converts if needed, and extracts
a parameter schema. Both formats are accepted.

To import from the ComfyUI server's saved workflows:

```bash
$COMFY --json workflow import --from-server
```

## Core Workflow

### Step 1: List Available Skills

```bash
$COMFY --json list
```

Returns all imported workflows with their parameter schemas. Required params
must be provided; optional params have sensible defaults.

### Step 2: Check Dependencies (First Run)

```bash
$COMFY --json deps check my-workflow
```

Reports missing custom nodes and models. If `is_ready` is false:

```bash
# Install missing nodes (requires ComfyUI Manager)
$COMFY --json deps install my-workflow --all

# Missing models must be downloaded manually — CLI tells you which folder
```

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

# Poll (each poll = separate command, do NOT loop in shell)
$COMFY --json status abc-123
# Returns: {"status": "running", "progress": {"value": 15, "max": 25}}

# When status = "success", outputs are in the response
```

### Step 4: Present Results

On success, the response contains output file paths. Show them to the user.
Images referenced in the output can be displayed via `vision_analyze` or
returned as file paths.

## Quick Decision Tree

| User says                         | Command                                        |
| --------------------------------- | ---------------------------------------------- |
| "generate an image" / "draw"      | `run <skill> --args '{"prompt": "..."}'`       |
| "import this workflow"            | `workflow import <path>`                       |
| "use this image" (img2img)        | `upload <image>` then `run` with the reference |
| "inpaint this"                    | `upload <mask> --mask` then `run`              |
| "what workflows do I have"        | `list`                                         |
| "what models are available"       | `models list checkpoints`                      |
| "check if everything's installed" | `deps check <skill>`                           |
| "what failed" / "show history"    | `history list <skill>`                         |
| "cancel that"                     | `cancel <prompt_id>`                           |
| "free up GPU memory"              | `free`                                         |
| "which nodes exist for X"         | `nodes search <query>`                         |

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
# Returns: {"filename": "photo.png", ...}

# Upload mask for inpainting
$COMFY --json upload /path/to/mask.png --mask --original photo.png

# Use in workflow args — if a param has type "image" and value is a local
# file path (starts with /, ./, ../, ~), the CLI auto-uploads it
$COMFY --json run inpaint --args '{"image": "/path/to/photo.png", "mask": "/path/to/mask.png", "prompt": "fill with flowers"}'
```

## Model Discovery

```bash
$COMFY --json models list                  # all folder types
$COMFY --json models list checkpoints      # checkpoint files
$COMFY --json models list loras            # LoRA files
$COMFY --json models list controlnet       # ControlNet models
```

Model folders: `checkpoints`, `loras`, `vae`, `controlnet`, `clip`, `clip_vision`,
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

If `unique-key-123` was already executed, returns the cached result instantly.

## Pitfalls

1. **Working directory matters** — The CLI reads `config.json` and `data/` from CWD.
   Always `cd` to the workspace directory before running commands. If `list` returns
   empty or `server status` fails, you're in the wrong directory.

2. **Editor format needs a live server** — Importing editor-format workflows requires
   a running ComfyUI instance (calls `/object_info` to resolve widget ordering).
   API-format imports work offline.

3. **Missing custom nodes** — Always `deps check` before first run of an imported
   workflow. "class_type not found" means missing nodes.

4. **JSON args quoting** — Wrap `--args` in single quotes to prevent bash from
   eating the double quotes: `--args '{"prompt": "a cat"}'`.

5. **Comfy Cloud differences** — Cloud uses `/api/` prefix and `X-API-Key` auth.
   The CLI handles this transparently when configured with `--api-key`.

6. **Model names are exact** — Case-sensitive, includes extension. Use
   `models list checkpoints` to discover installed models.

7. **Long generations** — Video and high-step workflows can take minutes. The `run`
   command blocks and streams progress. For very long jobs, use `submit` + `status`.

8. **Concurrent limits (Cloud)** — Free/Standard: 1 job. Creator: 3. Pro: 5.
   Extra submits queue automatically.

9. **Config portability** — Use `config export` / `config import` to transfer
   setups between machines.

## Verification Checklist

- [ ] `uv` or `uvx` available on PATH
- [ ] `comfyui-skill --json server status` returns online
- [ ] Workspace dir has `config.json` and `data/`
- [ ] At least one workflow imported (`list` returns non-empty)
- [ ] `deps check` passes for imported workflows
- [ ] Test run completes and outputs are saved
