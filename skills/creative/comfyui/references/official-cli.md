# comfy-cli Command Reference

Official CLI from [Comfy-Org/comfy-cli](https://github.com/Comfy-Org/comfy-cli).
Docs: https://docs.comfy.org/comfy-cli/getting-started

## Installation

```bash
pip install comfy-cli
# or
uvx --from comfy-cli comfy --help
```

First run may prompt for analytics. Disable non-interactively:
```bash
comfy --skip-prompt tracking disable
```

## Global Options

| Option | Description |
|--------|-------------|
| `--workspace <path>` | Target a specific ComfyUI workspace |
| `--recent` | Use most recently used workspace |
| `--here` | Use current directory as workspace |
| `--skip-prompt` | No interactive prompts (use defaults) |
| `-v` / `--version` | Print version |

Workspace resolution priority:
1. `--workspace` (explicit path)
2. `--recent` (from config)
3. `--here` (cwd)
4. `comfy set-default` path
5. Most recently used
6. `~/comfy/ComfyUI` (Linux) or `~/Documents/comfy/ComfyUI` (macOS)

## Commands

### `comfy install`

Download and install ComfyUI + ComfyUI-Manager.

```bash
comfy install                    # interactive GPU selection
comfy install --nvidia           # NVIDIA (CUDA)
comfy install --amd              # AMD (ROCm)
comfy install --m-series         # Apple Silicon (MPS)
comfy install --cpu              # CPU only
comfy install --fast-deps        # use uv for faster deps
comfy install --skip-manager     # skip ComfyUI-Manager
```

| Option | Description |
|--------|-------------|
| `--nvidia` | NVIDIA GPU |
| `--amd` | AMD GPU (ROCm) |
| `--m-series` | Apple Silicon |
| `--cpu` | CPU only |
| `--cuda-version` | 11.8, 12.1, 12.4, 12.6, 12.8, 12.9, 13.0 |
| `--rocm-version` | 6.1, 6.2, 6.3, 7.0, 7.1 |
| `--fast-deps` | Use uv for dependency resolution |
| `--skip-manager` | Don't install ComfyUI-Manager |
| `--skip-torch-or-directml` | Skip PyTorch install |
| `--version <ver>` | Specific ComfyUI version (e.g. `0.2.0`, `latest`, `nightly`) |
| `--commit <hash>` | Install specific commit |
| `--pr "#1234"` | Install from a PR |
| `--restore` | Restore deps for existing install |

Default location: `~/comfy/ComfyUI` (Linux), `~/Documents/comfy/ComfyUI` (macOS/Win).

### `comfy launch`

Start ComfyUI server.

```bash
comfy launch                           # foreground on :8188
comfy launch --background              # background daemon
comfy launch -- --listen 0.0.0.0       # listen on all interfaces
comfy launch -- --port 8190            # custom port
comfy launch -- --cpu                  # force CPU mode
comfy launch --background -- --listen 0.0.0.0 --port 8190
```

| Option | Description |
|--------|-------------|
| `--background` | Run as background daemon |
| `--frontend-pr "#456"` | Test a frontend PR |
| Extra args after `--` | Passed directly to ComfyUI's `main.py` |

Common extra args: `--listen`, `--port`, `--cpu`, `--lowvram`, `--novram`,
`--fp16-vae`, `--force-fp32`.

### `comfy stop`

Stop background ComfyUI instance.

```bash
comfy stop
```

### `comfy run`

Execute a raw workflow JSON file against a running server.

```bash
comfy run --workflow workflow_api.json
comfy run --workflow workflow_api.json --host 10.0.0.5 --port 8188
comfy run --workflow workflow_api.json --timeout 300 --wait
```

| Option | Description |
|--------|-------------|
| `--workflow` | Path to API-format workflow JSON (required) |
| `--host` | Server hostname (default: 127.0.0.1) |
| `--port` | Server port (default: 8188) |
| `--timeout` | Seconds to wait (default: 30) |
| `--wait/--no-wait` | Wait for completion (default: wait) |
| `--verbose` | Show per-node execution details |

**Limitations:** No parameter injection, no structured output, no image download.
For agent use, prefer `scripts/run_workflow.py` which adds those capabilities.

### `comfy which`

Show which ComfyUI workspace is currently targeted.

```bash
comfy which
comfy --recent which
```

### `comfy set-default`

Set the default workspace path.

```bash
comfy set-default /path/to/ComfyUI
comfy set-default /path/to/ComfyUI --launch-extras="--listen 0.0.0.0"
```

### `comfy update`

Update ComfyUI or custom nodes.

```bash
comfy update               # update ComfyUI core
comfy node update all      # update all custom nodes
```

---

## `comfy node` — Custom Node Management

All node operations use ComfyUI-Manager (cm-cli) under the hood.

```bash
comfy node show installed              # list installed nodes
comfy node show enabled                # list enabled nodes
comfy node show all                    # all available nodes
comfy node simple-show installed       # compact list

comfy node install comfyui-impact-pack # install by name
comfy node install <name> --uv-compile # with unified dep resolution (Manager v4.1+)
comfy node uninstall <name>            # remove
comfy node update <name>               # update one
comfy node update all                  # update all
comfy node enable <name>               # enable disabled node
comfy node disable <name>              # disable without uninstalling
comfy node fix <name>                  # fix broken dependencies

comfy node install-deps --workflow=workflow.json  # install all deps a workflow needs
comfy node deps-in-workflow --workflow=w.json --output=deps.json  # extract dep list

comfy node save-snapshot               # save current state
comfy node restore-snapshot <file>     # restore from snapshot

comfy node bisect start                # find culprit node (binary search)
comfy node bisect good                 # current set is fine
comfy node bisect bad                  # problem is in current set
comfy node bisect reset                # abort bisect
```

### Dependency Resolution Options

| Flag | Description |
|------|-------------|
| `--fast-deps` | comfy-cli built-in uv resolver |
| `--uv-compile` | ComfyUI-Manager v4.1+ unified resolver (recommended) |
| `--no-deps` | Skip dep installation |

Set uv-compile as default: `comfy manager uv-compile-default true`

---

## `comfy model` — Model Management

```bash
comfy model list                       # list all downloaded models
comfy model list --relative-path models/checkpoints  # specific folder

comfy model download --url <URL>       # download model
comfy model download --url <URL> --relative-path models/loras
comfy model download --url <URL> --filename custom_name.safetensors

comfy model remove                     # interactive removal
comfy model remove --relative-path models/checkpoints --model-names "model.safetensors"
```

| Option | Description |
|--------|-------------|
| `--url` | Download URL (CivitAI, HuggingFace, direct) |
| `--relative-path` | Subdirectory under workspace (e.g. `models/checkpoints`) |
| `--filename` | Custom filename to save as |
| `--set-civitai-api-token` | Set CivitAI API token |
| `--set-hf-api-token` | Set HuggingFace API token |
| `--downloader` | `httpx` (default) or `aria2` |

Model directory structure:
```
ComfyUI/models/
├── checkpoints/     # Full model files (.safetensors, .ckpt)
├── loras/           # LoRA adapters
├── vae/             # VAE models
├── controlnet/      # ControlNet models
├── clip/            # CLIP text encoders
├── clip_vision/     # CLIP vision encoders
├── upscale_models/  # Upscaler models (ESRGAN, etc.)
├── embeddings/      # Textual inversion embeddings
├── unet/            # UNet models
└── diffusion_models/ # Diffusion model files
```

---

## `comfy manager` — ComfyUI-Manager Settings

```bash
comfy manager disable              # disable Manager completely
comfy manager enable-gui           # enable new GUI
comfy manager disable-gui          # disable GUI (API-only)
comfy manager enable-legacy-gui    # legacy GUI
comfy manager uv-compile-default true   # make --uv-compile the default
comfy manager clear                # clear startup action
```

---

## `comfy pr-cache` — Frontend PR Cache

```bash
comfy pr-cache list                # list cached PR builds
comfy pr-cache clean               # clean all
comfy pr-cache clean 456           # clean specific PR
```

Cache expires after 7 days; max 10 builds kept.

---

## Configuration

Config file location:
- Linux: `~/.config/comfy-cli/config.ini`
- macOS: `~/Library/Application Support/comfy-cli/config.ini`
- Windows: `~/AppData/Local/comfy-cli/config.ini`

Stores: default workspace, recent workspace, background server info, API tokens,
manager GUI mode, launch extras.
