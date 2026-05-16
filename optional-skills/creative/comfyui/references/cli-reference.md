# comfyui-skill CLI Reference

Complete command map for `comfyui-skill` v0.2.x.

**Invocation:** `uvx --from comfyui-skill-cli comfyui-skill [OPTIONS] COMMAND [ARGS]`

Or if installed as a tool: `comfyui-skill [OPTIONS] COMMAND [ARGS]`

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-V` | Show version |
| `--json` | `-j` | JSON output (always use this for agent parsing) |
| `--output-format` | | `text`, `json`, or `stream-json` (NDJSON events) |
| `--server` | `-s` | Server ID override |
| `--dir` | `-d` | Data directory (default: CWD) |
| `--verbose` | `-v` | Verbose output |
| `--no-update-check` | | Skip CLI update check |

## Standalone Commands

### `list`
List all available skills across all enabled servers.

### `info <SKILL_ID>`
Show skill details and parameter schema. Skill ID format: `server_id/workflow_id` or `workflow_id`.

### `run <SKILL_ID> [OPTIONS]`
Execute a skill (blocking — waits for completion, streams progress).

| Option | Short | Description |
|--------|-------|-------------|
| `--args` | `-a` | JSON parameters (default: `{}`) |
| `--only` | | Comma-separated node IDs for partial execution |
| `--priority` | `-p` | Queue priority (lower = first, negative = jump queue; default: 0) |
| `--validate` | | Validate workflow without executing (dry run) |
| `--job-id` | | Idempotency key — reuse cached result if already executed |

### `submit <SKILL_ID> [OPTIONS]`
Submit a skill (non-blocking — returns `prompt_id` immediately). Same options as `run` except no streaming.

### `status <PROMPT_ID>`
Check execution status. Returns: `queued` (with `position`), `running` (with `progress`), `success` (with `outputs`), or `error`.

### `upload [FILE_PATH] [OPTIONS]`
Upload a file to ComfyUI for use in workflows.

| Option | Description |
|--------|-------------|
| `--from-output` | Reuse output from a previous prompt_id as input |
| `--mask` | Upload as mask (for inpainting) |
| `--original` | Original image filename (for mask upload) |

### `cancel <PROMPT_ID>`
Cancel a running or queued job.

### `free [OPTIONS]`
Release GPU memory.

| Option | Short | Description |
|--------|-------|-------------|
| `--models` | `-m` | Unload all models from VRAM |
| `--memory` | | Free all cached memory |

## Command Groups

### `server` — Manage ComfyUI Servers

| Subcommand | Description |
|------------|-------------|
| `server list` | List all configured servers |
| `server status [SERVER_ID]` | Check if server is online |
| `server stats [SERVER_ID]` | System stats: VRAM, RAM, GPU, versions (`--all` for all servers) |
| `server add` | Add server (`--id`, `--url` required; `--name`, `--output-dir`, `--auth`, `--api-key` optional) |
| `server enable <SERVER_ID>` | Enable a server |
| `server disable <SERVER_ID>` | Disable a server |
| `server remove <SERVER_ID>` | Remove a server |

### `workflow` — Manage Workflows

| Subcommand | Description |
|------------|-------------|
| `workflow import [JSON_PATH]` | Import workflow (`--name`, `--type` image/audio/video, `--from-server`, `--preview`, `--check-deps`) |
| `workflow enable <SKILL_ID>` | Enable a workflow |
| `workflow disable <SKILL_ID>` | Disable a workflow |
| `workflow delete <SKILL_ID>` | Delete a workflow |

### `models` — Discover Models

| Subcommand | Description |
|------------|-------------|
| `models list [FOLDER]` | List models in a folder (checkpoints, loras, vae, controlnet, etc.) |

### `nodes` — Discover Nodes

| Subcommand | Description |
|------------|-------------|
| `nodes list` | List all node classes (`-c` to filter by category) |
| `nodes info <NODE_CLASS>` | Full details of a node type |
| `nodes search <QUERY>` | Fuzzy search across names/categories |

### `deps` — Dependency Management

| Subcommand | Description |
|------------|-------------|
| `deps check <SKILL_ID>` | Check if dependencies are installed (returns `is_ready`) |
| `deps install <SKILL_ID>` | Install missing deps (`--repos` git URLs, `--models`, `--all`) |

### `history` — Execution History

| Subcommand | Description |
|------------|-------------|
| `history list [SKILL_ID]` | List history (`--server`, `--status`, `--limit`, `--sort`) |
| `history show <SKILL_ID> <RUN_ID>` | Show specific run details |

### `queue` — Queue Management

| Subcommand | Description |
|------------|-------------|
| `queue list` | Show running and pending jobs |
| `queue clear` | Clear all pending jobs |
| `queue delete <PROMPT_IDS...>` | Remove specific jobs from queue |

### `logs` — Server Logs

| Subcommand | Description |
|------------|-------------|
| `logs show` | Show recent server logs (`--lines` / `-n`, default: 50) |

### `templates` — Discover Templates

| Subcommand | Description |
|------------|-------------|
| `templates list` | Workflow templates from custom nodes |
| `templates subgraphs` | Reusable subgraph components |

### `config` — Configuration

| Subcommand | Description |
|------------|-------------|
| `config export` | Export config + workflows as bundle (`--output`, `--portable-only`) |
| `config import <INPUT_PATH>` | Import bundle (`--dry-run`, `--apply-environment`, `--no-overwrite`) |

## Config File Format

Located at `<workspace>/config.json`:

```json
{
  "default_server": "local",
  "servers": [
    {
      "id": "local",
      "name": "Local ComfyUI",
      "url": "http://127.0.0.1:8188",
      "enabled": true,
      "output_dir": "./outputs",
      "auth": "",
      "comfy_api_key": ""
    }
  ]
}
```

**Server fields:**
- `id` — unique identifier (no spaces/slashes/dots)
- `url` — ComfyUI base URL
- `enabled` — whether server is active
- `output_dir` — where outputs are saved (relative to workspace)
- `auth` — bearer token for authenticated servers
- `comfy_api_key` — Comfy Cloud API key (also sent as `extra_data.api_key_comfy_org` in prompts)
