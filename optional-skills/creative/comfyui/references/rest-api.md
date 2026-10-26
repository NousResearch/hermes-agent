# ComfyUI REST + WebSocket API Reference

ComfyUI exposes a REST + WebSocket interface for workflow execution and
management. **The same surface is used locally and on Comfy Cloud, with
auth/path differences.**

## Connection

| | Local ComfyUI | Comfy Cloud |
|---|---|---|
| Base URL | `http://127.0.0.1:8188` | `https://cloud.comfy.org` |
| API path prefix | none (`/prompt`, `/view`, …) | `/api/...` (`/api/prompt`, `/api/view`, …) |
| Auth | none (or bearer token if configured) | `X-API-Key` header |
| WebSocket | `ws://host:port/ws?clientId={uuid}` | `wss://cloud.comfy.org/ws?clientId={uuid}&token={API_KEY}` |
| `/api/view` response | direct bytes | 302 redirect → signed URL (use `curl -L`) |

The skill scripts route URLs automatically via `_common.resolve_url()`.

## Endpoint differences on Comfy Cloud

The cloud surface diverges from local ComfyUI in several ways. The skill
scripts handle these transparently; document them here so anyone calling
`curl` directly knows.

| Local path | Cloud path | Notes |
|------------|-----------|-------|
| `/system_stats` | `/api/system_stats` | Cloud version is **public** (no auth required) |
| `/object_info` | `/api/object_info` | **Paid tier only** — free returns 403 |
| `/queue` | `/api/queue` | Paid tier only |
| `/userdata` | `/api/userdata` | Paid tier only |
| `/prompt` (POST) | `/api/prompt` (POST) | Paid tier only |
| `/upload/image` | `/api/upload/image` | Paid tier only; `subfolder` accepted but ignored |
| `/upload/mask` | `/api/upload/mask` | Same as above |
| `/view` | `/api/view` | Paid tier only; **returns 302** to signed URL |
| `/history` | `/api/history_v2` | **Renamed**; old path returns 404 |
| `/history/{id}` | `/api/history_v2/{id}` or `/api/jobs/{id}` | Both work; `/jobs` returns full job |
| `/models` | `/api/experiment/models` | **Renamed** |
| `/models/{folder}` | `/api/experiment/models/{folder}` | **Renamed**; response shape differs (see below) |

### Cloud model-list response shape

- **Local:** `["a.safetensors", "b.safetensors", …]` — flat list of strings.
- **Cloud:** `[{"name": "a.safetensors", "pathIndex": 0}, …]` — list of objects.
- **Cloud 404 with `code: "folder_not_found"`** — folder is empty or unknown,
  not an "endpoint missing" error. Distinguish by reading the body.

The skill helper `_common.parse_model_list()` normalizes both.

## Workflow Execution

### Submit Workflow

```bash
# Local
curl -X POST "http://127.0.0.1:8188/prompt" \
  -H "Content-Type: application/json" \
  -d '{"prompt": '"$(cat workflow_api.json)"', "client_id": "'"$(uuidgen)"'"}'

# Cloud
curl -X POST "https://cloud.comfy.org/api/prompt" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": '"$(cat workflow_api.json)"'}'
```

**Response:**
```json
{"prompt_id": "abc-123-def", "number": 1, "node_errors": {}}
```

If `node_errors` is non-empty, the workflow has validation errors (missing
nodes, bad inputs).

### Check Job Status (Cloud)

```bash
curl -X GET "https://cloud.comfy.org/api/job/{prompt_id}/status" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY"
```

| Status        | Description                        |
| ------------- | ---------------------------------- |
| `pending`     | Job is queued and waiting to start |
| `in_progress` | Job is currently executing         |
| `completed`   | Job finished successfully          |
| `failed`      | Job encountered an error           |
| `cancelled`   | Job was cancelled by user          |

### Job detail with outputs (Cloud)

```bash
curl -X GET "https://cloud.comfy.org/api/jobs/{prompt_id}" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY"
```

Response includes `outputs` keyed by node ID. Cloud uses `video` (singular)
in the output structure; local uses `videos` (plural). The skill scripts
accept both.

### Get History (Local)

```bash
curl -s "http://127.0.0.1:8188/history"          # all
curl -s "http://127.0.0.1:8188/history/{id}"     # one prompt_id
```

Local entry shape:
```json
{
  "<prompt_id>": {
    "prompt": [...],
    "outputs": {"<node_id>": {"images": [...]}},
    "status": {
      "status_str": "success" | "error",
      "completed": true | false,
      "messages": [["execution_start", {...}], ["execution_error", {...}], …]
    }
  }
}
```

**Important:** when reading status, check `status_str == "error"` BEFORE
checking `completed`, because both can be true for failed runs.

### Download Output

```bash
# Local (direct bytes)
curl -s "http://127.0.0.1:8188/view?filename=ComfyUI_00001_.png&subfolder=&type=output" \
  -o output.png

# Cloud (302 → signed URL; -L follows; STRIP X-API-Key for the second hop)
curl -L "https://cloud.comfy.org/api/view?filename=...&type=output" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -o output.png
```

The skill's `run_workflow.py` strips `X-API-Key` automatically on the
cross-host redirect, so the signed URL never sees your auth.

## WebSocket Monitoring

Connect for real-time execution events.

```bash
# Local
wscat -c "ws://127.0.0.1:8188/ws?clientId=MY-UUID"

# Cloud
wscat -c "wss://cloud.comfy.org/ws?clientId=MY-UUID&token=$COMFY_CLOUD_API_KEY"
```

**Note:** on Cloud the `clientId` is currently ignored — all messages for a
user are broadcast to every connection. Filter messages client-side by
`data.prompt_id`.

### JSON Message Types

| Type | When | Key Fields |
|------|------|------------|
| `status` | Queue change | `status.exec_info.queue_remaining` |
| `notification` | User-friendly status string | `value` |
| `execution_start` | Workflow begins | `prompt_id` |
| `executing` | Node running (or end-of-run if `node` is null on local) | `node`, `prompt_id` |
| `progress` | Sampling steps | `node`, `value`, `max` |
| `progress_state` | Extended progress with per-node metadata | `nodes` (dict) |
| `executed` | Node output ready | `node`, `output` (with `images`/`video`/etc.) |
| `execution_cached` | Nodes skipped because of cache | `nodes` (list of IDs) |
| `execution_success` | All done | `prompt_id` |
| `execution_error` | Failure | `exception_type`, `exception_message`, `traceback`, `node_id` |
| `execution_interrupted` | Cancelled | `prompt_id` |

### Binary Frames (Preview Images)

| Type code | Meaning |
|-----------|---------|
| `0x00000001` | `PREVIEW_IMAGE` — `[type:4][image_type:4][data]` (image_type 1=JPEG, 2=PNG) |
| `0x00000003` | `TEXT` — `[type:4][nid_len:4][nid][text]` (UTF-8) |
| `0x00000004` | `PREVIEW_IMAGE_WITH_METADATA` — `[type:4][meta_len:4][json][image_data]` |

`scripts/ws_monitor.py --previews <dir>` saves preview frames to disk.

## File Upload

```bash
# Image
curl -X POST "http://127.0.0.1:8188/upload/image" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
# Returns: {"name": "photo.png", "subfolder": "", "type": "input"}

# Mask (linked to a previously uploaded image)
curl -X POST "http://127.0.0.1:8188/upload/mask" \
  -F "image=@mask.png" -F "type=input" \
  -F 'original_ref={"filename":"photo.png","subfolder":"","type":"input"}'
```

Cloud equivalent: prepend `https://cloud.comfy.org/api` and add `-H "X-API-Key: $COMFY_CLOUD_API_KEY"`.

## Node & Model Discovery

```bash
# All node types and their input specs
curl -s "http://127.0.0.1:8188/object_info" | python3 -m json.tool

# Specific node
curl -s "http://127.0.0.1:8188/object_info/KSampler"

# Models per folder (local)
curl -s "http://127.0.0.1:8188/models/checkpoints"
curl -s "http://127.0.0.1:8188/models/loras"

# Models per folder (cloud — note the experimental prefix)
curl -s "https://cloud.comfy.org/api/experiment/models/checkpoints" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY"
```

## Queue Management

```bash
# View queue
curl -s "http://127.0.0.1:8188/queue"

# Clear all pending
curl -X POST "http://127.0.0.1:8188/queue" \
  -H "Content-Type: application/json" \
  -d '{"clear": true}'

# Delete specific items
curl -X POST "http://127.0.0.1:8188/queue" \
  -H "Content-Type: application/json" \
  -d '{"delete": ["prompt_id_1", "prompt_id_2"]}'

# Cancel currently-running job
curl -X POST "http://127.0.0.1:8188/interrupt"
```

## System Management

```bash
# Stats (VRAM, RAM, GPU, ComfyUI version)
curl -s "http://127.0.0.1:8188/system_stats"

# Free GPU memory
curl -X POST "http://127.0.0.1:8188/free" \
  -H "Content-Type: application/json" \
  -d '{"unload_models": true, "free_memory": true}'
```

## ComfyUI-Manager Endpoints (Optional)

These require ComfyUI-Manager installed. Useful for installing nodes/models
via the API instead of `comfy-cli`.

```bash
# Install a custom node from a git URL
curl -X POST "http://127.0.0.1:8188/manager/queue/install" \
  -H "Content-Type: application/json" \
  -d '{"git_url": "https://github.com/user/comfyui-node.git"}'

# Check install queue status
curl -s "http://127.0.0.1:8188/manager/queue/status"

# Install model
curl -X POST "http://127.0.0.1:8188/manager/queue/install_model" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://...", "path": "models/checkpoints", "filename": "model.safetensors"}'
```

## POST /prompt Payload Format

```json
{
  "prompt": {
    "3": {
      "class_type": "KSampler",
      "inputs": {
        "seed": 42,
        "steps": 20,
        "cfg": 7.5,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "model": ["4", 0],
        "positive": ["6", 0],
        "negative": ["7", 0],
        "latent_image": ["5", 0]
      }
    }
  },
  "client_id": "unique-uuid-for-ws-filtering",
  "extra_data": {
    "api_key_comfy_org": "optional-PARTNER-NODE-key (NOT the cloud auth key)"
  }
}
```

- `prompt`: workflow graph in API format
- `client_id`: UUID — local server uses it to filter WebSocket events; cloud
  ignores it.
- `extra_data.api_key_comfy_org`: ONLY required when the workflow uses
  partner nodes (Flux Pro, Ideogram, etc.). Don't conflate with `X-API-Key`.

## Error Categories (cloud `execution_error` `exception_type`)

| Type | Meaning |
|------|---------|
| `ValidationError` | Bad workflow / inputs (often nicer to surface from `node_errors`) |
| `ModelDownloadError` | Required model not available |
| `ImageDownloadError` | Failed to fetch input image from URL |
| `OOMError` | Out of GPU memory |
| `InsufficientFundsError` | Account balance too low (partner nodes) |
| `InactiveSubscriptionError` | Subscription not active |
