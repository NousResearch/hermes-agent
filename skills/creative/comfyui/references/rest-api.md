# ComfyUI REST API Reference

ComfyUI exposes a REST API + WebSocket for workflow execution and management.
Same API surface for local servers and Comfy Cloud (with auth differences).

## Connection

| | Local | Cloud |
|---|---|---|
| Base URL | `http://127.0.0.1:8188` | `https://cloud.comfy.org` |
| Auth | None (or bearer token) | `X-API-Key` header |
| WebSocket | `ws://host:port/ws?clientId={uuid}` | `wss://cloud.comfy.org/ws?clientId={uuid}&token={API_KEY}` |
| Output download | Direct bytes from `/view` | 302 redirect → signed URL (use `curl -L`) |

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

If `node_errors` is non-empty, the workflow has validation errors (missing nodes, bad inputs).

### Check Job Status (Cloud)

```bash
curl -X GET "https://cloud.comfy.org/api/job/{prompt_id}/status" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY"
```

| Status | Description |
|--------|-------------|
| `pending` | Queued, waiting to start |
| `in_progress` | Currently executing |
| `completed` | Finished successfully |
| `failed` | Encountered an error |
| `cancelled` | Cancelled by user |

### Get History (Local)

```bash
# All history
curl -s "http://127.0.0.1:8188/history"

# Specific prompt
curl -s "http://127.0.0.1:8188/history/{prompt_id}"
```

Response contains `outputs` keyed by node ID with file references.

### Download Output

```bash
# Local
curl -s "http://127.0.0.1:8188/view?filename=ComfyUI_00001_.png&subfolder=&type=output" \
  -o output.png

# Cloud (follow redirect)
curl -L "https://cloud.comfy.org/api/view?filename=ComfyUI_00001_.png&subfolder=&type=output" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -o output.png
```

---

## WebSocket Monitoring

Connect to WebSocket for real-time execution progress.

### Connection

```bash
# Local
wscat -c "ws://127.0.0.1:8188/ws?clientId=MY-UUID"

# Cloud
wscat -c "wss://cloud.comfy.org/ws?clientId=MY-UUID&token=API_KEY"
```

### Message Types (JSON)

| Type | When | Key Fields |
|------|------|------------|
| `status` | Queue change | `queue_remaining` |
| `execution_start` | Workflow begins | `prompt_id` |
| `executing` | Node running | `node` (ID), `prompt_id` |
| `progress` | Sampling steps | `node`, `value`, `max` |
| `executed` | Node output ready | `node`, `output` |
| `execution_cached` | Nodes skipped | `nodes` (list of IDs) |
| `execution_success` | All done | `prompt_id` |
| `execution_error` | Failure | `exception_type`, `exception_message`, `traceback` |
| `execution_interrupted` | Cancelled | `prompt_id` |

When `executing` has `node: null`, the workflow is complete.

### Binary Messages (Preview Images)

Format: `[4B type][4B image_type: 1=JPEG, 2=PNG][image_data...]`

---

## File Upload

### Upload Image

```bash
curl -X POST "http://127.0.0.1:8188/upload/image" \
  -F "image=@photo.png" \
  -F "type=input" \
  -F "overwrite=true"
```

Response: `{"name": "photo.png", "subfolder": "", "type": "input"}`

### Upload Mask

```bash
curl -X POST "http://127.0.0.1:8188/upload/mask" \
  -F "image=@mask.png" \
  -F "type=input" \
  -F 'original_ref={"filename":"photo.png","subfolder":"","type":"input"}'
```

---

## Node & Model Discovery

### Object Info (All Nodes)

```bash
curl -s "http://127.0.0.1:8188/object_info" | python3 -m json.tool
# Returns all node types with input/output definitions

curl -s "http://127.0.0.1:8188/object_info/KSampler"
# Returns info for one specific node type
```

### Models by Folder

```bash
curl -s "http://127.0.0.1:8188/models/checkpoints"
curl -s "http://127.0.0.1:8188/models/loras"
curl -s "http://127.0.0.1:8188/models/vae"
curl -s "http://127.0.0.1:8188/models/controlnet"
curl -s "http://127.0.0.1:8188/models/clip"
curl -s "http://127.0.0.1:8188/models/upscale_models"
curl -s "http://127.0.0.1:8188/models/embeddings"
```

Returns arrays of filenames (relative to model folder).

---

## Queue Management

```bash
# View queue (running + pending)
curl -s "http://127.0.0.1:8188/queue"

# Clear all pending
curl -X POST "http://127.0.0.1:8188/queue" \
  -H "Content-Type: application/json" \
  -d '{"clear": true}'

# Delete specific items from queue
curl -X POST "http://127.0.0.1:8188/queue" \
  -H "Content-Type: application/json" \
  -d '{"delete": ["prompt_id_1", "prompt_id_2"]}'

# Cancel currently running job
curl -X POST "http://127.0.0.1:8188/interrupt"
```

---

## System Management

```bash
# System stats (VRAM, RAM, GPU, versions)
curl -s "http://127.0.0.1:8188/system_stats"

# Free GPU memory
curl -X POST "http://127.0.0.1:8188/free" \
  -H "Content-Type: application/json" \
  -d '{"unload_models": true, "free_memory": true}'
```

---

## ComfyUI Manager Endpoints (Optional)

These require ComfyUI-Manager installed.

```bash
# Install custom node from git repo
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

---

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
    "api_key_comfy_org": "optional-partner-node-key"
  }
}
```

- `prompt`: The workflow graph (API format)
- `client_id`: UUID for WebSocket event filtering
- `extra_data.api_key_comfy_org`: Required for paid partner nodes (Flux Pro, Ideogram, etc.)
