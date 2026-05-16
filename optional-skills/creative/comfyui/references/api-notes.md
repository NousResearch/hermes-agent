# ComfyUI REST API Notes

The `comfyui-skill` CLI wraps these endpoints. This reference is for debugging,
understanding errors, or advanced use when the CLI doesn't cover a specific need.

## Endpoints the CLI Uses

| Endpoint | Method | CLI Command |
|----------|--------|-------------|
| `/system_stats` | GET | `server status`, `server stats` |
| `/prompt` | POST | `run`, `submit` |
| `/history/{prompt_id}` | GET | `status`, `run` (polling) |
| `/history` | GET | `history list --server` |
| `/queue` | GET | `queue list` |
| `/queue` | POST | `queue clear`, `queue delete` |
| `/interrupt` | POST | `cancel` |
| `/free` | POST | `free` |
| `/object_info` | GET | `nodes list`, `workflow import` (schema extraction) |
| `/object_info/{class}` | GET | `nodes info` |
| `/models` | GET | `models list` |
| `/models/{folder}` | GET | `models list <folder>`, `deps check` |
| `/view` | GET | `run` (output download) |
| `/upload/image` | POST | `upload` |
| `/upload/mask` | POST | `upload --mask` |
| `/node_replacements` | GET | `workflow import` (deprecated node detection) |
| `/internal/logs/raw` | GET | `logs show` |
| `/workflow_templates` | GET | `templates list` |
| `/global_subgraphs` | GET | `templates subgraphs` |
| `/v2/userdata` | GET | `workflow import --from-server` |
| `/ws` | WebSocket | `run` (real-time progress) |

### Cloud-specific

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/jobs` | GET | Job listing with filtering |
| `/api/jobs/{id}` | GET | Job details |

### ComfyUI Manager (optional plugin)

| Endpoint | Method | CLI Command |
|----------|--------|-------------|
| `/manager/queue/start` | GET | `deps install` |
| `/manager/queue/install` | POST | `deps install` (custom nodes) |
| `/manager/queue/install_model` | POST | `deps install --models` |
| `/manager/queue/status` | GET | `deps install` (progress) |

## Local vs Cloud Differences

| | Local | Cloud |
|---|---|---|
| Base URL | `http://127.0.0.1:8188` | `https://cloud.comfy.org` |
| Route prefix | none | `/api` |
| Auth | none or bearer token | `X-API-Key` header |
| Job status | Poll `/history/{id}` | `/api/jobs/{id}` |
| Output download | Direct bytes from `/view` | 302 redirect → signed URL |
| WebSocket | `ws://host:port/ws?clientId={uuid}` | `wss://host/ws?clientId={uuid}&token={key}` |
| Concurrent jobs | Sequential | Tier-limited (Free: 1, Creator: 3, Pro: 5) |

The CLI handles all of these differences transparently based on the server config.

## Workflow JSON Format (API Format)

```json
{
  "node_id_string": {
    "class_type": "NodeClassName",
    "inputs": {
      "param_name": "value",
      "linked_input": ["source_node_id", output_index]
    }
  }
}
```

- Node IDs are strings (`"3"`, not `3`)
- Links: `["node_id", output_index]` — 0-based int
- `class_type` must match exactly (case-sensitive)

## POST /prompt Payload

```json
{
  "prompt": { "<workflow>" },
  "client_id": "uuid",
  "extra_data": {
    "api_key_comfy_org": "key-for-paid-api-nodes"
  }
}
```

The CLI constructs this from the imported workflow + injected parameters.

## WebSocket Message Types

| Type | When | Key Fields |
|------|------|------------|
| `execution_start` | Prompt begins | `prompt_id` |
| `executing` | Node running (`null` = done) | `node`, `prompt_id` |
| `progress` | Sampling steps | `node`, `value`, `max` |
| `executed` | Node output ready | `node`, `output` |
| `execution_success` | All nodes done | `prompt_id` |
| `execution_error` | Failure | `exception_type`, `exception_message` |
