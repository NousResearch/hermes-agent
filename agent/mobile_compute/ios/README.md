# Mobile Compute Node — iOS Implementation

This is the Swift/NIO implementation of the Mobile Compute server for iPhone 16e.

## Architecture

```
Hermes (Windows)
    ↓
MobileComputeBridge (Python)
    ↓
HTTP
    ↓
iPhone 16e (This Swift Server)
    ↓
```

## API Endpoints

### GET /health
Returns the health status of the iPhone.

**Response:**
```json
{
  "status": "ok",
  "device": "iPhone 16e"
}
```

**Compatible with:** `MobileComputeClient.health()` in `client.py`

---

### GET /capabilities
Returns the capabilities of the iPhone.

**Response:**
```json
{
  "device": "iPhone 16e",
  "compute": ["echo"],
  "coreml": false,
  "llm": false
}
```

**Compatible with:** `MobileComputeClient.capabilities()` in `client.py`

---

### POST /compute
Submits a compute task to the iPhone.

**Request body:**
```json
{
  "task_id": "optional-task-id",
  "task_type": "echo",
  "payload": {
    "text": "hello"
  }
}
```

**Response:**
```json
{
  "success": true,
  "task_id": "task-id",
  "status": "completed",
  "result": "hello"
}
```

**Compatible with:** `MobileComputeClient.submit_task()` in `client.py`

---

### GET /tasks
Lists all tasks.

**Response:**
```json
{
  "tasks": [...]
}
```

**Compatible with:** `MobileComputeClient.list_tasks()` in `client.py`

---

### GET /tasks/{task_id}
Gets the status of a specific task.

**Response:**
```json
{
  "task_id": "task-id",
  "task_type": "echo",
  "status": "completed",
  "payload": {...},
  "result": "hello",
  "error": null
}
```

**Compatible with:** `MobileComputeClient.get_task_status()` in `client.py`

## Configuration

The server can be configured via environment variables:

- `ENABLED`: Set to `true` to enable the server (default: true)
- `HOST`: Host to bind to (default: "0.0.0.0")
- `PORT`: Port to listen on (default: 8765)
- `TLS_ENABLED`: Enable TLS (default: false)
- `TLS_CERT_PATH`: Path to certificate file
- `TLS_KEY_PATH`: Path to private key file
- `TLS_CA_PATH`: Path to CA certificate file

## Building

```bash
# Build
swift build

# Build for release
swift build -c release

# Run
swift run MobileCompute
```

## Security

- By default, binds to `0.0.0.0` (all interfaces)
- For local network only usage, configure your WiFi network appropriately
- TLS/mTLS support is available but not enabled by default
- No authentication is enforced by default (local network only)

## Phase 0

This is Phase 0 implementation:
- No Core ML integration
- No LLM calls
- Echo tasks only (scaffold for future compute types)