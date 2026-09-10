# DS4 Dwarfstar Dashboard — API Reference

## REST Endpoints

### GET /api/status
Full DS4 status, config, and system metrics.

**Response:**
```json
{
  "state": "running",
  "running": true,
  "port": 8001,
  "model": "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32",
  "pid": 12345,
  "uptime_seconds": 3600,
  "checked_at": "2025-06-01T12:00:00Z",
  "telemetry": {
    "kv_cache": {
      "used_slots": 4096,
      "total_slots": 131072,
      "context_utilization_pct": 3.125
    },
    "generation": {
      "tokens_per_second": 42.5,
      "last_generation_ms": 1234
    }
  },
  "config": {
    "binary": "/Users/m4mbp/ds4/ds4-server",
    "primary_port": 8001,
    "model": "/Users/m4mbp/ds4/ds4flash.gguf",
    "mtp": "/Users/m4mbp/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
    "context_window": 131072,
    "kv_disk_cache": "/tmp/ds4-kv",
    "kv_cache_budget_mib": 51200,
    "metal_shader_dir": "/Users/m4mbp/ds4/metal"
  },
  "system": {
    "cpu": {"user_pct": 12.5, "system_pct": 3.2, "idle_pct": 84.3},
    "gpu": {"gpu_pct": 65.0, "cpu_pct": 15.0, "memory_pct": 45.0, "temp_c": 72.0},
    "memory": {"active": 32768, "wired": 16384, "pressure": "ok", "pressure_pct": 65.0},
    "process": {"pid": 12345, "cpu_pct": 42.0, "memory_mib": 2048},
    "smc_temp_c": 68.0
  },
  "kv_cache": {
    "path": "/tmp/ds4-kv",
    "budget_mib": 51200,
    "budget_bytes": 53687091200,
    "disk_used_bytes": 8589934592,
    "disk_fill_percent": 16.0,
    "used_slots": 4096,
    "total_slots": 131072,
    "context_utilization_pct": 3.125
  }
}
```

### GET /api/metrics
Live telemetry subset.

**Response:** Same as `/api/status` but filtered to `checked_at`, `state`, `running`, `port`, `telemetry`, `kv_cache`, `system`.

### GET /api/config
Current config snapshot (defaults + active overrides).

**Response:**
```json
{
  "binary": "/Users/m4mbp/ds4/ds4-server",
  "primary_port": 8001,
  "model": "/Users/m4mbp/ds4/ds4flash.gguf",
  "mtp": "/Users/m4mbp/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
  "context_window": 131072,
  "kv_disk_cache": "/tmp/ds4-kv",
  "kv_cache_budget_mib": 51200,
  "metal_shader_dir": "/Users/m4mbp/ds4/metal",
  "overrides": {
    "context_window": 65536
  }
}
```

### GET /api/config-schema?refresh=true
All discoverable DS4 CLI options.

**Response:**
```json
{
  "context_window": {
    "type": "integer",
    "default": 131072,
    "current": 131072,
    "desc": "Number of tokens to process in context",
    "source": "dashboard",
    "flag": "--ctx-size"
  },
  "model": {
    "type": "string",
    "default": "/Users/m4mbp/ds4/ds4flash.gguf",
    "current": "/Users/m4mbp/ds4/ds4flash.gguf",
    "desc": "Path to model file",
    "source": "dashboard",
    "flag": "--model"
  },
  "port": {
    "type": "integer",
    "default": 8001,
    "current": 8001,
    "desc": "Server port",
    "source": "dashboard",
    "flag": "--port"
  },
  "mtp": {
    "type": "choice",
    "default": "MTP model path",
    "current": "MTP model path",
    "desc": "MTP mode: off, 1, 2, 4",
    "source": "dashboard",
    "flag": "--mtp",
    "choices": ["off", "1", "2", "4"]
  }
}
```

### PATCH /api/config
Apply a config override.

**Request:**
```json
{"key": "context_window", "value": 65536}
```

**Response:**
```json
{
  "ok": true,
  "updated": true,
  "config": {
    "binary": "/Users/m4mbp/ds4/ds4-server",
    "context_window": 65536,
    "overrides": {
      "context_window": 65536
    }
  }
}
```

### DELETE /api/config/{key}
Remove a config override (reverts to default).

**Response:**
```json
{"ok": true, "removed": true, "config": {...}}
```

### GET /api/config-overrides
List active overrides only.

**Response:**
```json
{"overrides": {"context_window": 65536}}
```

### GET /api/benchmarks
Available suites and last results.

**Response:**
```json
{
  "suites": [
    {"id": "quick_smoke", "description": "Single completion, low latency", "default_iterations": 1},
    {"id": "token_throughput", "description": "Token generation throughput test", "default_iterations": 3},
    {"id": "latency_p90", "description": "Latency distribution (P50/P90/P99)", "default_iterations": 5},
    {"id": "kv_cache_stress", "description": "Large-context KV cache throughput", "default_iterations": 3},
    {"id": "mtp_sweep", "description": "Compare MTP modes (off/1/2/4)", "default_iterations": 3},
    {"id": "batch_size_sweep", "description": "Throughput vs batch size", "default_iterations": 3},
    {"id": "context_len_sweep", "description": "Throughput vs context length", "default_iterations": 3}
  ],
  "last_results": null
}
```

### POST /api/benchmarks/run
Execute a benchmark suite.

**Request:**
```json
{"suite_id": "quick_smoke", "iterations": 1, "compare_label": "baseline-20250601"}
```

**Response (single run):**
```json
{
  "suite_id": "quick_smoke",
  "iterations": 1,
  "completed_at": "2025-06-01T12:00:00Z",
  "metrics": {
    "latency_ms": 1234.5,
    "tokens_generated": 100,
    "tokens_per_second": 42.5,
    "context_window": 131072
  }
}
```

**Response (compare mode — with `compare_label`):**
```json
{
  "suite_id": "quick_smoke",
  "iterations": 1,
  "compare_label": "baseline-20250601",
  "completed_at": "2025-06-01T12:00:00Z",
  "metrics": {
    "latency_ms": 1234.5,
    "tokens_generated": 100,
    "tokens_per_second": 42.5
  },
  "compare": {
    "baseline": {
      "label": "baseline-20250601",
      "latency_ms": 1200.0,
      "tokens_per_second": 40.0
    },
    "current": {
      "label": "current",
      "latency_ms": 1234.5,
      "tokens_per_second": 42.5
    },
    "diff": {
      "latency_ms": "+34.5 (+2.9%)",
      "tokens_per_second": "+2.5 (+6.2%)"
    }
  }
}
```

### GET /api/benchmarks/results
Last benchmark results.

**Response:**
```json
{"results": {...}}
```

### GET /api/update/check
Check GitHub for latest DS4 release.

**Response:**
```json
{
  "latest_version": "v1.2.3",
  "published_at": "2025-06-01T10:00:00Z",
  "current_version": "v1.2.2",
  "update_available": true,
  "asset_url": "https://github.com/antirez/ds4/releases/download/v1.2.3/ds4-server-macos-universal.zip",
  "asset_sha256": "abc123..."
}
```

### POST /api/update
Apply a DS4 binary update.

**Request:**
```json
{"apply": true, "asset_url": "...", "sha256": "abc123..."}
```

**Response:**
```json
{
  "applied": true,
  "previous_path": "/Users/m4mbp/ds4/ds4-server",
  "backup_path": "/Users/m4mbp/ds4/ds4-server.bak",
  "restart_required": true
}
```

### GET /api/mcp/manifest
Full MCP tool + resource manifest.

**Response:**
```json
{
  "tools": [
    {"name": "get_status", "description": "...", "inputSchema": {...}},
    {"name": "get_metrics", "description": "...", "inputSchema": {...}},
    {"name": "set_config", "description": "...", "inputSchema": {...}},
    {"name": "get_config", "description": "...", "inputSchema": {...}},
    {"name": "get_schema", "description": "...", "inputSchema": {...}},
    {"name": "run_benchmark", "description": "...", "inputSchema": {...}},
    {"name": "update_ds4", "description": "...", "inputSchema": {...}}
  ],
  "resources": [
    {"uri": "ds4://status", "description": "...", "mimeType": "application/json"},
    {"uri": "ds4://metrics", "description": "...", "mimeType": "application/json"},
    {"uri": "ds4://config", "description": "...", "mimeType": "application/json"},
    {"uri": "ds4://config-schema", "description": "...", "mimeType": "application/json"},
    {"uri": "ds4://benchmarks", "description": "...", "mimeType": "application/json"},
    {"uri": "ds4://benchmarks/last", "description": "...", "mimeType": "application/json"}
  ]
}
```

### POST /mcp (JSON-RPC)
Full MCP protocol endpoint.

**Request (tools/list):**
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```

**Response:**
```json
{"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}
```

**Request (tools/call):**
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_status", "arguments": {}}}
```

**Response:**
```json
{"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "..."}], "structuredContent": {...}}}
```

### GET /mcp/sse (SSE streaming)
Server-sent events stream of telemetry at 2s intervals.

```text
event: telemetry
data: {"checked_at":"...","state":"running","telemetry":{...}}
```
