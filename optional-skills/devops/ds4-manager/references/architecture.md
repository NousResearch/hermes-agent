# DS4 Dwarfstar Dashboard — Architecture

## System Overview

```
┌──────────────────────────────────────────────────────┐
│                  macOS Host (TrinityMBPM5)           │
│                                                      │
│  ┌──────────────┐    ┌───────────────────────────┐   │
│  │   DS4 Server │    │   DS4 Dwarfstar Dashboard │   │
│  │   port 8001  │    │   port 8765               │   │
│  │              │    │                           │   │
│  │  /telem      │◄──│  bridge/engine_client.py  │   │
│  │  /metrics    │    │  (telemetry polling)      │   │
│  │  /v1/chat/   │    │                           │   │
│  │  completions │◄──│  benchmarks/runner.py     │   │
│  │              │    │  (benchmark execution)    │   │
│  └──────────────┘    └───────────────────────────┘   │
│         ▲                                            │
│         │ launchd (com.ds4.server)                   │
│         │                                            │
│  ┌──────┴──────────────────────────────────────┐     │
│  │  DS4 Config Manager (bridge/config_manager) │     │
│  │                                            │     │
│  │  - Parses --help for CLI options            │     │
│  │  - Reads /telem for active values           │     │
│  │  - Applies env-based defaults               │     │
│  │  - Stores overrides (in-memory)             │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  MCP Server (mcp/server.py)                 │   │
│  │  Dual transport:                             │   │
│  │  - JSON-RPC endpoint (/mcp POST)            │   │
│  │  - SSE streaming (/mcp/sse GET)             │   │
│  │  - REST wrappers (/api/mcp/*)               │   │
│  │  Tools: get_status, get_metrics, set_config,│   │
│  │         get_config, run_benchmark,           │   │
│  │         update_ds4, get_schema               │   │
│  │  Resources: ds4://status, ds4://metrics,    │   │
│  │         ds4://config, ds4://config-schema,   │   │
│  │         ds4://benchmarks, ds4://benchmarks/  │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Frontend (static/)                           │   │
│  │  - index.html (Dwarfstar cyberpunk theme)    │   │
│  │  - dashboard.js (SPA, fetches all endpoints) │   │
│  │  - style.css (dark theme, white dwarf SVG)   │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Updater (updater/updater.py)                │   │
│  │  - Checks GitHub releases for DS4 binary     │   │
│  │  - Downloads + verifies SHA256               │   │
│  │  - Replaces binary, triggers restart         │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  System Metrics (bridge/system_metrics.py)   │   │
│  │  - GPU utilization (MPS/iStats)              │   │
│  │  - CPU load (psutil)                         │   │
│  │  - Temperature sensors (macOS SMC)           │   │
│  │  - Memory pressure (vm_stat)                 │   │
│  │  - Disk KV cache usage                       │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## Data Flow

### Status Polling
```
Browser / Agent
    │
    ▼
GET /api/status
    │
    ▼
dashboard.py
    ├─► engine_client.get_status()  →  GET http://127.0.0.1:8001/telem
    ├─► config_manager.get_config()  →  merge defaults + overrides
    └─► system_metrics.get_metrics() →  psutil + SMC + KV cache stat
    │
    ▼
JSON response: {state, uptime, model, port, config, system, kv_cache}
```

### Config Schema Discovery
```
GET /api/config-schema?refresh=true
    │
    ▼
config_manager.get_schema(force_refresh=true)
    ├─► subprocess: ds4-server --help  →  parse flags
    ├─► GET /telem  →  read active values
    └─► merge dashboard defaults
    │
    ▼
JSON response: {key: {type, default, current, desc, source, flag, choices}}
```

### Config Override
```
PATCH /api/config  {"key": "context_window", "value": 65536}
    │
    ▼
config_manager.set_override(key, value)
    │
    ▼
Stored in memory (_overrides dict)
get_config() now merges defaults + overrides
    │
    ▼
Return: {ok, updated, config}
```

### Benchmark Execution
```
POST /api/benchmarks/run  {"suite_id": "quick_smoke", "iterations": 1}
    │
    ▼
benchmark_runner.run_suite("quick_smoke")
    ├─► engine_client.completion(...)  →  POST /v1/chat/completions
    ├─► measure tokens, latency, throughput
    └─► store result with compare_label (if provided)
    │
    ▼
JSON response: {suite_id, iterations, metrics, compare?}
```

### MCP Dual Transport
```
Transport 1: JSON-RPC over POST
POST /mcp  {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "get_status"}}

Transport 2: REST wrappers
POST /api/mcp/tools/get_status

Transport 3: SSE streaming
GET /mcp/sse  →  event: telemetry  data: {...}  (every 2s)
```

## Component Dependencies

```
dashboard.py
  ├── benchmarks/runner.py
  │     └── bridge/engine_client.py (completions)
  ├── bridge/config_manager.py
  │     ├── subprocess (ds4-server --help)
  │     └── bridge/engine_client.py (/telem)
  ├── bridge/engine_client.py
  │     └── httpx (async HTTP to DS4)
  ├── bridge/system_metrics.py
  │     ├── psutil (CPU/memory)
  │     └── subprocess (iStats GPU temps, SMC)
  ├── mcp/server.py
  │     ├── mcp/tools.py (tool registry)
  │     └── mcp/resources.py (resource registry)
  ├── updater/updater.py
  │     └── httpx (GitHub API)
  └── static/ (frontend assets)
```

## Ports

| Port | Service | Protocol |
| --- | --- | --- |
| 8001 | DS4 Server | HTTP (OpenAI API + telemetry) |
| 8765 | DS4 Dashboard | HTTP (REST + MCP + static) |
| 7777 | DS4 Telemetry | HTTP (internal, /telem endpoint) |

## Key Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DS4_HOME` | `~/ds4` | DS4 installation directory |
| `DS4_BINARY` | `$DS4_HOME/ds4-server` | DS4 server binary path |
| `DS4_MODEL` | `$DS4_HOME/ds4flash.gguf` | Model path |
| `DS4_MTP` | `$DS4_HOME/gguf/...MTP...gguf` | MTP model path |
| `DS4_METAL_DIR` | `$DS4_HOME/metal` | Metal shader directory |
| `DS4_KV_CACHE` | `/tmp/ds4-kv` | KV disk cache path |
| `DS4_PRIMARY_PORT` | `8001` | DS4 server port |
| `DS4_CONTEXT_WINDOW` | `131072` | Context window size |
| `DS4_KV_CACHE_BUDGET_MIB` | `51200` | KV cache budget (50 GiB) |
| `DS4_GITHUB_REPO` | `antirez/ds4` | GitHub repo for updates |
