---
name: ds4-manager
description: "Manage the DS4 Dwarfstar Dashboard — REST API, MCP tools, benchmark suites, config browser, updater, and launchd integration for DS4 inference engine. Triggers: ds4-dashboard, dwarfstar, ds4-manager, ds4 dashboard, ds4 telemetry"
version: 1.0.0
author: shagghiesuperstar
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [ds4, dashboard, inference, devops, macos, launchd]
    related_skills: [ds4-system-health, ds4-deepseek-tui-opencode-local-coding]
---

# DS4 Dwarfstar Dashboard Manager

Manage the DS4 Dwarfstar Dashboard — a FastAPI dashboard running on port 8765 that provides REST + MCP dual transport for DS4 inference engine telemetry, config management, benchmark execution, and binary updates.

## When to Use

- User asks to check DS4 status, metrics, KV cache, GPU/CPU temps
- User wants to browse or change DS4 config options via the dashboard
- User wants to run a benchmark suite and compare results
- User wants to check for or apply a DS4 binary update
- User wants to restart DS4 after config changes (launchd integration)
- User asks about the config schema browser or MCP tools

## Prerequisites

- DS4 server running on port 8001 (or configured port)
- DS4 Dashboard running on port 8765 (`dashboard.py`)
- DS4 binary at `~/ds4/ds4-server` and model at `~/ds4/ds4flash.gguf` (or env overrides)
- Python 3.9+ venv in the dashboard directory

## Setup

### Clone and start the dashboard

```bash
cd ~/ds4-dashboard
source .venv/bin/activate
python dashboard.py
# Starts on http://127.0.0.1:8765
```

### Launchd integration

A launchd plist (`com.ds4.dashboard.plist`) can manage the dashboard process and restart DS4 after config changes. See `scripts/start.sh` and the launchd section below.

## REST API Endpoints

All endpoints are at `http://127.0.0.1:8765`.

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Static frontend (index.html) |
| `/api/status` | GET | Full DS4 status: uptime, model, port, config snapshot, system metrics, KV cache |
| `/api/metrics` | GET | Live telemetry subset: telemetry, KV cache, system metrics |
| `/api/config` | GET | Current config snapshot (defaults + overrides) |
| `/api/config-schema` | GET | All discoverable DS4 CLI options with types, defaults, choices |
| `/api/config` | PATCH | Apply a config override (`{"key": "...", "value": ...}`) |
| `/api/config/{key}` | DELETE | Remove a specific config override |
| `/api/config-overrides` | GET | List current active overrides only |
| `/api/benchmarks` | GET | Available benchmark suites + last results |
| `/api/benchmarks/run` | POST | Run a benchmark suite (`suite_id`, `iterations`, `compare_label`) |
| `/api/benchmarks/results` | GET | Last benchmark results |
| `/api/update/check` | GET | Check GitHub for latest DS4 release |
| `/api/update` | POST | Apply a DS4 binary update from GitHub |
| `/api/mcp/manifest` | GET | Full MCP tool + resource manifest |
| `/api/mcp/tools/{tool_name}` | POST | Call an MCP tool by name |
| `/api/mcp/resources` | GET | List available MCP resources |
| `/api/mcp/resources/read` | GET | Read an MCP resource by URI (`?uri=...`) |
| `/mcp` | POST | MCP JSON-RPC endpoint (full protocol) |
| `/mcp/sse` | GET | SSE stream of telemetry events (2s interval) |

## MCP Tools (JSON-RPC)

Available via `/mcp` (POST) or `/api/mcp/tools/{tool_name}`.

| Tool | Input | Returns |
| --- | --- | --- |
| `get_status` | {} | Full status: uptime, model, port, config, system, KV cache |
| `get_metrics` | {} | Live telemetry, KV cache, GPU/CPU/temps |
| `get_config` | {} | Current config snapshot (defaults + overrides) |
| `set_config` | `key`, `value` | Apply override, returns updated config |
| `get_schema` | {} | All discoverable DS4 CLI options |
| `run_benchmark` | `suite_id`, `iterations`, `compare_label` | Run a suite, optionally compare with previous |
| `update_ds4` | `apply`, `asset_url`, `sha256` | Check or apply a DS4 binary update |

## MCP Resources (JSON-RPC)

Available via `/mcp` (POST) with `resources/list` or `resources/read`.

| Resource URI | Description |
| --- | --- |
| `ds4://status` | Full DS4 status snapshot |
| `ds4://metrics` | Live telemetry metrics |
| `ds4://config` | Current config snapshot |
| `ds4://config-schema` | Config schema browser |
| `ds4://benchmarks` | Available suites + last results |
| `ds4://benchmarks/last` | Last benchmark results only |

## Benchmark Suites

The dashboard defines benchmark suites for measuring DS4 performance:

| Suite ID | Description | Default Iterations |
| --- | --- | --- |
| `quick_smoke` | Single completion, low latency | 1 |
| `token_throughput` | Token generation throughput test | 3 |
| `latency_p90` | Latency distribution (P50/P90/P99) | 5 |
| `kv_cache_stress` | Large-context KV cache throughput | 3 |
| `mtp_sweep` | Compare MTP modes (off/1/2/4) | 3 |
| `batch_size_sweep` | Throughput vs batch size | 3 |
| `context_len_sweep` | Throughput vs context length | 3 |

### Compare mode

Pass `compare_label` when running a benchmark to label the result. The dashboard stores previous results and returns a `compare` field with both old and new for side-by-side diffing.

## Config Schema Browser

The dashboard auto-discovers DS4 CLI options by:
1. Parsing `--help` output for flag definitions
2. Reading telemetry `/telem` for active config values
3. Applying dashboard defaults from environment variables

Access via `GET /api/config-schema?refresh=true` to force re-discovery.

Each config option shows:
- `type`: string/integer/float/boolean/choice
- `default`: dashboard default value
- `current`: active value (from telemetry or override)
- `desc`: description from `--help`
- `source`: "dashboard", "telemetry", or "cli"
- `flag`: CLI flag (e.g. `--model`, `--ctx-size`)
- `choices`: enum choices for choice-type options

### Setting config overrides

```bash
# Set a config value
curl -X PATCH http://127.0.0.1:8765/api/config \
  -H 'Content-Type: application/json' \
  -d '{"key": "context_window", "value": 65536}'

# Remove an override (reverts to default)
curl -X DELETE http://127.0.0.1:8765/api/config/context_window
```

**After config changes, restart DS4 with the launchd integration** (see launchd section below).

## Launchd Integration

The DS4 dashboard can be managed via launchd for automatic restart after config changes.

### Plist: `~/Library/LaunchAgents/com.ds4.dashboard.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ds4.dashboard</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/m4mbp/ds4-dashboard/scripts/start.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>/Users/m4mbp/ds4-dashboard</string>
    <key>StandardOutPath</key>
    <string>/tmp/ds4-dashboard.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/ds4-dashboard-err.log</string>
</dict>
</plist>
```

### Restarting DS4 after config changes

```bash
# After setting config overrides, restart DS4
# The dashboard scripts/restart-ds4.sh handles this
scripts/restart-ds4.sh
# Or via launchd:
launchctl kickstart gui/$(id -u)/com.ds4.server
```

## Verification

```bash
# Check dashboard is running
curl -s http://127.0.0.1:8765/api/status | jq .

# Check metrics
curl -s http://127.0.0.1:8765/api/metrics | jq .

# List available config options
curl -s http://127.0.0.1:8765/api/config-schema | jq .

# Run quick smoke benchmark
curl -X POST http://127.0.0.1:8765/api/benchmarks/run \
  -H 'Content-Type: application/json' \
  -d '{"suite_id": "quick_smoke", "iterations": 1}'
```

## Troubleshooting

- **Dashboard won't start**: ensure `.venv/bin/activate` exists and dependencies are installed (`pip install -r requirements.txt`)
- **Status shows "stopped"**: DS4 server is not running — start DS4 via launchd or `ds4-server`
- **Config schema empty**: ensure DS4 binary path is correct and `--help` parsing works
- **Benchmarks fail**: check DS4 is running and accepting completions on the configured port
- **Update fails**: verify GitHub repo access and binary path write permissions
