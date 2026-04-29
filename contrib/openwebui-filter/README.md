# Hermes -> Open WebUI Filter Function

Eliminates UI lag when connecting Hermes Agent to Open WebUI via the API Server.

## The Problem

When Hermes streams long responses with tool calls through Open WebUI, the browser
freezes, scrolling becomes choppy, and the UI becomes unresponsive.

**Three root causes:**

1. **SSE event storm** — Hermes sends one SSE event per token (~500 events per 20s
   response). Open WebUI re-renders markdown on every event.

2. **DOM bloat from tool call JSON** — `write_file` with 24KB markdown creates
   ~300+ DOM nodes per tool card. No virtual scrolling.

3. **Giant `response.completed` payload** — All tool calls + outputs packed into one
   SSE line (400-848KB), silently hanging Open WebUI's SSE parser.

## The Solution (Two Layers)

### Layer 1: Server-Side SSE Token Batching

Implemented in Hermes API Server (PR #17541). Buffers text deltas for 50ms and
emits as a single event, reducing ~500 events to ~20 per response.

### Layer 2: Open WebUI Filter Function (this file)

Intercepts SSE events in Open WebUI's middleware pipeline, beautifies tool call
arguments, and trims oversized payloads before they reach the frontend renderer.

## Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| SSE events per 20s response | ~500 | ~20 | **-96%** |
| DOM nodes per tool card | ~300+ | ~5 | **-98%** |
| Frame render time | 600ms | ~80ms | **-87%** |
| response.completed payload | 848 KB | ~8 KB | **-99%** |
| CPU during streaming | 100% (frozen) | <20% | **solved** |
| UI freezing | yes | none | **solved** |

## Key Features

- **Emitter beautify** — 15+ tool emoji summaries (e.g., `path (24.5 KB)` instead of raw JSON)
- **Output summaries** — JSON results -> one-liner (e.g., `5 results` instead of full JSON)
- **call_id -> name tracking** — accurate tool name resolution across added/done event pairs
- **Multi-part output handling** — processes all output parts, not just `output[0]`
- **response.completed trimming** — 848KB -> ~8KB (largest single performance win)
- **Inline-output hint** — injects system note to encourage Hermes to output inline

## Deployment

### Persistent (survives restarts)

1. Copy `filter-function-v3.py` to Open WebUI's `data/functions/` directory
2. Restart Open WebUI (auto-discovers functions)
3. Admin Settings -> Functions -> activate "Hermes Tool Sanitizer" -> set Global

For Windows Store Python pip installs:

```
OW_DATA="%LOCALAPPDATA%\\Packages\\PythonSoftwareFoundation.Python.3.12_*\\
  LocalCache\\local-packages\\Python312\\site-packages\\open_webui\\data"
mkdir "%OW_DATA%\\functions"
copy filter-function-v3.py "%OW_DATA%\\functions\\"
```

### Quick (via API)

```bash
TOKEN=$(curl -s http://127.0.0.1:7899/api/v1/auths/signin \
  -H "Content-Type: application/json" \
  -d '{"email":"...","password":"..."}' | jq -r .token)

CODE=$(cat filter-function-v3.py | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")

curl -s http://127.0.0.1:7899/api/v1/functions/create \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"hermes-tool-sanitizer\",\"name\":\"Hermes Tool Sanitizer\",\"type\":\"filter\",\"content\":$CODE,\"is_active\":true,\"is_global\":true}"
```

## Related

- [hermes-agent#17537](https://github.com/NousResearch/hermes-agent/issues/17537) — Feature Request
- [hermes-agent#17541](https://github.com/NousResearch/hermes-agent/pull/17541) — SSE Batching PR
- [open-webui#20878](https://github.com/open-webui/open-webui/discussions/20878) — UI freeze investigation
- [open-webui#21884](https://github.com/open-webui/open-webui/pull/21884) — Frontend perf PR
