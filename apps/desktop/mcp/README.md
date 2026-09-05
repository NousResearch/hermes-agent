# Hermes Desktop Debug MCP

Native UI-debugging tools for LLM agents working on `apps/desktop`. Wraps the
perf-harness CDP client (`scripts/perf/lib/cdp.mjs`) so agents inspect and drive
the live renderer without hand-rolling `eval.mjs` one-liners each session.

Proposal thread: NousResearch/hermes-agent#95489.

## Tools

Read-only (always available):

| Tool | What it does |
|---|---|
| `desktop_ui_status` | Is the CDP port alive? Which targets/selectors exist? Call first. |
| `ui_inspect` | One element: tag, classes, box, visibility, computed styles, inherited-rule hint. |
| `ui_query` | Up to 20 matching elements with bounded text snippets. |
| `ui_console` | Renderer console ring captured while connected. |
| `ui_screenshot` | Capture the window as a PNG returned as MCP image content (no disk write). |

Mutating (require `DESKTOP_DEBUG_MCP_ALLOW_ACT=1` in the server env):

| Tool | What it does |
|---|---|
| `ui_click` / `ui_type` / `ui_press` | Real CDP Input events — blur/focus semantics match a human (this matters: synthetic DOM events skip the blur→cancel race the edit composer is known for). |
| `ui_eval` | Bounded JS eval escape hatch. |
| `ui_flow_edit` | Scripted edit flow: open edit on last user message → type → Enter → structured report (send accepted? composer stuck? timeline changed?). Reproduction harness for the chat-edit silent-fail races. |
| `ui_flow_model_switch` | Installs a MutationObserver over the thread to quantify model-switch row jank. |

## Running

```bash
cd apps/desktop/mcp
npm install
node server.mjs                       # stdio MCP server, port 9222 by default
```

Register with Hermes:

```bash
hermes mcp add desktop-debug \
  --command node \
  --args <abs-path>/apps/desktop/mcp/server.mjs
# mutating tools:
hermes mcp add desktop-debug --command node \
  --args <abs-path>/apps/desktop/mcp/server.mjs \
  --env DESKTOP_DEBUG_MCP_ALLOW_ACT=1
```

Flags/env: `--port N` / `DESKTOP_DEBUG_MCP_PORT`, `--match STR` (target URL filter,
default `5174`), `DESKTOP_DEBUG_MCP_ALLOW_ACT=1`.

## The port problem (read this first)

The CDP port exists **only for dev-server runs**; packaged builds never open it.
If `desktop_ui_status` reports `cdpAlive: false`:

1. Ask the user to start the dev server (`cd apps/desktop && npm run dev`), or
2. Launch an isolated probe instance (does not touch the user's app):

```bash
cd apps/desktop
HERMES_HOME=/tmp/cdp-probe-home \
HERMES_DESKTOP_DEV_SERVER=http://127.0.0.1:5174 \
HERMES_DESKTOP_CDP_PORT=9333 \
  npx electron . --user-data-dir=/tmp/cdp-probe-userdata
# then: node mcp/server.mjs --port 9333
```

**Never relaunch or kill the user's running app** to get a port.

## Safety rails

- Outputs are bounded (≤20 nodes, ≤80-char snippets, ≤4KB eval) — never dump full DOM.
- Mutating tools are opt-in via env; flows refuse to run without it.
- One shared connection; friendly errors instead of raw discovery dumps.
- Real input events only — no synthetic `dispatchEvent` shortcuts.
- **Isolation guard — target attestation (fail-closed).** Every consequential tool
  (read *and* mutating) refuses to run unless the *connected target proves* it is
  the isolated sandbox you declared. The Electron main process that opens the dev
  CDP port emits a per-instance descriptor (`__DEBUG_MCP_INSTANCE__ = { nonce,
  dataRoot }`) into the renderer. `dataRoot` is the **realized** Hermes home from
  the single home authority (`electron/hermes-home.ts`) — the same value the app
  actually uses, so a `HERMES_DESKTOP_USER_DATA_DIR`-only sandbox launch attests
  `<userData>/hermes-home`, never `~/.hermes`. The server reads it over CDP and
  checks two independent policies:
  1. **Protected home refusal** — if the realized home equals the server's default
     home (`HERMES_HOME` env or `~/.hermes`) or the OS user's literal `~/.hermes`,
     the call is REFUSED even when the declared `EXPECTED_HOME` agrees. Agreement
     is not permission.
  2. **Identity match** — the realized home must (canonically) equal
     `DESKTOP_DEBUG_MCP_EXPECTED_HOME`.
  If the env var is unset, the target exposes no descriptor, or either policy
  fails, every call returns `REFUSED`. This is **target-derived authority, not a
  caller declaration** — so a real dev Desktop running on `~/.hermes` cannot be
  reached by merely declaring a fake `DESKTOP_DEBUG_MCP_EXPECTED_HOME`. It
  prevents a debug run from reading/writing your real API keys and chat history —
  the exact failure mode of the 2026-08-26 incident where a manual `electron .`
  launch (with only `HERMES_DESKTOP_USER_DATA_DIR` set) silently used `~/.hermes`
  as `HERMES_HOME`.
- **`desktop_ui_status` is a preflight.** It deliberately skips connection and
  attestation so it can report `cdpAlive: false` when no dev instance is running.
  Every other tool (read included) still connects and attests first.

## Launching an isolated probe instance (REQUIRED before any act/flow)

Never point a debug instance at your real `~/.hermes`. Always give it its own
home with a mock provider config, and tell the server that home:

```bash
# 1. isolated home + mock config
mkdir -p /tmp/hermes-debug-home
cat > /tmp/hermes-debug-home/config.yaml <<'YAML'
model: { default: mock-model, provider: mock }
providers:
  mock: { api: http://127.0.0.1:53999/v1, name: Mock, api_mode: chat_completions,
          key_env: MOCK_API_KEY, models: { mock-model: {} }, context_length: 4096 }
YAML
echo "MOCK_API_KEY=debug" > /tmp/hermes-debug-home/.env
# 2. start vite + electron against that home, with CDP
cd apps/desktop
( npm run dev:renderer & )
HERMES_HOME=/tmp/hermes-debug-home \
HERMES_DESKTOP_PYTHON=<path-to-venv>/bin/python \
HERMES_DESKTOP_CDP_PORT=9333 \
HERMES_DESKTOP_DEV_SERVER=http://127.0.0.1:5174 \
HERMES_DESKTOP_USER_DATA_DIR=/tmp/cdp-probe-userdata \
HERMES_DESKTOP_IGNORE_EXISTING=1 \
  npx electron . --user-data-dir=/tmp/cdp-probe-userdata
# 3. start the server DECLARING the same home
DESKTOP_DEBUG_MCP_ALLOW_ACT=1 \
DESKTOP_DEBUG_MCP_EXPECTED_HOME=/tmp/hermes-debug-home \
DESKTOP_DEBUG_MCP_PORT=9333 \
  node mcp/server.mjs
```

If you skip step 3's `DESKTOP_DEBUG_MCP_EXPECTED_HOME`, all mutating tools
return `REFUSED`. If you skip step 1's isolated `HERMES_HOME`, the instance
falls back to `~/.hermes` and the guard blocks it anyway.
