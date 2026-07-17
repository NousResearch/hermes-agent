# boardstate-hermes-plugin

A drop-in dashboard plugin for [Hermes](https://github.com/NousResearch/hermes-agent)
(`hermes dashboard`) that brings a **live [Boardstate](https://github.com/100yenadmin/boardstate)
board** into the Hermes UI — the layout-as-data workspace an agent **builds and operates
through tools**, rendered live over a WebSocket to a local sidecar, styled to match the
active Hermes theme, and bound to real Hermes data.

The plugin adds a **Board** tab (after *Skills*). Opening it mounts the real
`<boardstate-view>` custom element over the Boardstate networked transport, backed by a
Node sidecar that owns the control plane.

![The Hermes agent composes a board live from a natural-language prompt](.github/media/flagship-agent-builds-board.png)

*The Hermes agent, from a plain-English prompt, discovers the widget catalog and composes
a live board — rendered inside Hermes' own chrome, zero error cells.*

## Features

- **The agent builds the board live.** Hermes reaches the board through a networked MCP
  endpoint on the sidecar; every `boardstate_*` tool call lands on the same host the
  browser is subscribed to, so widgets appear as the agent works — no reload.
- **Live Hermes data.** Data-source widgets (usage, sessions, instances, cron) and
  `source:"rpc"` bindings resolve against the live Hermes REST surface
  (`/api/analytics/usage`, `/api/sessions`, `/api/status`, cron) — real numbers, graceful
  empty states, never an error cell. Credentials are injected server-side only and never
  reach the browser or the workspace doc.
- **Native theming.** The board follows the active Hermes palette by aliasing Boardstate's
  `--bs-*` tokens to Hermes' `--color-*` / `--*-base` tokens, and **auto-follows live
  palette swaps** with zero JavaScript. Light and dark both render at WCAG-safe contrast.
- **One-click templates.** A picker swaps in a ready-made, live-bound board — *Agent HQ*,
  *Usage & Cost*, *Sessions Monitor* — each built from self-binding data widgets.
- **Secure by construction.** The sidecar's WS and MCP endpoints are gated by a per-spawn
  nonce; operator-only methods (`widget.approve` / `capability.approve` / `action.confirm`)
  are blocked over the networked transport; the browser authenticates through the
  dashboard's own session gate.

| ![Native teal theme + live data](.github/media/live-data.png) | ![Agent HQ template](.github/media/template-agent-hq.png) |
|:--:|:--:|
| Live usage/sessions data in the native Hermes theme | The *Agent HQ* template, one click |

## Architecture

```
browser  <boardstate-view> + createWsTransport         theme adapter aliases --bs-* → Hermes --color-*
   │  ws   /api/plugins/boardstate/ws                   (SDK buildWsUrl → authed by the Hermes WS gate)
   │  http /api/plugins/boardstate/mcp   ◄─ Hermes agent's MCP client (StreamableHTTP)
   ▼
plugin_api.py  ── FastAPI: WS bridge + MCP proxy + sidecar lifecycle
   │             injects a per-spawn nonce + (server-side only) Hermes URL + session token
   │  ws/http  127.0.0.1:<ephemeral>?nonce=…
   ▼
sidecar/server.js  ── @boardstate/server control plane (createInProcessHost +
   │                   registerBoardstateRpc + attachWsTransport), one MCP endpoint on the
   │                   SAME host, and read-scoped RPC handlers that resolve rpc data
   │                   bindings against Hermes REST
   ▼
$HERMES_HOME/boardstate-state/dashboard/workspace.json
```

Three things share **one** sidecar host so a write from any of them updates every live
view: the browser (over the WS bridge), the Hermes agent (over the MCP proxy), and the
data resolver (over Hermes REST).

### Why a WebSocket proxy (not a direct browser→sidecar connection)

The browser connects to the **dashboard origin**, so auth is the dashboard's canonical WS
gate (`web_server._ws_auth_ok`) — the same gate the bundled *kanban* plugin uses. It
accepts the right credential in every mode (loopback `?token=`, gated single-use
`?ticket=`, server-internal `?internal=`) and works under `--host` / gated OAuth / HTTPS
where a direct `ws://127.0.0.1:<port>` from the page would be blocked or unreachable. The
sidecar binds `127.0.0.1` only, behind a per-spawn nonce, and is never exposed to the
browser.

### How live rpc data bindings resolve

`<boardstate-view>` resolves a `source:"rpc"` binding by calling the binding's **method
directly as a networked RPC** (`transport.request("usage.status", …)`), not via
`dashboard.data.read`. The sidecar registers each allowlisted data method
(`usage.status` / `usage.cost` / `system-presence` / `sessions.list` / `cron.list` /
`node.list`) as a **read-scoped** RPC handler that maps to a Hermes REST call. The
dedicated data-source builtins self-bind to these methods, so a template shows real data
with no manual bindings.

### File tree

```
dashboard/
├── manifest.json          tab "Board" (/board), entry dist/index.js, css, api
├── plugin_api.py          FastAPI: WS bridge + MCP proxy + sidecar lifecycle + nonce + creds
├── src/
│   ├── index.tsx           React tab → mounts <boardstate-view>; theme adapter; template picker
│   ├── theme.ts            pure Hermes→Boardstate token mapping (unit-tested)
│   └── templates.ts        live-bound board templates
├── dist/index.js          built browser bundle (IIFE, React external)
├── sidecar/
│   ├── src/{server,mcp,hermes-data,chat-translate}.ts
│   └── server.js           built self-contained ESM bundle (all @boardstate/* inlined)
└── vendor/                 @boardstate/lit/browser bundle + stylesheet
build.mjs                   esbuild driver (resolves @boardstate/* from npm)
```

The built artifacts (`dist/`, `sidecar/server.js`, `vendor/*`) are committed — the plugin
is a **runtime drop-in and does no npm resolution at runtime** (only `node`).

## Install

Requires Node ≥ 20 on the machine running `hermes dashboard`.

```bash
# 1. Drop the plugin into the Hermes user-plugin dir (note the dashboard/ subdir).
mkdir -p ~/.hermes/plugins/boardstate
cp -r dashboard ~/.hermes/plugins/boardstate/dashboard

# 2. Enable it — user plugins are gated by plugins.enabled in ~/.hermes/config.yaml:
#    plugins:
#      enabled:
#        - boardstate

# 3. (Re)start the dashboard (backend API routes mount at startup).
hermes dashboard
```

Open the dashboard; the **Board** tab appears after *Skills*.

### Let the agent build it

Register the plugin's MCP endpoint so Hermes can build and operate the board via tools —
in `~/.hermes/config.yaml` (or a profile):

```yaml
mcp_servers:
  boardstate:
    url: http://127.0.0.1:9119/api/plugins/boardstate/mcp   # your dashboard port
    headers:
      X-Hermes-Session-Token: <dashboard session token>
```

Then ask Hermes, e.g. *"add a stat card showing 7 active agents"* — the widget appears on
the board live. Or click a **template** in the board toolbar for an instant live board.

### Desktop app

The same board runs in the Hermes **desktop app** (Electron) as a first-class page. The
desktop frontend is a single self-contained ESM `plugin.js` (boardstate is inlined,
because the desktop loader only resolves `@hermes/plugin-sdk` / `react*`); it reuses the
**exact same backend** — the Python `plugin_api.py` + sidecar the web tab uses — reaching
it over the desktop bridge (`window.hermesDesktop.getConnection()` → the same
`/api/plugins/boardstate/ws`). It self-styles to the desktop's `--ui-*` theme tokens.

```bash
# Install the Python backend (as above) AND the desktop frontend plugin:
mkdir -p ~/.hermes/desktop-plugins/boardstate
cp dashboard/desktop/plugin.js ~/.hermes/desktop-plugins/boardstate/plugin.js
```

The desktop app watches that directory — the **Board** page appears in the sidebar within
a few seconds (or ⌘K → *Reload desktop plugins*). On an OAuth remote the live board needs
a local gateway (single-use WS tickets); a poll fallback is planned.

![The board in the desktop app](.github/media/desktop-board.png)

### Config

| Setting | Default | Notes |
|---------|---------|-------|
| Board state dir | `$HERMES_HOME/boardstate-state` | Override with `BOARDSTATE_HERMES_STATE_DIR`. |
| Node binary | `node` on `PATH` | Override with `HERMES_NODE_BIN`. |
| Live Hermes data | on when the dashboard injects its URL + session token | The sidecar reads Hermes REST server-side only; the token never reaches the browser. |

## Do / Observe

| Do | Observe |
|----|---------|
| Open the **Board** tab | "Board connected"; the board renders in the active Hermes palette |
| Ask the agent to add a widget | The widget appears live, no reload |
| Click a **template** (e.g. *Agent HQ*) | The board swaps to a live-bound workspace; usage/sessions/instances/cron resolve real data with graceful empty states |
| Switch the Hermes theme | The board repaints to the new palette automatically |
| Bind a stat-card to `usage.status` | It shows today's cost/tokens from `/api/analytics/usage` |

## Dev loop

```bash
npm install        # esbuild + @boardstate/* pinned from npm (build-time only)
npm run build      # → dashboard/dist/index.js, sidecar/server.js, vendored element bundle
npm test           # or: node test/*.mjs  (see below)
```

No local Boardstate monorepo needed — `build.mjs` resolves `@boardstate/*` from npm.

### Tests (all run in CI)

| Test | Proves |
|------|--------|
| `test/sidecar-smoke.mjs` | control plane + nonce gate |
| `test/mcp-liveness.mjs` | an MCP tool call produces a live board update on the single host |
| `test/hermes-data.mjs` | rpc data methods resolve Hermes REST shapes over the **real render path** |
| `test/chat-translate.mjs` | Hermes agent events → SPEC §14 `AgentStreamEvent` |
| `test/theme.mjs` | Hermes→Boardstate token mapping + WCAG luminance split |
| `test/templates.mjs` | every template is a schema-valid, allowlisted workspace |
| `test/plugin_api_check.py` | router shape (WS + MCP proxy + nonce forwarding) |

## License

MIT
