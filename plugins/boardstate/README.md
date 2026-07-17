# boardstate-hermes-plugin

> **The board your agent builds with you — durable, auditable, operator-governed.**

A drop-in plugin for [Hermes](https://github.com/NousResearch/hermes-agent) that adds a
live **Board** to both the web dashboard (`hermes dashboard`) and the **desktop app**:
a layout-as-data workspace the Hermes agent **builds and operates through tools** — and
that you co-own, audit, and govern. Built on [Boardstate](https://github.com/100yenadmin/boardstate)
(`@boardstate/*` on npm, MIT — a 45k-line workspace library).

![One unedited take: the agent composes the board; a template binds live data; a real OfficeCLI grant is approved; the mutation parks; confirming produces an actual .docx](.github/media/hero.gif)

**Upstream:** bundling PRs are open on Hermes — tracker
[hermes-agent#66413](https://github.com/NousResearch/hermes-agent/issues/66413), web
[PR #66381](https://github.com/NousResearch/hermes-agent/pull/66381), desktop
[PR #66425](https://github.com/NousResearch/hermes-agent/pull/66425).

## Features

- **The agent builds the board live.** A networked MCP endpoint (19 `boardstate_*`
  tools) rides the plugin backend against the same single host the browser watches —
  widgets appear as the agent works, no reload.
- **Live Hermes data, zero config.** Usage, sessions, instances, and cron widgets
  self-bind to the dashboard's own REST surface — real numbers, graceful empty states,
  never an error cell. Credentials stay server-side.
- **16 built-in widget kinds** — stat cards, charts (line/bar/area/sparkline/gauge),
  tables, markdown, notes, activity, action buttons/forms, chat, and the live
  data-source widgets — plus **one-click templates**: Agent HQ · Usage & Cost ·
  Sessions Monitor · Office Ops.
- **Sandboxed custom widgets.** Install a widget bundle (a game, a calculator, a
  tracker); it lands **pending** (assets uniformly 404), you approve it, and it mounts
  in an opaque-origin iframe with a no-network CSP — served through a tokenized,
  traversal-jailed asset route. The library's 2048, installed, approved, and played
  inside Hermes:

  ![The library's 2048 bundle: pending → approved → mounted sandboxed → played](.github/media/game-2048.gif)

- **The operational layer.** Connect external MCP tools (an **OfficeCLI** preset ships)
  through operator-governed grants: the agent *requests* tools, you approve per-tool;
  reads pass a manifest-hash gate (a connector that changes its tools re-pends);
  **mutations always park for your confirm** — bounded, never hangs. Every grant is
  visible and revocable in the approvals widget.
- **Native to Hermes, both design systems.** Theme tokens alias to the host palette and
  follow live swaps (![swap](.github/media/theme-swap.gif) shows a whole-palette swap);
  the web skin matches the dashboard's own design language (numerically, against the
  kanban page), the desktop skin the app's macOS language. The library ships 20 locales.

## Security model (designed for review)

- Browser connects only to the dashboard origin; auth is the dashboard's own WS gate.
  The Node sidecar binds loopback-only behind a per-spawn nonce; **one sidecar per
  state dir** (port-file adoption, `chmod 600`) so web + desktop never double-write.
- **Operator verbs** (approve / confirm / deny) are unreachable from the browser WS and
  the agent MCP surface. They flow only through an authenticated plugin route gated by a
  dedicated per-spawn **operator secret (never persisted)** plus a
  `boardstate.operators.json` allowlist (absent ⇒ loopback-only; gated multi-user ⇒
  denied without an allowlist).
- **Connector config never leaves the server**: length-agnostic, longest-first redaction
  (command/url/args/env values + the nonce) on every agent-facing error; anti-rug-pull
  manifest-hash re-pend on every agent-reachable connector call.
- **Custom-widget assets**: approved-only with uniform 404, sandbox CSP preserved
  verbatim through the proxy, traversal segments rejected before any upstream request.
- Hardened through three independent adversarial review passes plus an external review
  round; the holes they found ship with **revert-checked** regression tests.

## Architecture

```
browser  <boardstate-view>                     desktop app  (single-file ESM plugin)
   │  ws   /api/plugins/boardstate/ws  ←──────────┘  (same backend, desktop bridge)
   │  http /api/plugins/boardstate/mcp   ◄─ Hermes agent (StreamableHTTP, 19 tools)
   │  http /api/plugins/boardstate/operator  ◄─ approvals UI (session + allowlist)
   │  http /boardstate-widget-assets/<token>/…  ◄─ custom-widget iframes (capability token)
   ▼
plugin_api.py — WS bridge · MCP proxy · operator gate · tokenized asset route · sidecar lifecycle
   ▼            (per-spawn nonce + operator secret, injected via env, never persisted)
sidecar/server.js — ONE host, one writer:  control plane · live Hermes data RPCs ·
   connector broker (boardstate.connectors.json) · pending-action engine · /widgets (CSP)
   ▼
$HERMES_HOME/boardstate-state/dashboard/workspace.json   ← the board IS this document
```

The board is a **validated document** — one writer, every mutation through gated verbs —
which is what makes undo, history, export/import, templates, and audit possible.

## Install (web dashboard)

Requires Node ≥ 20 on the machine running `hermes dashboard`.

```bash
mkdir -p ~/.hermes/plugins/boardstate
cp -r dashboard ~/.hermes/plugins/boardstate/dashboard
# enable it in ~/.hermes/config.yaml:   plugins: { enabled: [boardstate] }
hermes dashboard          # the Board tab appears after Skills
```

### Install (desktop app)

The desktop frontend is a single file on the desktop plugin surface (backend above must
also be installed):

```bash
mkdir -p ~/.hermes/desktop-plugins/boardstate
cp dashboard/desktop/plugin.js ~/.hermes/desktop-plugins/boardstate/plugin.js
```

The app hot-loads it — a **Board** entry appears in the sidebar (real-Electron proven):

![The Board inside the real Hermes desktop app — live template data, macOS design language](.github/media/desktop-board.png)

### Let the agent build it

Register the plugin's MCP endpoint so Hermes builds and operates the board via tools —
in `~/.hermes/config.yaml` (or a profile):

```yaml
mcp_servers:
  boardstate:
    url: http://127.0.0.1:9119/api/plugins/boardstate/mcp   # your dashboard port
    headers:
      X-Hermes-Session-Token: <dashboard session token>
```

### Connect external tools (the operational layer)

```bash
# state dir = $HERMES_HOME/boardstate-state
cat > ~/.hermes/boardstate-state/boardstate.connectors.json <<'EOF'
{ "connectors": [ { "name": "officecli", "transport": "stdio", "command": "officecli", "args": ["mcp"] } ] }
EOF
```

Restart; the approvals widget shows the connector's **requested** grant. Approve the
tools you want — nothing runs until you do. See [docs/connectors/officecli.md](docs/connectors/officecli.md).

### Config

| Setting | Default | Notes |
|---------|---------|-------|
| Board state dir | `$HERMES_HOME/boardstate-state` | `BOARDSTATE_HERMES_STATE_DIR` overrides |
| Node binary | `node` on `PATH` | `HERMES_NODE_BIN` overrides |
| Connectors | none | operator-authored `boardstate.connectors.json` in the state dir — never the board doc |
| Operator allowlist | none (loopback-only) | `boardstate.operators.json`; **required** in gated multi-user mode |
| Mutation confirm timeout | 300 000 ms | `BOARDSTATE_MUTATION_TIMEOUT_MS` (must be > 0) |

## Do / Observe

| Do | Observe |
|----|---------|
| Open the **Board** tab (web or desktop) | renders in the active theme; follows a palette swap live |
| Ask the agent to add a widget | it appears live, no reload |
| Click **Agent HQ** | live usage / sessions / instances / cron — zero error cells |
| Author `boardstate.connectors.json`, restart | approvals widget shows the **requested** grant |
| Approve, then run the Office Ops action | it **parks**; confirm; a real `.docx` lands |
| Install + approve a widget bundle (e.g. `twenty48`) | pending assets 404; approved → mounts sandboxed, playable |
| Kill Node / uninstall | dashboard unaffected; the tab degrades to a clear message |

## Dev loop

```bash
npm ci && npm run build   # web tab + desktop plugin + sidecar + vendored bundles (npm-pinned @boardstate/*)
```

### Tests (all run in CI — 23 node suites + 3 python)

| Highlights | Proves |
|------|--------|
| `operational-e2e.mjs` | the full loop headless: request → approve → invoke → **park** → confirm → result |
| `custom-widget.mjs` | pending assets 404; approved serves with the CSP jail |
| `rugpull-repend.mjs` | manifest drift re-pends the grant, never executes (revert-checked) |
| `secret-redaction.mjs` | connector config — incl. env values — never reaches the agent surface (revert-checked) |
| `operator-secret.mjs` | port-file knowledge cannot drive the operator plane |
| `invoke-timeout.mjs` | an unconfirmed mutation settles as parked, never hangs |
| `asset_proxy.py` | tokenized asset route at runtime: traversal-jailed, CSP forwarded verbatim |
| `operator_wire.py` | the operator gate's wire contract, incl. gated-mode 403s |
| + sidecar smoke, MCP liveness, data wire-contract, chat translator, theme (48), templates (27), skins, desktop bundle contract (14) | every seam has a test |

Full history in [CHANGELOG.md](CHANGELOG.md) (v1.0.0 → v1.4.0).

## License

MIT
