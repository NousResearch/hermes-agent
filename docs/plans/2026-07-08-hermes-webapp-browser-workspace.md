# Hermes Webapp Browser Workspace Implementation Plan

> **For Hermes:** Use the subagent-driven-development skill to implement this plan step by step.

**Goal:** Make Hermes Webapp a first-class browser-native portal for Hermes Agent: each chat owns a persistent workspace with chat, files, previews, and one or more controllable browser tabs that the user and agent can both see.

**Architecture:** Keep Hermes Desktop and Hermes Dashboard intact. Add Hermes Webapp as a named browser portal surface that reuses the existing server/session backbone at first, then grows native webapp panes and browser-workspace APIs behind `hermes webapp`. Share code with Desktop where practical, but do not require Electron for Webapp-only functionality.

**Tech Stack:** FastAPI dashboard server, React/Vite web workspace, existing Hermes session database, existing PTY/event WebSocket channels, future browser worker using isolated local browser contexts, optional extension-capable local Chromium on the host.

---

## Product vocabulary

- **Hermes Dashboard**: the existing browser management/admin surface (`hermes dashboard`). It manages config, profiles, models, keys, sessions, logs, cron, skills, plugins, and can embed the TUI in a Chat tab.
- **Hermes Desktop**: the Electron app (`hermes desktop`). It can use Electron-only APIs: native file dialogs, webviews, local app packaging, OS window controls.
- **Hermes Webapp**: the portable browser-native workspace (`hermes webapp`). It is not a Dashboard rename and not a Desktop replacement. It should become the main multi-device portal: chat, files, previews, responsive browser tabs, and agent-assisted browser/RPA workflows.
- **Hermes Backend / Serve**: the headless JSON-RPC/WebSocket server (`hermes serve`) used by Desktop, Webapp, and future clients.

## Networking boundary

Hermes Webapp must be explicit about where it is reachable:

1. **Localhost mode**: `127.0.0.1` only. Only that computer can use it. This is the safe default.
2. **LAN mode**: bound to a private LAN interface. Devices on that LAN can reach it. Must require auth and clear UI warning.
3. **Tailscale mode**: bound to a Tailscale IP/interface. Devices in that tailnet can reach it. Must require auth and clear UI warning.
4. **Public mode**: out of scope for ad-hoc binds. Public deployment should be through a deliberate hosted/deployment path, not accidental `0.0.0.0` sharing.

Hermes Webapp is not a remote-access product. It can expose a web UI on chosen networks, but it must not pretend to be Chrome Remote Desktop, RustDesk, or a hosted browser-control service.

## Browser workspace model

Each chat gets a **workspace**:

```text
chat_session_id
  ├─ chat transcript / live agent process
  ├─ file panes / project browser state
  ├─ preview panes / responsive viewport presets
  └─ browser workspace
       ├─ browser_context_id
       ├─ tabs[]
       │   ├─ tab_id
       │   ├─ url/title/favicon
       │   ├─ viewport preset / custom size
       │   ├─ last screenshot frame
       │   └─ agent-control state
       ├─ extension profile / user settings
       └─ RPA event log
```

The browser workspace must survive refresh and device switch when the backend is still alive. A phone/laptop browser can reconnect to the same Webapp route and see active/inactive chats plus their browser/file/preview state.

## Browser implementation principles

1. **Visible first**: when the agent clicks or types, the user sees it. Use smooth mouse/cursor animation and visible focus transitions.
2. **Shared ownership**: the user can manually browse; the agent can request control; the user can interrupt or take over.
3. **Session isolation**: browser contexts are per chat/workspace, not one global default profile. Do not attach to a user's personal default Chrome profile with raw CDP.
4. **Extension support**: support loading/using extensions in the browser context, but do not ship user-specific extensions. Users may install LastPass, Dark Reader, etc. The product supplies the extension/profile mechanism, not the extension choices.
5. **Responsive by default**: each browser tab has viewport presets and free resize. Presets are shortcuts, not locks.
6. **Agent visibility**: the agent can list tabs, capture screenshots, inspect page accessibility/DOM metadata, and target a tab by ID.
7. **Permissioned RPA**: browser actions should be visible, logged, and gated by the same safety/approval concepts Hermes already uses for tools.

## Why the current Dashboard chat is insufficient

The Dashboard Chat tab currently embeds `hermes --tui` in xterm over PTY. That is valuable because it preserves TUI behavior, but it is not a native webapp workspace:

- no browser-native multi-pane chat layout;
- no per-chat browser contexts;
- no extension-capable browser surface;
- no first-class responsive viewport controls;
- no agent-visible browser tab registry;
- no browser-RPA control surface.

Hermes Webapp should not fight xterm. It should reuse the existing backend/event/session machinery but build a native web workspace around it.

## PR strategy

- Upstream PRs should stay reviewable and focused:
  - CLI/product naming (`hermes webapp`) and docs;
  - shared Desktop/Webapp attachment fixes;
  - shared pane/preview sizing components;
  - server APIs for workspace/browser sessions;
  - UI slices that can be tested independently.
- Steven's fork can merge broader integrated work directly to its own main branch while still opening clean PRs upstream.
- Desktop-compatible features should be PR'd to Desktop/shared code when they make Desktop better.
- Webapp-only behavior should not pretend to be Desktop. Electron APIs must remain optional.

## Task 1: Add the product surface name

**Objective:** Make `hermes webapp` a real command without breaking `hermes dashboard`.

**Files:**
- Modify: `hermes_cli/subcommands/dashboard.py`
- Modify: `hermes_cli/main.py`
- Test: `tests/hermes_cli/test_serve_command.py`
- Test: `tests/hermes_cli/test_subcommands_batch.py`
- Test: `tests/hermes_cli/test_dashboard_lifecycle_flags.py`

**Verification:**

```bash
python -m pytest tests/hermes_cli/test_serve_command.py tests/hermes_cli/test_subcommands_batch.py::test_dashboard_builder_two_handlers tests/hermes_cli/test_dashboard_lifecycle_flags.py::TestWebappProcessDetection -q -o 'addopts='
```

Expected: all pass.

## Task 2: Add docs and glossary

**Objective:** Give contributors/users stable vocabulary before more code lands.

**Files:**
- Create: `website/docs/user-guide/hermes-webapp.md`
- Modify: `website/sidebars.ts`
- Modify: `website/docs/user-guide/features/web-dashboard.md`
- Modify: `website/docs/user-guide/desktop.md`

**Verification:**

```bash
npm --prefix website run build
```

Expected: docs build succeeds.

## Task 3: Define workspace state contract

**Objective:** Add typed backend/frontend schema for chat workspace state without launching browsers yet.

**Files:**
- Create: `hermes_cli/webapp_workspace.py`
- Add tests under: `tests/hermes_cli/test_webapp_workspace.py`
- Add frontend types under: `web/src/lib/webapp-workspace.ts`

**State:**

```json
{
  "session_id": "...",
  "workspace_id": "...",
  "panes": [],
  "browser": {
    "context_id": "...",
    "tabs": []
  }
}
```

**Verification:** Python unit tests and web typecheck.

## Task 4: Add browser tab registry APIs

**Objective:** Let Webapp list/create/close browser tabs for a chat workspace, even before full RPA exists.

**API sketch:**

```text
GET  /api/webapp/workspaces/{session_id}
POST /api/webapp/workspaces/{session_id}/browser/tabs
POST /api/webapp/workspaces/{session_id}/browser/tabs/{tab_id}/navigate
POST /api/webapp/workspaces/{session_id}/browser/tabs/{tab_id}/close
```

**Verification:** FastAPI TestClient tests against isolated `HERMES_HOME`.

## Task 5: Build the browser-workspace rail in Webapp

**Objective:** Add a native webapp right rail with chat + browser/file/preview tabs. Do not rely on Electron `webview`.

**Files:**
- Add under `web/src/pages` or `web/src/features/webapp-workspace`
- Reuse shared UI patterns where practical.

**Verification:** Vitest/jsdom tests and browser dogfood against local Vite server.

## Task 6: RPA control loop

**Objective:** Add visible agent/browser control primitives: screenshot, click, type, scroll, navigate, active-tab list.

**Important:** This should be a service-gated capability, not a core model tool blindly added to every Hermes turn.

**Potential implementation:** Browser worker process + local WebSocket/SSE stream. Consider BrowserOS ideas, but strip to Hermes-owned control/events and avoid attaching to default Chrome profiles.

## Task 7: Extension support

**Objective:** Support user-installed browser extensions per Webapp browser profile/context.

**Constraints:**
- No bundled LastPass/Dark Reader.
- User-controlled installation/profile import path.
- Document privacy/security boundaries.

## Acceptance criteria

Hermes Webapp is real when a user can:

1. Run `hermes webapp`.
2. Open the same active chat from two devices on an explicitly chosen network scope.
3. See the same chat, file panes, preview panes, and browser tabs.
4. Resize a browser tab to phone/tablet/desktop/ultrawide and freely drag it after presets.
5. Install or load browser extensions in their own Webapp browser profile.
6. Let Hermes visibly click/type/navigate inside a tab with interrupt/takeover controls.
7. Keep all of this local/LAN/Tailscale-scoped unless they intentionally deploy a public service.
