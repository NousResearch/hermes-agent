# Hermes Agent - Future Improvements

---



## 3. Local Browser Control via CDP 🌐

**Status:** Not started (currently Browserbase cloud only)
**Priority:** Medium

Support local Chrome/Chromium via Chrome DevTools Protocol alongside existing Browserbase cloud backend.

**What other agents do:**
- **OpenClaw**: Full CDP-based Chrome control with snapshots, actions, uploads, profiles, file chooser, PDF save, console messages, tab management. Uses local Chrome for persistent login sessions.
- **Cline**: Headless browser with Computer Use (click, type, scroll, screenshot, console logs)

**Our approach:**
- Add a `local` backend option to `browser_tool.py` using Playwright or raw CDP
- Config toggle: `browser.backend: local | browserbase | auto`
- `auto` mode: try local first, fall back to Browserbase
- Local advantages: free, persistent login sessions, no API key needed
- Local disadvantages: no CAPTCHA solving, no stealth mode, requires Chrome installed
- Reuse the same 10-tool interface -- just swap the backend
- Later: Chrome profile management for persistent sessions across restarts

---

## 4. Signal Integration 📡

**Status:** Not started
**Priority:** Low

New platform adapter using signal-cli daemon (JSON-RPC HTTP + SSE). Requires Java runtime and phone number registration.

**Reference:** OpenClaw has Signal support via signal-cli.

---

## 5. Plugin/Extension System 🔌

**Status:** Partially implemented (event hooks exist in `gateway/hooks.py`)
**Priority:** Medium

Full Python plugin interface that goes beyond the current hook system.

**What other agents do:**
- **OpenClaw**: Plugin SDK with tool-send capabilities, lifecycle phase hooks (before-agent-start, after-tool-call, model-override), plugin registry with install/uninstall.
- **Pi**: Extensions are TypeScript modules that can register tools, commands, keyboard shortcuts, custom UI widgets, overlays, status lines, dialogs, compaction hooks, raw terminal input listeners. Extremely comprehensive.
- **OpenCode**: MCP client support (stdio, SSE, StreamableHTTP), OAuth auth for MCP servers. Also has Copilot/Codex plugins.
- **Codex**: Full MCP integration with skill dependencies.
- **Cline**: MCP integration + lifecycle hooks with cancellation support.

**Our approach (phased):**

### Phase 1: Enhanced hooks
- Expand the existing `gateway/hooks.py` to support more events: `before-tool-call`, `after-tool-call`, `before-response`, `context-compress`, `session-end`
- Allow hooks to modify tool results (e.g., filter sensitive output)

### Phase 2: Plugin interface
- `~/.hermes/plugins/<name>/plugin.yaml` + `handler.py`
- Plugins can: register new tools, add CLI commands, subscribe to events, inject system prompt sections
- `hermes plugin list|install|uninstall|create` CLI commands
- Plugin discovery and validation on startup

### Phase 3: MCP support (industry standard)
- MCP client that can connect to external MCP servers (stdio, SSE, HTTP)
- This is the big one -- Codex, Cline, and OpenCode all support MCP
- Allows Hermes to use any MCP-compatible tool server (hundreds exist)
- Config: `mcp_servers` list in config.yaml with connection details
- Each MCP server's tools get registered as a new toolset

---

## 6. MCP (Model Context Protocol) Support 🔗

**Status:** Not started
**Priority:** High -- this is becoming an industry standard

MCP is the protocol that Codex, Cline, and OpenCode all support for connecting to external tool servers. Supporting MCP would instantly give Hermes access to hundreds of community tool servers.

**What other agents do:**
- **Codex**: Full MCP integration with skill dependencies
- **Cline**: `use_mcp_tool` / `access_mcp_resource` / `load_mcp_documentation` tools
- **OpenCode**: MCP client support (stdio, SSE, StreamableHTTP transports), OAuth auth

**Our approach:**
- Implement an MCP client that can connect to external MCP servers
- Config: list of MCP servers in `~/.hermes/config.yaml` with transport type and connection details
- Each MCP server's tools auto-registered as a dynamic toolset
- Start with stdio transport (most common), then add SSE and HTTP
- Could also be part of the Plugin system (#5, Phase 3) since MCP is essentially a plugin protocol

---

## 8. Filesystem Checkpointing / Rollback 🔄

**Status:** Not started
**Priority:** Low-Medium

Automatic filesystem snapshots after each agent loop iteration so the user can roll back destructive changes to their project.

**What other agents do:**
- **Cline**: Workspace checkpoints at each step with Compare/Restore UI
- **OpenCode**: Git-backed workspace snapshots per step, with weekly gc
- **Codex**: Sandboxed execution with commit-per-step, rollback on failure

**Our approach:**
- After each tool call (or batch of tool calls in a single turn) that modifies files, create a lightweight checkpoint of the affected files
- Git-based when the project is a repo: auto-commit to a detached/temporary branch (`hermes/checkpoints/<session>`) after each agent turn, squash or discard on session end
- Non-git fallback: tar snapshots of changed files in `~/.hermes/checkpoints/<session_id>/`
- `hermes rollback` CLI command to restore to a previous checkpoint
- Agent-accessible via a `checkpoint` tool: `list` (show available restore points), `restore` (roll back to a named point), `diff` (show what changed since a checkpoint)
- Configurable: off by default (opt-in via `config.yaml`), since auto-committing can be surprising
- Cleanup: checkpoints expire after session ends (or configurable retention period)
- Integration with the terminal backend: works with local, SSH, and Docker backends (snapshots happen on the execution host)

---

## Implementation Priority Order

### Tier 1: Next Up

1. MCP Support -- #6

### Tier 2: Quality of Life

3. Local Browser Control via CDP -- #3
4. Plugin/Extension System -- #5

### Tier 3: Nice to Have

5. Session Branching / Checkpoints -- #7
6. Filesystem Checkpointing / Rollback -- #8
7. Signal Integration -- #4
