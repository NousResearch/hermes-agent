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

### Phase 3: MCP support (industry standard) ✅ DONE
- ✅ MCP client that connects to external MCP servers (stdio + HTTP/StreamableHTTP)
- ✅ Config: `mcp_servers` in config.yaml with connection details
- ✅ Each MCP server's tools auto-registered as a dynamic toolset
- Future: Resources, Prompts, Progress notifications, `hermes mcp` CLI command

---

## 6. MCP (Model Context Protocol) Support 🔗 ✅ DONE

**Status:** Implemented (PR #301)
**Priority:** Complete

Native MCP client support with stdio and HTTP/StreamableHTTP transports, auto-discovery, reconnection with exponential backoff, env var filtering, and credential stripping. See `docs/mcp.md` for full documentation.

**Still TODO:**
- `hermes mcp` CLI subcommand (list/test/status)
- `hermes tools` UI integration for MCP toolsets
- MCP Resources and Prompts support
- OAuth authentication for remote servers
- Progress notifications for long-running tools

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

1. ~~MCP Support -- #6~~ ✅ Done (PR #301)

### Tier 2: Quality of Life

3. Local Browser Control via CDP -- #3
4. Plugin/Extension System -- #5

### Tier 3: Nice to Have

5. Session Branching / Checkpoints -- #7
6. Filesystem Checkpointing / Rollback -- #8
7. Signal Integration -- #4
