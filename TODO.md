# Hermes Agent - Future Improvements

---

## What We Already Have (for reference)

**44+ tools** across 13 toolsets: web (search, extract), terminal + process management, file ops (read, write, patch, search), vision, MoA reasoning, image gen, browser (10 tools via Browserbase), skills (41 bundled + agent-managed via `skill_manage`), **todo (task planning)**, cronjobs, RL training (10 tools via Tinker-Atropos), TTS, cross-channel messaging.

**Skills System**: All skills live in `~/.hermes/skills/` (single source of truth). Bundled skills seeded on install via manifest-based sync (`tools/skills_sync.py`). `hermes update` adds only genuinely new bundled skills without overwriting edits or re-adding deletions. Agent can create, patch, edit, delete any skill via `skill_manage` tool. Hub search/install/inspect/audit/uninstall/publish/snapshot across 4 registries (GitHub, ClawHub, Claude Code marketplaces, LobeHub). Security scanner with trust-aware policy. CLI (`hermes skills ...`) and `/skills` slash command. agentskills.io spec compliant.

**Persistent Memory**: MEMORY.md (agent notes, 2200 char) + USER.md (user profile, 1375 char) in `~/.hermes/memories/`. Injected into system prompt as frozen snapshot. Agent manages via `memory` tool (add/replace/remove). Session search via `session_search` tool over SQLite store.

**4 platform adapters**: Telegram, Discord, WhatsApp, Slack -- all with typing indicators, image/voice auto-analysis, dangerous command approval, interrupt support, background process watchers.

**Other**: Context compression, context files (SOUL.md, AGENTS.md), session JSONL transcripts, batch runner with toolset distributions, 13 personalities, DM pairing auth, PTY mode, model metadata caching.

---

## The Knowledge System (how Memory, Skills, Sessions, and Subagents interconnect)

> **Current status:** Procedural memory (Skills ✅), Declarative memory (MEMORY.md ✅), Identity memory (USER.md ✅), Episodic memory (Session search ✅) are all implemented. Error memory (Learnings) and Subagents are not yet started.

These four systems form a continuum of agent intelligence. They should be thought of together:

**Types of agent knowledge:**
- **Procedural memory (Skills)** -- reusable approaches for specific task types. "How to deploy a Docker container." "How to fine-tune with Axolotl." Created when the agent works through something difficult and succeeds.
- **Declarative memory (MEMORY.md)** -- facts about the environment, projects, tools, conventions. "This repo uses Poetry, not pip." "The API key is stored in ~/.config/keys."
- **Identity memory (USER.md / memory_summary.md)** -- who the user is, how they like to work, their preferences, communication style. Persists across all sessions.
- **Error memory (Learnings)** -- what went wrong and the proven fix. "pip install fails on this system because of X; use conda instead."
- **Episodic memory (Session summaries)** -- what happened in past sessions. Searchable for when the agent needs to recall prior conversations.

**The feedback loop:** After complex tasks, especially ones involving difficulty or iteration, the agent should:
1. Ask the user for feedback: "How was that? Did it work out?"
2. If successful, offer to save: "Would you like me to save that as a skill for next time?"
3. Update general memory with any durable insights (user preferences, environment facts, lessons learned)

**Storage evolution:** Sessions use SQLite (`~/.hermes/state.db`). Memory uses flat files (`~/.hermes/memories/`). Skills use flat files (`~/.hermes/skills/`). Learnings will use SQLite when implemented.

---

## 1. Subagent Architecture (Context Isolation) 🎯

**Status:** Implemented ✅
**Priority:** High -- this is foundational for scaling to complex tasks

The main agent becomes an orchestrator that delegates context-heavy tasks to subagents with isolated context. Each subagent returns a summary, keeping the orchestrator's context clean.

**Core design:**
- `delegate_task(goal, context, toolsets=[])` tool -- spawns a fresh AIAgent with its own conversation, limited toolset, and task-specific system prompt
- Parent passes a goal string + optional context blob; child returns a structured summary
- Configurable depth limit (e.g., max 2 levels) to prevent runaway recursion
- Subagent inherits the same terminal/browser session (task_id) but gets a fresh message history

**What other agents do:**
- **OpenClaw**: `sessions_spawn` + `subagents` tool with list/kill/steer actions, depth limits, rate limiting. Cross-session agent-to-agent coordination via `sessions_send`.
- **Codex**: `spawn_agent` / `send_input` / `close_agent` / `wait_for_agent` with configurable timeouts. Thread manager for concurrent agents. Also uses subagents for memory consolidation (Phase 2 spawns a dedicated consolidation agent).
- **Cline**: Up to 5 parallel subagents per invocation. Subagents get restricted tool access (read, list, search, bash, skill, attempt). Progress tracking with stats (tool calls, tokens, cost, context usage).
- **OpenCode**: `TaskTool` creates subagent sessions with permission inheritance. Resumable tasks via `task_id`. Parent-child session relationships.

**Our approach:**
- Start with a single `delegate_task` tool (like Cline's model -- simple, bounded)
- Subagent gets: goal, context excerpt, restricted toolset, fresh conversation
- Returns: summary string (success/failure + key findings + any file paths created)
- Track active subagents so parent can reference them; limit concurrency to 3
- Primary use cases: parallelizing distinct work (research two topics, work on two separate code changes), handling context-heavy tasks that would bloat the parent's context
- Later: add `send_input` for interactive subagent steering (Codex-style)
- Later: cross-session coordination for gateway (OpenClaw-style `sessions_send`)

---

## 2. Interactive Clarifying Questions ❓

**Status:** Implemented ✅
**Priority:** Medium-High -- enables the knowledge system feedback loop

Allow the agent to present structured choices to the user when it needs clarification or feedback. Rich terminal UI in CLI mode, graceful fallback on messaging platforms.

**What other agents do:**
- **Codex**: `request_user_input` tool for open-ended questions
- **Cline**: `ask_followup_question` tool with structured options
- **OpenCode**: `question` tool for asking the user

**Our approach:**
- `clarify` tool with parameters: `question` (string), `choices` (list of up to 6 strings), `allow_freetext` (bool)
- CLI mode: Rich-powered selection UI (arrow keys + number shortcuts)
- Gateway/messaging mode: numbered list with "reply with number or type your answer"
- Returns the user's selection as a string

**Use cases (beyond simple clarification):**
- Before starting expensive operations: "Which approach do you prefer?"
- **Post-task feedback**: "How did that work out?" with choices like [Worked perfectly / Mostly good / Had issues / Didn't work]
- **Skill creation offer**: "Want me to save that approach as a skill?" with [Yes / Yes, but let me review it first / No]
- **Memory update prompt**: "I noticed you prefer X. Should I remember that for future sessions?" with [Yes / No / It depends]

This tool is lightweight on its own but becomes critical for the proactive feedback loop in the knowledge system (skills, memory, learnings).

**File:** `tools/clarify_tool.py` -- presentation layer differs per platform, core logic is simple

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

## 7. Session Branching / Checkpoints 🌿

**Status:** Not started
**Priority:** Low-Medium

Save and restore conversation state at any point. Branch off to explore alternatives without losing progress.

**What other agents do:**
- **Pi**: Full branching -- create branches from any point in conversation. Branch summary entries. Parent session tracking for tree-like session structures.
- **Cline**: Checkpoints -- workspace snapshots at each step with Compare/Restore UI
- **OpenCode**: Git-backed workspace snapshots per step, with weekly gc

**Our approach:**
- `checkpoint` tool: saves current message history + working directory state as a named snapshot
- `restore` tool: rolls back to a named checkpoint
- Stored in `~/.hermes/checkpoints/<session_id>/<name>.json`
- For file changes: git stash or tar snapshot of working directory
- Useful for: "let me try approach A, and if it doesn't work, roll back and try B"
- Later: full branching with tree visualization

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

## 9. Programmatic Tool Calling (Code-Mediated Tool Use) 🧬

**Status:** Implemented (MVP) ✅
**Priority:** High -- potentially the single biggest efficiency win for agent loops

Instead of the LLM making one tool call, reading the result, deciding what to do next, making another tool call (N round trips), the LLM writes a Python script that calls multiple tools, processes results, branches on conditions, and returns a final summary -- all in one turn.

**What Anthropic just shipped (Feb 2026):**

Anthropic's new `web_search_20260209` and `web_fetch_20260209` tools use "dynamic filtering" -- Claude writes and executes Python code that calls the search/fetch tools, filters the HTML, cross-references results, retries with different queries, and returns only what's relevant. Results: **+11% accuracy, -24% input tokens** on average across BrowseComp and DeepsearchQA. Quora/Poe found it "achieved the highest accuracy on our internal evals" and described it as behaving "like an actual researcher, writing Python to parse, filter, and cross-reference results rather than reasoning over raw HTML in context."

Source: [claude.com/blog/improved-web-search-with-dynamic-filtering](https://claude.com/blog/improved-web-search-with-dynamic-filtering)

**Why this matters for agent loops:**

The standard agent loop is:
```
LLM call -> tool call -> result -> LLM call -> tool call -> result -> LLM call -> ...
```
Every round trip costs: a full LLM inference (prompt + generation), network latency, and the growing context window carrying all previous tool results. For a 10-step task, that's 10+ LLM calls with increasingly large contexts.

With programmatic tool calling:
```
LLM call -> writes Python script -> script calls tools N times, processes results,
branches on conditions, retries on failure -> returns summary -> LLM call
```
One LLM call replaces many. The intermediate tool results never enter the context window -- they're processed in the code sandbox and only the final summary comes back. The LLM pre-plans its decision tree in code rather than making decisions one-at-a-time in the conversation.

**Which of our tools benefit most:**

| Tool | Current pattern (N round trips) | With programmatic calling (1 round trip) |
|------|--------------------------------|----------------------------------------|
| **web_search + web_extract** | Search -> read results -> pick URLs -> extract each -> read each -> synthesize | Script: search, fetch top 5, extract relevant sections, cross-reference, return summary |
| **browser (10 tools)** | navigate -> snapshot -> click -> snapshot -> type -> snapshot -> ... | Script: navigate, loop through elements, extract data, handle pagination, return structured result |
| **file ops (read, search, patch)** | Search for pattern -> read matching files -> decide which to patch -> patch each | Script: search, read all matches, filter by criteria, apply patches, verify, return diff summary |
| **session_search** | Search -> read results -> search again with refined query -> ... | Script: search with multiple queries, deduplicate, rank by relevance, return top N |
| **terminal** | Run command -> check output -> run follow-up -> check again -> ... | Script: run command, parse output, branch on exit code, run follow-ups, return final state |

### The hard problem: where does the code run?

Our tools don't all live in one place. The terminal backend can be local, Docker, Singularity, SSH, or Modal. Browser runs on Browserbase cloud. Web search/extract are Firecrawl API calls. File ops go through whatever terminal backend is active. Vision, image gen, TTS are all remote APIs.

If we just "run Python in the terminal," we hit a wall:
- **Docker/Modal/SSH backends**: The remote environment doesn't have our Python tool code, our API keys, our `handle_function_call` dispatcher, or any of the hermes-agent packages. It's a bare sandbox.
- **Local backend**: Could import our code directly, but that couples execution to local-only and creates a security mess (LLM-generated code running in the agent process).
- **API-based tools** (web, browser, vision): These need API keys and specific client libraries that aren't in the terminal backend.

**The code sandbox is NOT the terminal backend.** This is the key insight. The sandbox runs on the **agent host machine** (where `run_agent.py` lives), separate from both the LLM and the terminal backend. It calls tools through the same `handle_function_call` dispatcher that the normal agent loop uses. No inbound network connections needed -- everything is local IPC on the agent host.

### Architecture: Local subprocess with Unix domain socket RPC

```
Agent Host (where run_agent.py runs)

  ┌──────────────┐         ┌──────────────────────┐
  │ Code Sandbox │  UDS    │ Agent Process         │
  │ (child proc) │ ◄─────► │ (parent)              │
  │              │ (local) │                       │
  │ print()──────┼─stdout──┼─► captured as output  │
  │ errors───────┼─stderr──┼─► captured for debug  │
  │              │         │                       │
  │ hermes_tools │         │ handle_function_call  │
  │ .web_search()├─socket──┼─► web_search ─────────┼──► Firecrawl API
  │ .terminal()  ├─socket──┼─► terminal ───────────┼──► Docker/SSH/Modal
  │ .browser_*() ├─socket──┼─► browser ────────────┼──► Browserbase
  │ .read_file() ├─socket──┼─► read_file ──────────┼──► terminal backend
  └──────────────┘         └──────────────────────┘
```

The sandbox is a **child process** on the same machine. RPC goes over a **Unix domain socket** (not stdin/stdout/stderr -- those stay free for their natural purposes). The parent dispatches each tool call through the existing `handle_function_call` -- the exact same codepath the normal agent loop uses. Works with every terminal backend because the sandbox doesn't touch the terminal backend directly.

### RPC transport: Unix domain socket

Why not stdin/stdout/stderr for RPC?
- **stdout** is the script's natural output channel (`print()`). Multiplexing RPC and output on the same stream requires fragile marker parsing. Keep it clean: stdout = final output for the LLM.
- **stderr** is for Python errors, tracebacks, warnings, and `logging` output. Multiplexing RPC here means any stray `logging.warning()` or exception traceback corrupts the RPC stream.
- **Extra file descriptors (fd 3/4)** work on Linux/macOS but are clunky with subprocess.Popen.

A **Unix domain socket** gives a clean dedicated RPC channel:
1. Parent creates a temp UDS: `/tmp/hermes_rpc_<uuid>.sock`
2. Parent starts listening (single-client, since there's one sandbox)
3. Parent spawns child with `HERMES_RPC_SOCKET=/tmp/hermes_rpc_<uuid>.sock` in env
4. Child's `hermes_tools` module connects to the socket on first tool call
5. Protocol: newline-delimited JSON. Child writes `{"tool": "web_search", "args": {...}}\n`, reads `{"result": ...}\n` back
6. Parent reads each request, calls `handle_function_call`, writes the response
7. After child exits, parent cleans up the socket file

**Channels stay clean:**
- `stdout` → captured by parent as the tool's return value to the LLM
- `stderr` → captured by parent for error reporting (included in response on failure)
- UDS → dedicated tool call RPC (invisible to the script's normal I/O)

Works on Linux and macOS (our target platforms). Windows fallback: named pipes or the marker-on-stderr approach if we ever need it.

### The auto-generated `hermes_tools` module

The parent writes this into a temp directory before spawning the child. Each function is a thin RPC stub:

```python
# Auto-generated: /tmp/hermes_sandbox_<uuid>/hermes_tools.py
import json, os, socket

_sock = None

def _connect():
    global _sock
    if _sock is None:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.connect(os.environ["HERMES_RPC_SOCKET"])
        _sock.settimeout(300)  # 5 min max per tool call
    return _sock

def _call(tool_name, args):
    """RPC: send tool call to parent, get result back."""
    conn = _connect()
    request = json.dumps({"tool": tool_name, "args": args}) + "\n"
    conn.sendall(request.encode())
    # Read response (newline-delimited)
    chunks = []
    while True:
        data = conn.recv(65536)
        if not data:
            raise RuntimeError("Agent process disconnected")
        chunks.append(data.decode())
        if chunks[-1].endswith("\n"):
            break
    raw = "".join(chunks).strip()
    # Tool responses are JSON strings; parse them into dicts
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

# --- Tool functions (one per enabled tool) ---

def web_search(query):
    """Search the web. Returns dict with 'results' list."""
    return _call("web_search", {"query": query})

def web_extract(urls):
    """Extract content from URLs. Returns markdown text."""
    return _call("web_extract", {"urls": urls})

def read_file(path, offset=1, limit=500):
    """Read a file. Returns dict with content and metadata."""
    return _call("read_file", {"path": path, "offset": offset, "limit": limit})

def terminal(command, timeout=None):
    """Run a shell command. Returns dict with stdout, exit_code."""
    return _call("terminal", {"command": command, "timeout": timeout})

def search(pattern, target="content", path=".", file_glob=None, limit=50):
    """Search file contents or find files."""
    return _call("search", {"pattern": pattern, "target": target,
                             "path": path, "file_glob": file_glob, "limit": limit})

# ... generated for each enabled tool in the session
```

This module is generated dynamically because the available tools vary per session and per toolset configuration. The generator reads the session's enabled tools and emits a function for each one that's on the sandbox allow-list.

### What Python libraries are available in the sandbox

The sandbox runs the same Python interpreter as the agent. Available imports:

**Python standard library (always available):**
`json`, `re`, `math`, `csv`, `datetime`, `collections`, `itertools`, `textwrap`, `difflib`, `html`, `urllib.parse`, `pathlib`, `hashlib`, `base64`, `string`, `functools`, `operator`, `statistics`, `io`, `os.path`

**Not restricted but discouraged via tool description:**
`subprocess`, `socket`, `requests`, `urllib.request`, `os.system` -- the tool description says "use hermes_tools for all I/O." We don't hard-block these because the user already trusts the agent with `terminal()`, which is unrestricted shell access. Soft-guiding the LLM via the description is sufficient. If it occasionally uses `import os; os.listdir()` instead of `hermes_tools.search()`, no real harm done.

**The tool description tells the LLM:**
```
Available imports:
- from hermes_tools import web_search, web_extract, read_file, terminal, ...
- Python standard library: json, re, math, csv, datetime, collections, etc.

Use hermes_tools for all I/O (web, files, commands, browser).
Use stdlib for processing between tool calls (parsing, filtering, formatting).
Print your final result to stdout.
```

### Platform support

**Linux / macOS**: Fully supported. Unix domain sockets work natively.

**Windows**: Not supported. `AF_UNIX` nominally exists on Windows 10 17063+ but is unreliable in practice, and Hermes-Agent's primary target is Linux/macOS (bash-based install, systemd gateway, etc.). The `execute_code` tool is **disabled at startup on Windows**:

```python
import sys
SANDBOX_AVAILABLE = sys.platform != "win32"

def check_sandbox_requirements():
    return SANDBOX_AVAILABLE
```

If the LLM tries to use `execute_code` on Windows, it gets: `{"error": "execute_code is not available on Windows. Use normal tool calls instead."}`. The tool is excluded from the tool schema entirely on Windows so the LLM never sees it.

### Which tools to expose in the sandbox (full audit)

The purpose of the sandbox is **reading, filtering, and processing data across multiple tool calls in code**, collapsing what would be many LLM round trips into one. Every tool needs to justify its inclusion against that purpose. The parent only generates RPC stubs for tools that pass this filter AND are enabled in the session.

**Every tool, one by one:**

| Tool | In sandbox? | Reasoning |
|------|------------|-----------|
| `web_search` | **YES** | Core use case. Multi-query, cross-reference, filter. |
| `web_extract` | **YES** | Core use case. Fetch N pages, parse, keep only relevant sections. |
| `read_file` | **YES** | Core use case. Bulk read + filter. Note: reads files on the terminal backend (Docker/SSH/Modal), not the agent host -- this is correct and intentional. |
| `search` | **YES** | Core use case. Find files/content, then process matching results. |
| `terminal` | **YES (restricted)** | Command chains with branching on exit codes. Foreground only -- `background`, `check_interval`, and `pty` parameters are stripped/blocked. |
| `write_file` | **YES (with caution)** | Scripts need to write computed outputs (generated configs, processed data). Partial-write risk if script fails midway, but same risk as normal tool calls. |
| `patch` | **YES (with caution)** | Bulk search-and-replace across files. Powerful but risky if the script's patch logic has bugs. The upside: script can read → patch → verify in a loop, which is actually safer than blind patching. |
| `browser_navigate` | **YES** | Browser automation loops are one of the biggest wins. |
| `browser_snapshot` | **YES** | Needed for reading page state in browser loops. Parent passes `user_task` from the session context. |
| `browser_click` | **YES** | Core browser automation. |
| `browser_type` | **YES** | Core browser automation. |
| `browser_scroll` | **YES** | Core browser automation. |
| `browser_back` | **YES** | Navigation within browser loops. |
| `browser_press` | **YES** | Keyboard interaction in browser loops. |
| `browser_close` | **NO** | Ends the entire browser session. If the script errors out after closing, the agent has no browser to recover with. Too destructive for unsupervised code. |
| `browser_get_images` | **NO** | Niche. Usually paired with vision analysis, which is excluded. |
| `browser_vision` | **NO** | This calls the vision LLM API -- expensive per call and requires LLM reasoning on the result. Defeats the purpose of avoiding LLM round trips. |
| `vision_analyze` | **NO** | Expensive API call per invocation. The LLM needs to SEE and reason about images directly, not filter them in code. One-shot nature. |
| `mixture_of_agents` | **NO** | This IS multiple LLM calls. Defeats the entire purpose. |
| `image_generate` | **NO** | Media generation. One-shot, no filtering logic benefits. |
| `text_to_speech` | **NO** | Media generation. One-shot. |
| `process` | **NO** | Background process management from an ephemeral script is incoherent. The script exits, but the process lives on -- who monitors it? |
| `skills_list` | **NO** | Skills are knowledge for the LLM to read and reason about. Loading a skill inside code that can't reason about it is pointless. |
| `skill_view` | **NO** | Same as above. |
| `schedule_cronjob` | **NO** | Side effect. Should not happen silently inside a script. |
| `list_cronjobs` | **NO** | Read-only but not useful in a code-mediation context. |
| `remove_cronjob` | **NO** | Side effect. |
| `send_message` | **NO** | Cross-platform side effect. Must not happen unsupervised. |
| `todo_write` | **NO** | Agent-level conversational state. Meaningless from code. |
| `todo_read` | **NO** | Same. |
| `clarify` | **NO** | Requires interactive user input. Can't block in a script. |
| `execute_code` | **NO** | No recursive sandboxing. |
| All RL tools | **NO** | Separate domain with its own execution model. |

**Summary: 14 tools in, 28+ tools out.** The sandbox exposes: `web_search`, `web_extract`, `read_file`, `write_file`, `search`, `patch`, `terminal` (restricted), `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`, `browser_press`.

The allow-list is a constant in `code_execution_tool.py`, not derived from the session's enabled toolsets. Even if the session has `vision_analyze` enabled, it won't appear in the sandbox. The intersection of the allow-list and the session's enabled tools determines what's generated.

### Error handling

| Scenario | What happens |
|----------|-------------|
| **Syntax error in script** | Child exits immediately, traceback on stderr. Parent returns stderr as the tool response so the LLM sees the error and can retry. |
| **Runtime exception** | Same -- traceback on stderr, parent returns it. |
| **Tool call fails** | RPC returns the error JSON (same as normal tool errors). Script decides: retry, skip, or raise. |
| **Unknown tool called** | RPC returns `{"error": "Unknown tool: foo. Available: web_search, read_file, ..."}`. |
| **Script hangs / infinite loop** | Killed by timeout (SIGTERM, then SIGKILL after 5s). Parent returns timeout error. |
| **Parent crashes mid-execution** | Child's socket connect/read fails, gets a RuntimeError, exits. |
| **Child crashes mid-execution** | Parent detects child exit via `process.poll()`. Collects partial stdout + stderr. |
| **Slow tool call (e.g., terminal make)** | Overall timeout covers total execution. One slow call is fine if total is under limit. |
| **Tool response too large in memory** | `web_extract` can return 500KB per page. If the script fetches 10 pages, that's 5MB in the child's memory. Not a problem on modern machines, and the whole point is the script FILTERS this down before printing. |
| **User interrupt (new message on gateway)** | Parent catches the interrupt event (same as existing `_interrupt_event` in terminal_tool), sends SIGTERM to child, returns `{"status": "interrupted"}`. |
| **Script tries to call excluded tool** | RPC returns `{"error": "Tool 'vision_analyze' is not available in execute_code. Use it as a normal tool call instead."}` |
| **Script calls terminal with background=True** | RPC strips the parameter and runs foreground, or returns an error. Background processes from ephemeral scripts are not supported. |

The tool response includes structured metadata:
```json
{
  "status": "success | error | timeout | interrupted",
  "output": "...",
  "errors": "...",
  "tool_calls_made": 7,
  "duration_seconds": 12.3
}
```

### Resource limits

- **Timeout**: 120 seconds default (configurable via `config.yaml`). Parent sends SIGTERM, waits 5s, SIGKILL.
- **Tool call limit**: max 50 RPC tool calls per execution. After 50, further calls return an error. Prevents infinite tool-call loops.
- **Output size**: stdout capped at 50KB. Truncated with `[output truncated at 50KB]`. Prevents the script from flooding the LLM's context with a huge result (which would defeat the purpose).
- **Stderr capture**: capped at 10KB for error reporting.
- **No recursive sandboxing**: `execute_code` is not in the sandbox's tool list.
- **Interrupt support**: respects the same `_interrupt_event` mechanism as terminal_tool. If the user sends a new message while the sandbox is running, the child is killed and the agent can process the interrupt.

### Tool call logging and observability

Each tool call made inside the sandbox is **logged to the session transcript** for debugging, but **NOT added to the LLM conversation history** (that's the whole point -- keeping intermediate results out of context).

The parent logs each RPC-dispatched call:
```jsonl
{"type": "sandbox_tool_call", "tool": "web_search", "args": {"query": "..."}, "duration": 1.2}
{"type": "sandbox_tool_call", "tool": "web_extract", "args": {"urls": [...]}, "duration": 3.4}
```

These appear in the JSONL transcript and in verbose logging, but the LLM only sees the final `execute_code` response containing the script's stdout.

For the gateway (messaging platforms): show one typing indicator + notification for the entire `execute_code` duration. Internal tool calls are silent. Later enhancement: progress updates like "execute_code (3/7 tool calls)".

### Stateful tools work correctly

Tools like `terminal` (working directory, env vars) and `browser_*` (page state, cookies) maintain state per `task_id`. The parent passes the session's `task_id` to every RPC-dispatched `handle_function_call`. So if the script runs:
```python
terminal("cd /tmp")
terminal("pwd")  # returns /tmp -- state persists between calls
```
This works because both calls go through the same terminal environment, same as normal tool calls.

### Each `execute_code` invocation is stateless

The sandbox subprocess is fresh each time. No Python state carries over between `execute_code` calls. If the agent needs state across multiple `execute_code` calls, it should:
- Output the state as part of the result, then pass it back in the next script as a variable
- Or use the tools themselves for persistence (write to a file, then read it in the next script)

The underlying *tools* are stateful (same terminal session, same browser session), but the *Python sandbox* is not.

### When should the LLM use `execute_code` vs normal tool calls?

This goes in the tool description:

**Use `execute_code` when:**
- You need 3+ tool calls with processing logic between them
- You need to filter/reduce large tool outputs before they enter your context
- You need conditional branching (if X then do Y, else do Z)
- You need to loop (fetch N pages, process N files, retry on failure)

**Use normal tool calls when:**
- Single tool call with no processing needed
- You need to see the full result and apply complex reasoning (the LLM is better at reasoning than the code it writes)
- The task requires human interaction (clarify tool)

### Open questions for implementation

1. **Should the parent block its main thread while the sandbox runs?** Currently `handle_function_call` is synchronous, so yes -- same as any other tool call. For long sandbox runs (up to 120s), the gateway's typing indicator stays active. The agent can't process new messages during this time, but it can't during any long tool call either. Interrupt support (above) handles the "user sends a new message" case.

2. **Should `browser_snapshot` pass `user_task`?** `handle_browser_function_call` accepts `user_task` for task-aware content extraction. The parent should pass the user's original query from the session context when dispatching sandbox browser calls.

3. **Terminal parameter restrictions**: The sandbox version of `terminal` should strip/ignore: `background=True` (no background processes from ephemeral scripts), `check_interval` (gateway-only feature for background watchers), `pty=True` (interactive PTY makes no sense in a script). Only `command`, `timeout`, and `workdir` are passed through.

### Future enhancements (not MVP)

- **Concurrent tool calls via threading**: The script could use `ThreadPoolExecutor` to fetch 5 URLs in parallel. Requires the UDS client to be thread-safe (add a `threading.Lock` around socket send/receive). Significant speedup for I/O-bound workflows like multi-page web extraction.
- **Streaming progress to gateway**: Instead of one notification for the entire run, send periodic progress updates ("execute_code: 3/7 tool calls, 12s elapsed").
- **Persistent sandbox sessions**: Keep the subprocess alive between `execute_code` calls so Python variables carry over. Adds complexity but enables iterative multi-step workflows where the agent refines its script.
- **RL/batch integration**: Atropos RL environments use ToolContext instead of handle_function_call. Would need an adapter so the RPC bridge dispatches through the right mechanism per execution context.
- **Windows support**: If there's demand, fall back to TCP localhost (127.0.0.1:random_port) instead of UDS. Same protocol, different transport. Security concern: localhost port is accessible to other processes on the machine. Could mitigate with a random auth token in the RPC handshake.

### Relationship to other items
- **Subagent Architecture (#1)**: A code sandbox that calls tools IS a lightweight subagent without its own LLM inference. Handles "mechanical multi-step" cases at near-zero LLM cost. Full subagents still needed for tasks requiring LLM reasoning at each step.
- **Browser automation (#3)**: Biggest win. Browser workflows are 10+ round trips today. A script that navigates, clicks, extracts, paginates in a loop collapses that to 1 LLM turn.
- **Web search**: Directly matches Anthropic's dynamic filtering results.
- **File ops**: Bulk read-search-patch workflows become one call.

**Files:** `tools/code_execution_tool.py` (subprocess management, UDS server, RPC dispatch, hermes_tools generator), tool schema in `model_tools.py`

---

## Implementation Priority Order

### Done ✅

- **Memory System.** MEMORY.md + USER.md, bounded, system prompt injection, `memory` tool.
- **Agent-Managed Skills.** `skill_manage` tool (create/patch/edit/delete/write_file/remove_file), unified `~/.hermes/skills/` dir, manifest-based sync.
- **SQLite State Store & Session Search.** `~/.hermes/state.db` with sessions, messages, FTS5 search, `session_search` tool.
- **Interactive Clarifying Questions.** `clarify` tool with arrow-key selection UI in CLI, configurable timeout, CLI-only.
- **Programmatic Tool Calling.** `execute_code` tool -- sandbox child process with UDS RPC bridge to 7 tools (`web_search`, `web_extract`, `read_file`, `write_file`, `search`, `patch`, `terminal`). Configurable timeout and tool call limits via `config.yaml`.
- **Subagent Architecture.** `delegate_task` tool -- spawn child AIAgents with isolated context and terminal sessions. Single-task and batch (up to 3 parallel) modes via ThreadPoolExecutor. Depth limit of 2, blocked tool enforcement, interrupt propagation.

### Tier 1: Next Up

1. MCP Support -- #6

### Tier 2: Quality of Life

3. Local Browser Control via CDP -- #3
4. Plugin/Extension System -- #5

### Tier 3: Nice to Have

5. Session Branching / Checkpoints -- #7
6. Filesystem Checkpointing / Rollback -- #8
7. Signal Integration -- #4
