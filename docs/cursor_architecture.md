# Cursor Provider Architecture

Hermes routes the `cursor` provider through the Cursor Agent CLI (`cursor-agent`), not an HTTP chat-completions API. Each chat turn spawns a short-lived subprocess; Hermes translates line-delimited JSON events into an OpenAI-shaped response and wires Cursor's internal tool activity into the same progress surfaces used by native Hermes tools. Conversation history, memory, skills, compression, and resume are all Hermes-managed; cursor sees a fresh prompt each turn, identical to every other provider in the system.

```
                       HERMES SIDE (source of truth)
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐   │
│  │  AIAgent     │    │  conversation_  │    │  session SQLite DB    │   │
│  │  (run_agent) │◄──►│  loop.py         │◄──►│  (transcript, /resume)│   │
│  └──────┬───────┘    └────────┬────────┘    └──────────────────────┘   │
│         │                     │                                        │
│         │ tool_registry +     │ /compress hook (duck-typed)             │
│         │ approvals.mode      │   → reset_context_baseline()            │
│         │                     │                                        │
│         │              ┌──────▼──────────┐    ┌─────────────────────┐  │
│         │              │ context         │    │ memory_tool +       │  │
│         │              │ compression     │    │ skill_manager_tool   │  │
│         │              │ (aux LLM)       │    │ (cross-provider)     │  │
│         │              └─────────────────┘    └─────────────────────┘  │
│         ▼                                                              │
│  ┌──────────────────────┐                                              │
│  │ tool_executor        │  ◄── Hermes-side tool calls:                 │
│  │ + activity feed UI   │      ⚡ shell, 📖 read, 🧠 memory,           │
│  └──────────────────────┘      🔧 skill_manage, 💬 narrate, etc.       │
│         ▲                                                              │
└─────────┼──────────────────────────────────────────────────────────────┘
          │
          │ ① one-instance-per-session, lazy-init scratch dir
          │
┌─────────┴──────────────────────────────────────────────────────────────┐
│                       BRIDGE                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              CursorAgentClient  (OpenAI-compat shim)              │  │
│  │  • chat.completions.create(messages, tools, …)                    │  │
│  │  • _session_workspace  (one dir per client; reused across calls)  │  │
│  │  • _context_high_water (bar floor; reset on compress)             │  │
│  │  • tool_progress_callback (forwards activity to AIAgent)          │  │
│  └──────────┬──────────────────────────────────────────────────▲────┘  │
└─────────────┼──────────────────────────────────────────────────┼───────┘
              │ ② per-turn subprocess spawn                      │
              │   (prompt on stdin, NDJSON on stdout)             │
              ▼                                                  │
┌───────────────────────────────────────────────────────────────┼────────┐
│                    CURSOR SIDE                                 │        │
│  ┌──────────────────────────────────────────────────────────┐ │        │
│  │  cursor-agent -p --output-format stream-json              │ │        │
│  │             --model X --workspace <session_workspace>     │ │        │
│  │             --force --trust  [--mode ask|plan, or none]   │ │        │
│  │                                                           │ │        │
│  │  • talks to cursor.com (proprietary ConnectRPC)           │ │        │
│  │  • runs OWN built-in tools internally:                    │ │        │
│  │    shellToolCall, readToolCall, editToolCall, …            │ │        │
│  │  • emits one JSON object per line:                        │ │        │
│  │    {type:system|thinking|assistant|tool_call|result}      │ │        │
│  └────────────────────────┬─────────────────────────────────┘ │        │
│                           │ ③ stream-json events                │        │
│                           ▼                                     │        │
│  ┌──────────────────────────────────────────────────────────┐ │        │
│  │  _StreamJsonAccumulator                                   │ │        │
│  │   • assembles assistant text → response.content           │ │        │
│  │   • lifts <tool_call>{…}</tool_call> → response.tool_calls├─┘        │
│  │   • normalizes tool_call events → _build_tool_event_bridge│          │
│  │     (forwards as tool_progress → activity feed) ──────────┘          │
│  │   • aggregates usage (per-round avg, not billing sum)                │
│  │   • detects terminal result → returns synthesized response           │
│  └──────────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────────┘

Legend / lifecycle:
  ① ONE CursorAgentClient per chat session. Workspace dir is created
     lazily on first call and reused for every subsequent call (perf
     fix 2026-05-28; saves ~4-5s/turn). Cleaned up on close()/`/new`.
  ② ONE cursor-agent subprocess per request. After the terminal
     `result` event we wait up to 700ms for natural exit before SIGTERM
     (Node.js shutdown hooks were delaying the exit it was about to do).
  ③ Two distinct tool channels (NOT confused, NOT merged):
       A) <tool_call> text blocks → lifted by accumulator →
          Hermes' tool_executor (memory, skill_manage, mcp, etc.)
       B) cursor's stream-json tool_call events → tool_progress_callback
          → activity feed (cursor's OWN shell/read/edit runs IN-PROCESS;
          Hermes observes but doesn't gate them past Hermes' approvals.mode)
```

**Key files:** `agent/cursor_agent_client.py` (runtime + accumulator + bridge), `plugins/model-providers/cursor/` (provider profile), `hermes_cli/auth.py` (credentials + status), `agent/agent_runtime_helpers.py:create_openai_client()` (client factory), `agent/conversation_compression.py` (compress + duck-typed reset hook), `agent/display.py` (`get_cute_tool_message`, `extract_edit_diff`; unified diff rendering for cursor edits).

---

## Authentication

Cursor is registered as `auth_type="external_process"` in `PROVIDER_REGISTRY` and `HERMES_OVERLAYS`. The marker base URL `cursor://agent` is never dereferenced over HTTP; it selects the subprocess client path.

| Path | Mechanism | Storage |
|------|-----------|---------|
| **CLI login (default)** | `cursor-agent login` → browser OAuth | OS keyring via Cursor CLI (`~/.config/cursor-agent/`) |
| **API key** | `CURSOR_API_KEY` env var | `~/.hermes/.env` (if pasted in setup) or shell env |

**Credential resolution** (`resolve_external_process_provider_credentials("cursor")`):

1. Resolve CLI path: `HERMES_CURSOR_COMMAND` → `CURSOR_AGENT_PATH` → `cursor-agent`.
2. Fail fast with `AuthError(missing_cursor_cli)` if the binary is not on PATH.
3. Return `api_key: CURSOR_API_KEY or "cursor-agent-login"`. The sentinel string tells Hermes "use the CLI session"; `CursorAgentClient` filters sentinels and **does not** pass them to `--api-key` (forwarding them caused `BrokenPipeError`).

**Status probing** (`get_external_process_provider_status("cursor")`):

- Runs `cursor-agent status` and parses `✓ Logged in as <email>`.
- Treats a set `CURSOR_API_KEY` as authenticated even without a status email.
- Surfaces command path, resolved binary, and login state to `hermes status`, `hermes auth status cursor`, and the model picker.

Aliases resolving to `cursor`: `cursor-agent`, `cursor-cli`, `cursor-sub`, `cursor-subscription`, `anysphere`.

---

## Subprocess Client

`CursorAgentClient` implements a minimal `client.chat.completions.create(**kwargs)` surface compatible with the rest of Hermes.

**Per-request lifecycle:**

1. **Prompt assembly.** `_format_messages_as_prompt()` flattens the OpenAI message list (system/user/assistant/tool) into a single stdin prompt. Tool schemas are inlined as JSON; the model is instructed to emit Hermes tool calls as `<tool_call>{...}</tool_call>` blocks (grammar shared with `copilot_acp_client`).
2. **Workspace.** Session-scoped: one temp dir per `CursorAgentClient` instance, reused for every call. Created lazily on first call as `hermes-cursor-*`, tracked in `_ephemeral_dirs` for cleanup at `close()`. Override with `HERMES_CURSOR_WORKSPACE` or the `workspace` ctor arg. A fresh dir per call previously cost roughly 4 to 5 seconds of "first-time workspace bootstrap" tax on every turn; fixed by reusing the dir across the session.
3. **Argv.** `cursor-agent -p --output-format stream-json --model <m> --workspace <ws> --force --trust` plus optional `--mode`, `--api-key`, and `HERMES_CURSOR_ARGS`.
4. **Mode mapping:**

   | Hermes `HERMES_CURSOR_MODE` | CLI behaviour |
   |-----------------------------|---------------|
   | `agent` **(default)** | **omit** `--mode`; Cursor's full default permissionMode (shell, write, edit, …). Matches `cursor-agent -p` direct usage. |
   | `ask` | `--mode ask`; read-only; Cursor's built-in shell/write disabled |
   | `plan` | `--mode plan`; read-only planning |

   The default flipped from `ask` to `agent` on 2026-05-28 to remove a silent demotion that made cursor feel "broken" out of the box (users coming from `cursor-agent` directly expected full power). Hermes' own `approvals.mode` config (manual/smart/off) gates dangerous tool execution on top of this, identical to every other provider.

5. **I/O.** prompt written to stdin (avoids argv length limits); stdout read line-by-line in a background thread; stderr drained concurrently for auth/flag diagnostics.
6. **Timeout.** default 600 s; process terminated on expiry.
7. **Termination.** after the terminal `result` event, the client waits up to 700 ms for cursor-agent's natural exit before SIGTERM (Node.js shutdown hooks otherwise delay the exit it was already about to do). Force-kill fallback after 1.5 s.
8. **Session-level conversation continuity.** Hermes is the source of truth for conversation history (transcript sent fresh every turn). cursor's own `--resume [chatId]` / `--continue` flags are deliberately NOT used; splitting that authority would desync `/clear`, `/new`, `/compress`, and switch_model. Files cursor wrote on turn N remain in the session workspace for turn N+1 (workspace reuse, item 2 above).

Client construction happens in `agent/agent_runtime_helpers.py:create_openai_client()` when `provider == "cursor"` or `base_url` starts with `cursor://`.

---

## stream-json Parsing

The CLI emits **one JSON object per line** on stdout. `_StreamJsonAccumulator.feed()` consumes events until a terminal `result` arrives.

| Event `type` | Handling |
|--------------|----------|
| `system` | Capture `model`, `session_id` |
| `thinking` | Append to reasoning buffer |
| `assistant` | Extract text blocks from `message.content[]` |
| `tool_call` | `subtype=started` / `completed` → `_CursorToolEvent` (see below) |
| `result` | Terminal: `is_error`, `duration_ms`, `usage`, final `result` text |

**Usage normalization:** Cursor's camelCase keys (`inputTokens`, `outputTokens`, `cacheReadTokens`) map to OpenAI `usage` / `prompt_tokens_details.cached_tokens`.

**Response assembly:**

1. Join accumulated assistant text.
2. Run `_extract_tool_calls_from_text()` to lift `<tool_call>` blocks into OpenAI `tool_calls`.
3. Attach `cursor_internal_tools`; audit list of Cursor's own harness invocations.
4. Set `finish_reason` to `tool_calls` or `stop`.

**Streaming note:** Hermes disables true streaming for Cursor in `conversation_loop.py` (same as `copilot-acp`). If a caller passes `stream=True`, `_synthesise_stream_chunks()` yields a small OpenAI-style chunk iterator from the fully assembled response (defence-in-depth).

---

## Tool-Event Surfacing in the Hermes UI

Two parallel tool channels exist; they must not be conflated.

### 1. Hermes-side tool calls (host agent loop)

When the Cursor **model** emits `<tool_call>` blocks, Hermes extracts them and executes tools via `tool_executor.py`; same path as OpenAI/Anthropic/Grok. These appear in session DB `tool_calls`, increment `tool_call_count`, and fire the normal `tool.started` / `tool.completed` callbacks.

The prompt deliberately frames the model as **"pure LLM backend, NOT cursor-agent"** so side-effecting work (shell, writes) escalates to Hermes rather than running silently inside the subprocess.

### 2. Cursor-internal tool calls (subprocess harness)

When `cursor-agent` runs its **own** built-in tools (shell/read/edit/grep/…), events arrive as stream-json `tool_call` envelopes (`shellToolCall`, `readToolCall`, …). `_StreamJsonAccumulator._consume_tool_call_event()` builds `_CursorToolEvent` records and invokes `_build_tool_event_bridge()`:

```
cursor stream-json  →  _CursorToolEvent  →  tool_progress_callback
                                              ("tool.started" | "tool.completed")
```

**Name mapping** (`_normalize_cursor_tool_name`): e.g. `shellToolCall` → `shell`, `readToolCall` → `read_file`.

**Preview strings** (`_build_cursor_tool_preview`): command, path, or pattern; same spirit as `tool_executor._build_tool_preview`.

**UI consumers:**

| Surface | Callback wiring | What the user sees |
|---------|-----------------|-------------------|
| Classic CLI | `cli.py:_on_tool_progress` | Spinner label + optional scrollback lines (`tool_progress_mode`) |
| TUI (`hermes --tui`) | `tui_gateway/server.py` → `tool.progress` JSON-RPC event | Activity feed in Ink `thinking.tsx` |
| Gateway / API | `gateway/run.py` progress callback | Platform progress messages / SSE |

The agent's `tool_progress_callback` is passed into `CursorAgentClient` at construction time (`agent_runtime_helpers.py`). Callback errors are swallowed so a broken UI never aborts a chat call.

Completed internal events also populate `response.cursor_internal_tools` / `message.cursor_internal_tools` for session audit, even though they are not Hermes `tool_calls`.

---

## Known Limitations

**Architecture**

- **One subprocess per request, but warm workspace.** `--resume` not used (Hermes is source-of-truth for history); however the session workspace is reused across calls within one `CursorAgentClient` instance, cutting ~4-5 s of cursor-side bootstrap tax per turn (2026-05-28 perf pass). Cold-start latency on the FIRST call of a session is still ~12-15 s (cursor.com server warm-up); subsequent calls in the same session land at ~13-15 s.
- **Not true streaming.** Tokens arrive only after the subprocess completes (or synthetic chunks are replayed). Streaming is explicitly disabled for cursor in `conversation_loop.py:_use_streaming`.
- **Workspace isolation.** Cursor-agent's CLI doesn't sandbox absolute paths; even with `--workspace /tmp/scratch`, a `shell` tool call can `cat /home/user/secrets`. The workspace is a *cwd hint*, not a security boundary. Hermes tools operate on the real cwd as always.

**Tool semantics**

- **Dual tool stacks.** Even with hardened prompting, Cursor's harness may intercept reads/listings/greps internally. Those do **not** become Hermes `tool_calls`; they surface only via internal `tool_call` stream events that fire `tool_progress_callback("narrate"/"shell"/...)` in the activity feed. Audit-conscious deployments should set `HERMES_CURSOR_MODE=ask` to disable cursor's internal mutation tools and force all writes through Hermes' `<tool_call>` channel.
- **Default mode is `agent`.** Cursor's full permission mode (shell, write, edit). Hermes' `approvals.mode` config (manual/smart/off) provides the cross-provider safety gate. Set `HERMES_CURSOR_MODE=ask` for read-only.
- **Read-side cache/audit.** Hermes `read_file` dedup and read receipts do not fire for Cursor-internal reads. Prefer a pure-LLM provider if every read must appear in the UI.

**Auth & ops**

- **Separate CLI auth from IDE.** Cursor IDE login (`~/.config/Cursor/`) is not auto-imported; users run `cursor-agent login` or set `CURSOR_API_KEY`.
- **Sentinel api_key filtering.** placeholders like `cursor-agent-login` must never reach `--api-key`.
- **CLI flag drift.** only `ask` and `plan` are valid `--mode` values for cursor-agent; unknown values cause a hard crash. Our synthetic `agent` value is encoded by omitting `--mode` entirely (cursor's own default permissionMode).
- **Model catalog.** live list via `cursor-agent --list-models`; falls back to curated snapshot in `hermes_cli/models.py` when CLI is missing or unauthenticated.
- **No native Windows binary.** cursor-agent is macOS/Linux only (and Windows-via-WSL). The picker detects Windows and prints WSL install instructions instead of running `curl | bash`.

**Policy**

- Programmatic use of `cursor-agent` as an LLM proxy sits in the same grey area as other CLI-bridge providers (Copilot ACP, Gemini CLI). Hermes forwards the user's own identity and does not redistribute responses.

---

## Environment Variables (optional)

| Variable | Default | Purpose |
|----------|---------|---------|
| `CURSOR_API_KEY` |; | API key; bypasses CLI login |
| `HERMES_CURSOR_COMMAND` / `CURSOR_AGENT_PATH` | `cursor-agent` | CLI binary path |
| `HERMES_CURSOR_ARGS` |; | Extra argv appended to every invocation |
| `HERMES_CURSOR_MODE` | `agent` | `agent` (default, full power) / `ask` (read-only) / `plan` (read-only planning) |
| `HERMES_CURSOR_WORKSPACE` | session-scoped temp dir | Pin workspace directory (reused across all turns of one session by default) |
| `HERMES_CURSOR_BASE_URL` | `cursor://agent` | Provider marker (not HTTP) |
