# Handoff — session-orchestration answerable feed (2026-07-01)

Goal: make `@Hermès so omp <repo> "<task>"` spawn a **managed** session that lands
in the **session_orchestration** unified feed — needs-input items posted with a
clickable thread link, text-reply-in-thread relayed into omp, auto-nudge on
stall. NO reactions (user preference). This is the `session_orchestration/`
subsystem (already built + `enabled: true`); the work here was fixing the
entry-point routing + a stack of spawn/detection/feed bugs.

## Architecture (which path is live)
- **`so <agent> <repo> "<task>"`** (Discord) → Hermès LLM → `session_spawn` tool
  (`session_orchestration/spawn_tool.py`) → `spawn.spawn_session` → omp/claude
  adapter launches a tmux session → registry row (`~/.hermes/state.db`) →
  `session-orchestration-watch` cron (every 1m) polls panes, opens attention
  items, and reconciles the **feed digest** (single edited message in
  `feed_channel_id=1520995687393792040`).
- The OLD `so`-MCP path (z-harness `scripts/hermes/mcp_hermes_orchestrator.py`
  FastMCP + `so-watcher` cron + reactions) is DEPRECATED and now disabled.

## Changes made (ALL UNCOMMITTED)

### hermes-agent repo (`~/.hermes/hermes-agent`, git repo)
- `session_orchestration/adapters/omp.py` — 5 spawn/detection fixes:
  1. Pane `:0.0` hardcode → resolve real pane id via `list-panes -F '#{pane_id}'`
     (broke under the user's `base-index 1`/`pane-base-index 1` tmux config).
  2. `shlex.join` for the omp launch command (prompt with `?`/specials broke zsh).
  3. `launch()` boots omp **bare** (no prompt arg) — prompt seeded via `drive()`
     (spawn_session step 7), matching claude adapter. Prevented double-delivery.
  4. New `_wait_for_launch` — waits for omp TUI chrome (`╭`/`╰`), not the idle
     prompt (which never appears while omp is busy on a launch prompt).
  5. New `_wait_for_ready` (used by `drive`) — omp has NO `❯`; ready == TUI box
     present AND not busy (`ACTIVITY_REGEX`). Old `_wait_for_prompt` never matched.
  6. `INPUT_REQUEST_REGEX` + `parse_pane_lifecycle` — omp selection menus (footer
     "enter select / esc cancel", no `❯`, spinner masking) now → `WAITING_USER`.
- `session_orchestration/adapters/claude_code.py` — same pane `:0.0`→real-pane-id fix.
- `session_orchestration/spawn_tool.py` — `spawn_tool_handler(..., **_injected)`
  (dispatcher injects `task_id`; handler crashed on the unexpected kwarg).
- `scripts/session-orchestration-watch.sh` — source `~/.hermes/.env` so the
  watcher has `DISCORD_BOT_TOKEN` (feed post silently `discord_failed` without it).
- `plugins/platforms/discord/adapter.py` — **REVERT CANDIDATE.** Old-path
  reaction wiring (`_render_so_needs_input`, `_so_reaction_index`,
  `_navigate_so_session`, `on_raw_reaction_add`, `_so_inflight`). Dead now that
  the old path is disabled — safe to revert (was aimed at the wrong subsystem).

### ~/.hermes config + skills (not in hermes-agent git)
- `~/.hermes/config.yaml`: `mcp_servers.so.enabled: false` (disabled old so-MCP);
  `session_orchestration.enabled: true` (already on).
- `~/.hermes/skills/autonomous-ai-agents/so-orchestration/SKILL.md`: rewritten
  (v3.0.0) to route `so` → `session_spawn` + hard "STOP on error, no manual, no
  mcp_so_*" rules. Original at `SKILL.md.deprecated-bak`.

### z-harness repo — committed this session (main branch, unpushed)
`c8f031f`, `de8e65d`, `adf440a`, `936c7c9` (old-path so-poller fixes — now moot
since the old path is disabled, but harmless). Working tree: `mcp_hermes_orchestrator.py`
shows modified — verify/discard.

## What WORKS (verified live)
- Spawn creates a real managed omp session + registry row (RUNNING).
- Watcher flips it to WAITING_USER when omp shows a menu.
- Feed digest posts/edits (verified: `reconcile` returned `status: edited`).

## REMAINING GAPS
1. **THREAD (next task) — real plumbing change, do in a fresh session.**
   Session has empty `discord_thread_id` → feed links a raw task-id (not
   `<#thread>`), AND text replies can't route to omp
   (`_handle_managed_thread_reply` at `gateway/run.py:~10990` keys off
   `discord_thread_id`). Root cause: `spawn_tool_handler`
   (`session_orchestration/spawn_tool.py:~185`) builds a `SpawnRequest` with NO
   `thread_creator`/`parent_chat_id`, so `spawn.py` step 6 (thread creation,
   gated on both) is skipped. The tool path CANNOT easily get the parent chat:
   the tool executor `model_tools.py:handle_function_call` → `registry.dispatch`
   injects ONLY `task_id` + `user_task` (verified). `task_id` =
   `_session_key_for_source(source)` (a SESSION KEY like `discord:...`, NOT the
   raw channel snowflake `create_handoff_thread` needs). `/so-spawn`
   (`gateway/run.py:_handle_so_spawn_command` → `spawn.handle_spawn_command`)
   works only because it's a command with the full `MessageEvent` + platform
   adapter: it reads `parent_chat_id = source.chat_id` and builds `thread_creator`
   from `platform_adapter.create_handoff_thread` (async, wrapped sync).
   **Fix recipe:** (a) plumb `chat_id` (raw `source.chat_id`) from the gateway
   agent-run (`_handle_message_with_agent`, where `source` is in scope) →
   `handle_function_call(..., chat_id=...)` → `registry.dispatch(..., chat_id=...)`
   → `spawn_tool_handler(**_injected)` reads `chat_id`; (b) in `spawn_tool_handler`
   get the Discord adapter via `gateway/run.py:_gateway_runner_ref()` →
   `.adapters.get(Platform.DISCORD)` → build a sync `thread_creator` around
   `create_handoff_thread` (mirror the wrapper in `spawn.handle_spawn_command`);
   (c) pass `parent_chat_id`+`thread_creator` into `spawn_session`. GOTCHA:
   `create_handoff_thread` needs a TEXT-CHANNEL parent — if the `so` convo is
   already in a Discord auto-thread, pass the parent channel id, not the thread.
   Simpler alt to consider: create the thread in the gateway AFTER the tool
   returns (where `source` is known) and `registry.upsert(discord_thread_id=...)`.
2. z-plan came up "What task should I plan?" — the task text didn't reach z-plan's
   args (prompt-delivery / z_command handling). Secondary.
3. omp not emitting `HERMES_MARKER_FILE` markers — pane detection covers it; non-blocking.

## TEST
`@Hermès so omp z-harness "add a docstring to navigate_so_session"` → expect a
thread created, session in feed with a clickable thread link, reply-in-thread →
into omp. Gateway restart picks up code changes: `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`.
Registry: `sqlite3 ~/.hermes/state.db "select task_id,state,discord_thread_id from session_orchestration order by rowid desc limit 3;"`

## COMMIT (when satisfied)
hermes-agent repo: commit the 4 keeper files (omp.py, claude_code.py,
spawn_tool.py, session-orchestration-watch.sh) + the thread fix; decide on
reverting adapter.py reaction code. config.yaml + the skill live outside the
repo — note them separately.
