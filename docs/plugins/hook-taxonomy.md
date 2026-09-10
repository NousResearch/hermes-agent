# Hermes Plugin Hook Taxonomy

Reference documentation for every hook in `VALID_HOOKS` (37 as of this
writing). `hermes_cli/plugins.py` references this file twice (lines ~215
and ~6561); until now it did not exist. All fire sites, payload keys, and
dispatch modes below were extracted from source and verified against the
tree at time of writing — line numbers drift, file:hook names are the
stable contract.

## The contract in one paragraph

A Hermes plugin is a directory with `plugin.yaml` + `__init__.py` whose
`register(ctx)` calls `ctx.register_hook(name, fn)` (plus any of ~24 other
registration verbs). **`register(ctx)` is the only mount point** — a
`hooks:` list in plugin.yaml and module-level functions named after hooks
register NOTHING (verified 2026-08-26 the hard way: two guards silently
dead in production). Hook callbacks receive only the kwargs they declare
by name (plus `**kwargs` absorbs the rest) — the dispatcher filters the
payload to your signature, so narrow signatures survive payload evolution.
Every callback runs inside its own try/except: a raising plugin logs a
warning and the loop continues.

## Dispatch modes (4)

| Mode | Semantics | Hooks |
|---|---|---|
| **observer** | fire-and-forget; return ignored; failures logged, swallowed | most hooks |
| **transform** | first non-`None` return wins (1-step waterfall) | `transform_llm_output`, `transform_tool_result`, `transform_terminal_output`, `transform_api_error_classification` |
| **directive** | first valid `{"action": "block"\|"approve", "message": str}` wins; `resolve_pre_tool_block` (plugins.py:5935) is the single entry point, fail-closed for approvals | `pre_tool_call` |
| **middleware** | around-style, registered via `ctx.register_middleware` not hooks; 4 kinds (tool_request/tool_execution/llm_request/llm_execution) | orthogonal to this table |

Transforms and directives compose policy; observers must never mutate
anything the loop reads (payload dicts passed to observers are shared
snapshots — treat as read-only).

## The 37 hooks, grouped by fire site

### LLM/API path

- **`pre_llm_call`** — once per TURN, before context build (turn_context.py:1156).
  `session_id, task_id, turn_id, user_message, conversation_history,
  is_first_turn, model, platform, parent_session_id, sender_id`.
  Observer + context-injection: return `{"context": str}` to append text to
  the user message (oversized output spills to disk via hook_output_spill).
- **`pre_api_request`** — once per provider API CALL (conversation_loop.py:2675).
  `task_id, turn_id, api_request_id, session_id, user_message,
  conversation_history, platform, model, provider, base_url, api_mode,
  api_call_count, retry_count, request_messages, system_prompt,
  message_count, tool_count, approx_input_tokens, request_char_count,
  max_tokens, started_at, middleware_trace, request`.
  The richest observability hook; `request` is a sanitised payload dict.
  NOTE: `request_messages`/`conversation_history` are raw passthroughs
  (may contain secrets) — prefer `request["body"]["messages"]`.
- **`post_api_request`** — after each provider call (conversation_loop.py:6262).
  `task_id, turn_id, api_request_id, session_id, platform, model, provider,
  base_url, api_mode, api_call_count, api_duration, started_at, ended_at,
  finish_reason, message_count, response_model, response, usage,
  assistant_message`.
- **`api_request_error`** — provider call failed (run_agent.py:2919).
  `task_id, turn_id, api_request_id, session_id, platform, model, provider,
  base_url, api_mode, api_call_count, api_duration, started_at, ended_at,
  status_code, retry_count, max_retries, retryable, reason, error, request`.
- **`transform_llm_output`** — first non-None wins (turn_finalizer.py:609).
  `response_text, session_id, model, platform` → return new text. Fires only
  when a turn produced a final response without interruption.
- **`post_llm_call`** — turn finalised (turn_finalizer.py:595). Fires only
  when `final_response and not interrupted`. `session_id, task_id, turn_id,
  user_message, assistant_response, conversation_history, model, platform,
  usage`. NOT a reliable turn boundary — use time-window heuristics.

### Tool path

- **`pre_tool_call`** — the directive hook (model_tools.py:1375). `task_id,
  session_id, tool_call_id, turn_id, api_request_id, middleware_trace,
  function_name, function_args` → return `{"action": "block"|"approve",
  "message": str, "rule_key"?: str}`. First valid directive wins.
  resolve_pre_tool_block fail-closes approve paths (gate error → block).
- **`post_tool_call`** — after every tool result (model_tools.py:1168).
  `tool_name, args, result, task_id, session_id, tool_call_id, turn_id,
  api_request_id, duration_ms, status, error_type, error_message,
  middleware_trace`.
- **`transform_tool_result`** — first non-None wins (model_tools.py:1558).
  Same payload as post_tool_call → return replacement result.
- **`transform_terminal_output`** — (terminal_tool.py:3673). `command,
  output, returncode, task_id, env_type` → return new output text.

### Streaming (async, off the token path)

Enqueued via `enqueue_plugin_stream_hook` — never block these; they fire
off-thread from the streaming loop.

- **`on_stream_start`** — `turn_id, iteration, session_id, model, provider, surface`
- **`on_stream_delta`** — base + `delta` text
- **`on_stream_end`** — base + finish info
- **`on_interim_message`** — base + `text, already_streamed` (run_agent.py:6408)

### Session lifecycle

- **`on_session_start`** — new session created (conversation_loop.py:721).
  `session_id, model, platform`. Not fired on continuation turns.
- **`on_session_end`** — `session_id, task_id, turn_id, api_request_id,
  completed, interrupted, model, platform, reason`
- **`on_session_reset`** — (slash_commands.py:331). `session_id, platform,
  reason, old_session_id, new_session_id`
- **`on_session_finalize`** — hard-close (lifecycle.py:40 `finalize_session`)
- **`on_skill_lifecycle`** — (skill_usage.py:828). `action, skill_name,
  provenance, task_id, session_id, use_count, reused, reuse_after_patch,
  exc_info`

### Verification & commands

- **`pre_verify`** — final-answer verification gate (plugins.py:6034).
  `session_id, platform, model, coding, attempt, final_response,
  changed_paths`. Return non-None to reject the answer (re-loop).
- **`pre_command`** — slash-command intercept (plugins.py:5761). `surface,
  command, alias_used, args_raw, session_key, platform`.

### Approval lifecycle (tools/approval.py)

- **`pre_approval_request`** — `command, description, pattern_key,
  pattern_keys, session_key, surface, turn_id, tool_call_id` (redacted).
- **`post_approval_response`** — `turn_id, tool_call_id, choice, command,
  description` + tool info.

### Subagents

- **`subagent_start`** — (delegate_tool.py:1880). `parent_session_id,
  parent_turn_id, parent_subagent_id, child_session_id, child_subagent_id,
  child_role, child_goal`.
- **`subagent_stop`** — (delegate_tool.py:3228). `parent_session_id,
  parent_turn_id, child_session_id, child_role, child_summary,
  child_status, tool_call_history, duration_ms`.

### Kanban (all best-effort, post-commit)

Fired via `_fire_kanban_lifecycle_hook` (kanban_db.py:188) after the state
write commits; plugin failure never breaks a board transition. Common:
`task_id, board, profile_name`.

- **`kanban_task_claimed`** — + `assignee, run_id`
- **`kanban_task_completed`** / **`kanban_task_blocked`** — + result fields
- **`on_kanban_task_updated`** — field-level updates
- **`on_kanban_dispatch_tick`** — `board, profile_name, dry_run, outcome, result`
- **`on_kanban_worker_spawned` / `on_kanban_worker_exited` /
  `on_kanban_worker_stale_claim`** — worker lifecycle

### Gateway boundary

- **`gateway_platform_event`** — `invoke_hook("gateway_platform_event",
  **event)` — additive payload contract, event-typed (gateway/run.py:14768).
  Observer-only by design (#64176).
- **`pre_gateway_dispatch`** — before gateway dispatch. `event, gateway,
  session_store, register_body` + request metrics.

### Transcription

- **`pre_transcription`** — (transcription_tools.py:1409 helper). `file_path,
  provider, model, language, prompt, source`.

## Porting notes (DSH → Hermes)

- DSH's emit/waterfall/parallel/serial dispatch modes collapse here to:
  observer / transform / directive / middleware. Waterfall-with-short-circuit
  = the transform family + pre_tool_call directives.
- DSH's "model-visible means logged" maps to `pre_api_request` +
  `post_api_request` (the request ledger mount points — session-guard uses
  the turn-level `pre_llm_call` instead for cheapness).
- Signature discipline: declare only what you need + `**kwargs`. The
  dispatcher filters to your declared params, so narrow signatures survive
  payload evolution. BUT name them exactly as the fire site spells them
  (`conversation_history`, NOT `messages` — the mismatch that silently
  killed session-guard for 2.5h).

## Pitfalls

- **register(ctx) or nothing.** plugin.yaml `hooks:` lists are decorative.
- **Payload keys are the contract.** There is no schema validation on hook
  payloads (that's what the P4 seam adds for tools). Read the fire site.
- **Observers are shared snapshots.** Don't mutate payloads in place.
- **Streaming hooks are async-enqueued.** Slow work must go elsewhere.
- **post_llm_call skips interrupted turns** — not a turn boundary.
