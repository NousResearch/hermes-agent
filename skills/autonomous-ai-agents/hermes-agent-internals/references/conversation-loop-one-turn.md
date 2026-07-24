# conversation_loop.py — one turn line anchors

Repo-relative paths. Re-verify with `search_files` after pulling `main`.

## Entry

| Symbol | File | Approx lines |
|--------|------|--------------|
| `run_conversation` | `agent/conversation_loop.py` | 497+ |

## Prologue

| What | Lines (approx) | Notes |
|------|----------------|-------|
| Comment block "Per-turn setup (the prologue)" | 542–549 | |
| `build_turn_context(...)` | 550–566 | `agent/turn_context.py` |
| Unpack `_ctx` (messages, `_should_review_memory`, etc.) | 567–577 | |
| Codex app-server bypass (optional) | 603–610 | Skips normal loop |

## Main loop

| What | Lines (approx) | Notes |
|------|----------------|-------|
| `while` iteration guard | 612+ | `max_iterations`, `iteration_budget` |
| Build `api_messages` from `messages` | 724–950+ | `api_messages` starts ~761 |
| Model call + inner retries | 979–1465 | |
| Normalize `finish_reason` | 1466–1501 | Per `api_mode` / transport |
| `content_filter` refusal path | 1517–1603 | Early return |
| `finish_reason == "length"` (output cap / truncation) | 1605–1700+ | Continuation / exhaust paths |
| Context overflow → `_compress_context` | 2830–3316, 4412–4421 | Also `conversation_history_after_compression` |
| `if assistant_message.tool_calls:` | 4064+ | Not user-visible final answer |
| `_execute_tool_calls` | 4331 | → `run_agent.py` ~5258 → `tool_executor.py` |
| `continue` after tools | 4427 | Next lap |
| `else` — no tool calls (final response path) | 4429–4818 | `break` on success ~4813–4818 |

## Epilogue

| What | Lines (approx) | Notes |
|------|----------------|-------|
| Handoff to `finalize_turn` | 4877–4895 | `agent/turn_finalizer.py` |
| `_persist_session` (SQLite `state.db`) | turn_finalizer ~163–188 | |
| `_spawn_background_review` (memory/skills) | turn_finalizer ~435–461 | After answer; best-effort |

## Related files

- `agent/turn_context.py` — prologue implementation
- `agent/turn_finalizer.py` — epilogue implementation
- `agent/tool_executor.py` — `execute_tool_calls_sequential` / `concurrent`
- `run_agent.py` — `AIAgent`, `_execute_tool_calls` dispatcher

## Quick check (teaching)

**Q:** Model returns `finish_reason: tool_calls`. Does Hermes show the user a final reply?

**A:** No. Run tools, append tool messages, loop again. User gets text when the model stops with no pending tool_calls (`stop` path).