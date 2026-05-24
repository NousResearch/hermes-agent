---

## Pass #69 – Agent Loop, Turn Processing & Conversation State Machine Deep Dive – 2026-05-25T16:30:00Z

Scope: agent/conversation_loop.py, agent/iteration_budget.py, agent/tool_executor.py, agent/chat_completion_helpers.py, run_agent.py, acp_adapter/server.py

---

### P69-1 · Main loop termination is robust — one unguarded bare except — LOW

**File:** `agent/conversation_loop.py` line 4241  
**Severity:** LOW

The `run_conversation()` main loop (line 644):

```python
while (api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
    if agent._interrupt_requested: break
    api_call_count += 1
    agent._api_call_count = api_call_count
    # ...api call...
```

- **Loop counter is incremented unconditionally at line 656** before any await, so a crash after increment but before the API call is made results in an iteration "lost" from the budget (not refunded, api_call_count higher). See lines 3040-3042 for explicit refunds on restart paths.
- The loop condition itself prevents infinite looping even if `api_call_count` gets out of sync.
- `_budget_grace_call` (lines 663-664) is consumed correctly before budget check.
- **No `try/finally` wrapping the main loop body.** A crash between `api_call_count += 1` and the API call loses one iteration with no recovery.

One bare `except` at line 4241 swallowing hook failures is intentional (best-effort hooks).

**Verdict:** Loop termination is well-guarded. The biggest risk is a hard crash between `api_call_count += 1` and the API call itself, losing one iteration. Not critical.

---

### P69-2 · Turn exit reason state machine — 14 valid exit paths, no corruption possible — CLEAN

**File:** `agent/conversation_loop.py` lines 581-4254  
**Severity:** CLEAN

The `_turn_exit_reason` local variable tracks why the tool loop exited. All 14 paths:

| Exit reason | Line | Circumstance |
|---|---|---|
| `unknown` | 581 | Default initialization |
| `interrupted_by_user` | 651 | `_interrupt_requested` at top of loop |
| `budget_exhausted` | 666 | Budget consume fails, no grace call |
| `ollama_runtime_context_too_small` | 942 | Ollama context check failure, with refund |
| `interrupted_during_api_call` | 3037 | Streaming interrupted mid-call |
| `all_retries_exhausted_no_response` | 3063 | `response is None` after all retries |
| `guardrail_halt` | 3481 | Tool guardrail blocked execution |
| `partial_stream_recovery` | 3593 | Stream backfill succeeded |
| `fallback_prior_turn_content` | 3620 | Fallback model returned valid prior content |
| `empty_response_exhausted` | 3797 | Empty response loop exhausted |
| `text_response(finish_reason=...)` | 3896 | Normal text response |
| `error_near_max_iterations(...)` | 3945 | Error within 5 of max iterations |
| `max_iterations_reached(...)` | 3959 | Budget exhausted after loop exit |
| (final log at 4054) | — | Diagnostic summary |

All paths that increment `api_call_count` either consume budget or call `refund()`. The refund pattern at lines 945-950 (Ollama context), 3041-3042 (compression restart), 3520 (guardrail halt) ensures no iteration is permanently lost on legitimate retry paths.

**Verdict:** State machine is clean. No impossible states or state corruption on errors.

---

### P69-3 · Error recovery — mostly graceful, one unguarded mid-loop crash risk — MEDIUM

**File:** `agent/conversation_loop.py` (multiple locations)  
**Severity:** MEDIUM

**What works well:**
- `_persist_session()` called on ALL exit paths: normal completion (4024), all retries exhausted (3065), budget exhaustion (1501), interrupt during backoff (1359), thinking exhaustion (1501). Excellent.
- `_drop_trailing_empty_response_scaffolding()` (run_agent.py 1182-1233) strips retry scaffolding from tails before persisting — prevents "user, user" role alternation corruption.
- `_flush_messages_to_session_db()` uses `_last_flushed_db_idx` to avoid duplicate writes (bug #860 fix).
- `try/except` wraps every individual API call, retry loop, and tool execution dispatch.
- Backoff sleep is interruptible (lines 1356-1377): polls `_interrupt_requested` every 0.2s.

**What is NOT guarded:**
- The block between `api_call_count += 1` (line 656) and the actual API call (line 1141) has no `try/except`. A crash there (OOM, signal) loses the iteration and corrupts the loop counter.
- No `try/except` around step_callback dispatch (lines 673-697) — only bare `except Exception as _step_err: logger.debug(...)`. Loop continues if it raises.

**Verdict:** Recovery is comprehensive for API-level errors. Gap is between loop counter increment and API call.

---

### P69-4 · Budget/exhaustion handling — well-designed, user notified — CLEAN

**File:** `agent/conversation_loop.py` lines 644-669, 3952-4003; `agent/iteration_budget.py`  
**Severity:** CLEAN

`IterationBudget` (62 lines, thread-safe):
- `consume()` returns `False` when exhausted — loop breaks cleanly at line 665-669.
- `refund()` used for: Ollama context errors (948), compression restarts (3042), tool guardrail halts (3520), execute_code program iterations (documented in class docstring).
- Grace call: when budget exhausts, `_budget_grace_call = True` and loop runs one more time. Flag cleared at line 663. After that call, budget check triggers and loop exits.
- `_budget_exhausted_injected` flag (agent_init.py:495) ensures grace call only fires once per exhaustion.

**User notification:**
- Line 668: `agent._safe_print(f"\n⚠️  Iteration budget exhausted ({used}/{max_total})...")`
- Lines 3957-3968: `_handle_max_iterations` triggers a summary request.
- `_turn_exit_reason` includes `max_iterations_reached(n/m)` for diagnostics.

**Verdict:** Budget handling is well-engineered. Thread-safe counter, explicit refunds for non-progress iterations, graceful summary call when exhausted.

---

### P69-5 · Conversation branching/resumption — session persistence covers most cases, one gap — MEDIUM

**File:** `run_agent.py` lines 1171-1297; `acp_adapter/server.py` lines 1086-1160  
**Severity:** MEDIUM

**Session persistence (`_persist_session` run_agent.py:1171):**
- Saves to JSON log and SQLite using `_last_flushed_db_idx` to avoid duplicate writes.
- `_drop_trailing_empty_response_scaffolding()` prevents corrupted tails on retry-exhausted turns.
- Called on every exit path — good.

**ACP `load_session` / `resume_session` (acp_adapter/server.py:1086-1160):**
- Both replay session history via `_replay_session_history()` before returning the response.
- Replay failures caught and logged but do not fail load/resume — partial transcript may be missing but session is usable.
- `cancel()` sets `cancel_event` and calls `agent.interrupt()` on the running agent — graceful.

**Mid-turn state preservation gap:**
- Agent does NOT preserve mid-turn state (tool execution in progress) across a session resume. If a session resumes while a tool is running, the tool is cancelled via interrupt and the turn restarts from the last user message.
- `_pending_steer` is dropped on hard interrupt (run_agent.py:1726) — correct since the turn it was meant for is gone.
- ACP `cancel` captures `interrupted_prompt_text` (line 1167) so the prompt is not lost — good.

**Verdict:** Session persistence is comprehensive. Mid-turn state is not preserved on resume — known limitation, not a bug.

---

### P69-6 · Interrupt mechanism — well-engineered, one thread-safety nuance — LOW

**File:** `run_agent.py` lines 1627-1726; `agent/tool_executor.py` lines 74-355  
**Severity:** LOW

**What works:**
- `interrupt()` (run_agent.py:1627): sets `_interrupt_requested = True`, fans out to execution thread via `_set_interrupt(True, _execution_thread_id)`, fans out to all concurrent tool worker threads, propagates to child agents.
- `clear_interrupt()` (1695): clears all bits, drops any pending `/steer`.
- Tool executor checks `agent._interrupt_requested` at: pre-flight (75), worker start (209), per-tool in loop (314), tool result (354), sequential dispatch (475).
- Worker tid tracking with `_set_interrupt(False, _worker_tid)` in `finally` block (tool_executor.py:255-261) — ensures clean exit from interrupt set.
- `_interrupt_thread_signal_pending` flag (agent_init.py:416) handles race where interrupt arrives before `run_conversation` sets `_execution_thread_id`.

**One nuance:**
- `_interrupt_requested` is a plain boolean (not atomic). `interrupt()` writes it without a lock. Tool executor reads it without a lock at multiple points. On x86 a bool write is atomic; on ARM it may not be. Low-risk — worst case is one extra tool call before interrupt is seen.

**Verdict:** Interrupt mechanism is well-designed with proper fan-out to workers and child agents. The bool write race is negligible in practice.

---

### P69-7 · Streaming API call interrupt — stale timeout + interrupt check — CLEAN

**File:** `agent/chat_completion_helpers.py` lines 79-276, 1211+  
**Severity:** CLEAN

Both `interruptible_api_call` and `interruptible_streaming_api_call` run the HTTP request in a background thread:

- Stale call detector: non-streaming kills at `_stale_timeout` (line 203-259). Streaming has 90s stale stream detection (conversation_loop.py:1097-1107).
- Interrupt check during polling: line 261-273 — if `_interrupt_requested` fires during the 0.3s poll loop, client is force-closed and `InterruptedError` is raised.
- `#29507` fix: thread ownership tracking for FD-recycling race — stranger threads only `abort()` the socket rather than fully closing, preventing kernel FD reuse bugs.

**Verdict:** Robust — stale timeout prevents infinite hangs, interrupt check terminates promptly, FD ownership tracking prevents kernel FD races.

---

**Summary:** The agent loop is well-engineered. Budget handling, interrupt propagation, session persistence, and error recovery are all thoughtfully implemented. The main gaps are: (1) no `try/finally` around the main loop body so a crash between counter increment and API call loses an iteration; (2) mid-turn state not preserved on session resume. Neither is critical. The codebase shows careful attention to retry loops, backoff, compression restarts, and graceful degradation across many edge cases.

**Files examined:** agent/conversation_loop.py (4258 lines), agent/iteration_budget.py (62 lines), agent/tool_executor.py (912 lines), agent/chat_completion_helpers.py (2170 lines), run_agent.py (4309 lines, key sections), acp_adapter/server.py (1952 lines, key sections), agent/agent_init.py (1637 lines, key sections).

*Pass #69 complete — 2026-05-25*