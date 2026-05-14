# /loop Fix Plan: Live Source Audit Report

> **Audited:** 2026-05-14 | **Plan:** `docs/plans/loop-fix-plan.md` | **Live source:** `gateway/run.py`, `cli.py`, `hermes_cli/loop.py`

---

## Summary

| # | Claim | Verdict | Detail |
|---|-------|---------|--------|
| 1 | Attribute name: `self._event_loop` on GatewayRunner | **❌ WRONG** | See §1 |
| 2 | Injection: `call_soon_threadsafe` + `ensure_future` | **⚠️ NON-STANDARD** | See §2 |
| 3 | Daemon `_loop_ticker` thread exists in live code | **❌ FALSE** | See §3 |
| 4 | `_post_turn_loop_continuation` was `pass` (Bug 1) | **✅ FIXED** in live | See §4 |
| 5 | Kickoff + post-turn hook use `_enqueue_fifo` | **✅ CORRECT** | See §5 |
| 6 | `hermes_cli/loop.py`: MIN_INTERVAL, no [Loop check], clamping | **✅ CORRECT** | See §6 |
| 7 | `cli.py`: [Loop check] removed | **✅ CORRECT** | See §7 |
| 8 | `MessageEvent`, `MessageType` importable | **✅ CORRECT** | See §8 |
| A | "Changes Already Applied" — Change B (daemon thread) | **❌ NOT IN LIVE CODE** | See §A |
| B | Plan Task 1 code uses wrong attr name | **❌ WOULD BREAK** | See §B |

---

## §1 — Attribute Name: `_event_loop` vs `_gateway_loop` (CRITICAL)

### Plan says:
- Line 205 (Task 1 code): `loop = getattr(self, "_event_loop", None)`
- Line 560 (Edge Cases): "The `_event_loop` attribute on `GatewayRunner` is set during initialization."

### Live code says:
- Line 1224: `self._gateway_loop: Optional[asyncio.AbstractEventLoop] = None`
- Line 3280: `self._gateway_loop = asyncio.get_running_loop()`
- Line 11413: `loop = getattr(self, "_gateway_loop", None)`
- No occurrences of `_event_loop` anywhere in `gateway/run.py`.

### Verdict: **❌ WRONG**

The plan's proposed `_dispatch_loop_prompt` method at line 205 uses `self._event_loop`, which **does not exist** in the GatewayRunner class. This would always evaluate to `None`, causing `_dispatch_loop_prompt` to silently return (`if loop is None: return` at plan line 206-208). The daemon ticker would tick every second but never fire.

**Correction:** Use `self._gateway_loop` — the attribute that actually exists (declared at line 1224, set at line 3280, and referenced elsewhere at line 11413).

---

## §2 — Injection Mechanism: `call_soon_threadsafe` + `ensure_future` vs `run_coroutine_threadsafe`

### Plan says (line 224):
```python
loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_run()))
```

### Live code conventions say:
The gateway uses `asyncio.run_coroutine_threadsafe` pervasively (10 call sites):
- Line 11420: `asyncio.run_coroutine_threadsafe(self._rename_telegram_topic_for_session_title(...), loop)`
- Line 15023: `asyncio.run_coroutine_threadsafe(_hooks_ref.emit(...), ...)`
- Line 15056: `asyncio.run_coroutine_threadsafe(_status_adapter.send(...), _loop_for_step)`
- Line 15230, 15329, 15411, 15546, 15577, 16547, 16604 — all use `run_coroutine_threadsafe`

There is **zero** usage of `call_soon_threadsafe` in the entire file.

### Verdict: **⚠️ NON-STANDARD — should use `run_coroutine_threadsafe`**

The plan's mechanism works in principle but is inconsistent with the codebase. `asyncio.run_coroutine_threadsafe(coro, loop)` is:
- More idiomatic (standard Python API for cross-thread coroutine submission)
- Already used throughout the gateway
- More concise than `call_soon_threadsafe(lambda: asyncio.ensure_future(...))`
- Returns a `concurrent.futures.Future` for error handling

Additionally, `asyncio.ensure_future` is semi-deprecated in favor of `asyncio.create_task` in Python 3.11+, though it still functions. The plan's code at line 16047 shows the one place `ensure_future` is used (`_executor_task = asyncio.ensure_future(self._run_in_executor_with_context(run_sync))`), which is a different use case (already inside an async context).

**Recommendation:** Replace with:
```python
asyncio.run_coroutine_threadsafe(_run(), loop)
```

---

## §3 — Daemon `_loop_ticker` Thread (CRITICAL)

### Plan says:
- Line 59-60: "Fix applied: Added a background daemon thread (`_loop_ticker`) inside `_handle_loop_command` that ticks every second..."
- Line 139, 150-156: "Change B: Daemon thread + immediate kickoff in `_handle_loop_command` (line ~10532)" — listed under "Changes Already Applied"
- Lines 254-282 (Task 2): Shows the OLD code with `_loop_ticker` daemon thread, asking to replace its `_enqueue_fifo` call with `_dispatch_loop_prompt`

### Live code says:
- `_handle_loop_command` at lines 10517–10607
- **No `_loop_ticker` anywhere.** Searched entire file — **zero hits**.
- No daemon thread creation.
- No `sid` variable defined before any thread creation.
- The only loop-driving logic is:
  1. Immediate kickoff via `_enqueue_fifo` at lines 10591-10605
  2. Post-turn hook at line 9800-9861

### Verdict: **❌ FALSE — daemon thread does NOT exist in live code**

The plan claims Change B (daemon thread) was "already applied and synced." It was NOT. The live `_handle_loop_command` has no background thread. This means:

- **Bug 2** (post-turn hook cannot drive timer) is **UNRESOLVED** in the live gateway.
- After the immediate kickoff fires and the agent responds, the post-turn hook runs once, then nothing calls it again until the next user message arrives.
- This also means Task 2 in the plan (which describes modifying an existing `_loop_ticker`) is editing code that doesn't exist yet. The daemon thread must be **added first**, before it can be updated to use `_dispatch_loop_prompt`.

**Impact:** The plan's "already applied" list is misleading. The daemon thread must be implemented as part of the remaining work (Task 2), not merely updated. The plan Task 2 Step 3 (lines 315-357) effectively contains the full implementation of the daemon thread — it's just presented as a replacement of existing code that isn't there.

**Correction:** Task 2 should be renamed to "Add daemon ticker thread and use `_dispatch_loop_prompt`" and should not assume the daemon already exists.

---

## §4 — `_post_turn_loop_continuation` (Bug 1 Fix)

### Plan says:
- Lines 25-38: It was `pass` (Bug 1), replaced with real timer logic.
- Lines 139-148: "Change A: `_post_turn_loop_continuation` rewritten (line ~9746)"

### Live code says (lines 9800–9861):
```python
async def _post_turn_loop_continuation(self, *, session_entry, source, final_response):
    """Check loop timer and enqueue next turn if interval elapsed."""
    try:
        from hermes_cli.loop import load_loop, save_loop
        import time as _t
    except Exception:
        return
    if session_entry is None or not getattr(session_entry, "session_id", None):
        return
    sid = session_entry.session_id
    state = load_loop(sid)
    if state is None or state.status != "active":
        return
    now = _t.time()
    elapsed = now - state.last_fired_at
    if state.last_fired_at > 0 and elapsed < state.interval_seconds:
        return  # not time yet
    # Fire!
    state.last_fired_at = now
    state.turns_completed += 1
    save_loop(sid, state)
    # ... enqueue via _enqueue_fifo ...
```

### Verdict: **✅ FIXED in live code**

The `pass` is gone. The method contains real timer-checking logic as described. It is called at line 6808 after agent turns complete. However, as noted in §3, it only fires when agent turns complete — it cannot drive continuous ticking on its own.

**Note:** The plan says this method exists at "line ~9746" but it's actually at line 9800. The reference is approximately correct (off by ~54 lines).

---

## §5 — `_enqueue_fifo` Usage

### Plan claims:
- Kickoff (line ~10596) uses `_enqueue_fifo`
- Post-turn hook (line ~9849) uses `_enqueue_fifo`

### Live code:
- Line 10603: `self._enqueue_fifo(_quick_key, kickoff_event, adapter)` ✅
- Line 9858: `self._enqueue_fifo(_quick_key, kickoff_event, adapter)` ✅

### Verdict: **✅ CORRECT**

Both injection points use `_enqueue_fifo` exactly as described in the plan.

### Bug 5 analysis confirmed:
The `_enqueue_fifo` method (lines 2039-2053) correctly described in the plan:
- Line 2041: `if adapter is None: return` — silent no-op for TUI
- Line 2043-2044: `pending_slot = getattr(adapter, "_pending_messages", None)` / `if pending_slot is None: return`
- `_pending_messages` exists on `BasePlatformAdapter` at `gateway/platforms/base.py:1288`

The TUI path (`Platform.LOCAL`) has no `BasePlatformAdapter` subclass, confirming `_enqueue_fifo` is a dead end for TUI sessions.

---

## §6 — `hermes_cli/loop.py` Verification

| Claim | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| `MIN_INTERVAL_SECONDS` at line 32 | `MIN_INTERVAL_SECONDS = 60` | Line 32: `MIN_INTERVAL_SECONDS = 60` | ✅ |
| Interval clamping in `set()` | `effective_interval = max(int(interval_seconds) if interval_seconds else 300, MIN_INTERVAL_SECONDS)` | Lines 227-229: exact match | ✅ |
| `[Loop check]` prefix removed | No `[Loop check]` anywhere | `grep "[Loop check]"` → 0 results | ✅ |
| `on_message` popup removed | No callback block | Tick fires at line 396: `prompt = state.prompt` then `self._pending_input.put(prompt)` — no on_message | ✅ |

### Verdict: **✅ ALL CORRECT**

All three changes described in plan Section 1 (lines 110-130) are verified in the live `hermes_cli/loop.py`.

---

## §7 — `cli.py` Verification

| Claim | Expected | Actual | Verdict |
|-------|----------|--------|---------|
| `[Loop check]` removed from CLI kickoff | `self._pending_input.put(state.prompt)` | Line 7350: `self._pending_input.put(state.prompt)` | ✅ |

### Verdict: **✅ CORRECT**

The CLI `_handle_loop_command` at line 7350 injects `state.prompt` without any `[Loop check]` prefix.

---

## §8 — Import Verification

### Plan says (line 230-237):
Ensure `asyncio`, `threading` are imported; `MessageEvent` should be accessible.

### Live code:
- Line 27: `import asyncio` ✅
- Line 38: `import threading` ✅
- Line 641-648: `from gateway.platforms.base import (..., MessageEvent, MessageType, ...)` ✅

Both `MessageEvent` and `MessageType` are module-level imports, accessible everywhere in the class.

### Verdict: **✅ CORRECT**

All required imports exist at module top level.

---

## §A — "Changes Already Applied" Audit

The plan's Section 3 (lines 108-160) lists changes "Already Applied (Fork → Install Synced)":

### Change A: `_post_turn_loop_continuation` rewritten
- **Status: ✅ IN LIVE CODE** — Lines 9800-9861 contain the rewritten version

### Change B: Daemon thread + immediate kickoff in `_handle_loop_command`
- **Status: ❌ NOT IN LIVE CODE** — `_handle_loop_command` (lines 10517-10607) has the immediate kickoff but NO daemon thread
- Searched for `_loop_ticker` → 0 hits
- No `sid` variable, no `threading.Thread(...)`, no background ticker

**This is a significant discrepancy.** The plan treats Change B as done, but it's not in the live gateway. Task 2 of the plan assumes the daemon thread exists and needs updating, when it actually needs to be created from scratch.

---

## §B — Plan Task 1 Code Analysis

The proposed `_dispatch_loop_prompt` at lines 190-227 has two issues:

1. **Wrong attribute name** (see §1): `self._event_loop` → should be `self._gateway_loop`

2. **`_handle_message_with_agent` signature mismatch** (line 219):
   ```python
   await self._handle_message_with_agent(event, source)
   ```
   The actual signature at line 7077 is:
   ```python
   async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
   ```
   The method requires 4 positional arguments: `event`, `source`, `_quick_key`, and `run_generation`. The plan passes only 2. This would raise a `TypeError` at runtime.

   **Correction:** The call needs to generate `_quick_key` via `self._session_key_for_source(source)` and `run_generation` (likely 1 for a fresh turn, or incremented from existing state).

3. **`MessageEvent` construction**: The plan's code uses `message_id=None` which matches the field definition at `base.py:931` (`message_id: Optional[str] = None`). ✅

---

## §C — Additional Observations

### C1. Confirmation message accuracy
At live line 10607:
```python
return f"⊙ Loop set ({interval_seconds}s interval): {state.prompt}"
```
`interval_seconds` is the RAW parsed value, not the clamped value from `mgr.set()`. If user types `/loop 10s test`, the confirmation would show `10s` even though the loop is clamped to 60s. The plan does not address this. The plan's Task 2 Step 3 doesn't change the return statement.

### C2. Plan line number drift
The plan references lines in `gateway/run.py` that are approximate:
- Plan says `_post_turn_loop_continuation` at ~9746 → actual: 9800 (off by +54)
- Plan says `_handle_loop_command` at ~10532 → actual: 10517 (off by -15)
- Plan says `_enqueue_fifo` at ~2039 → actual: 2039 (exact)
- Plan says kickoff at ~10596 → actual: 10603 (off by +7)

These are reasonable approximations given file churn.

### C3. Thread safety concern
The plan's daemon thread (when implemented) sleeps 1 second between ticks. If the agent takes longer than 1 second to process a loop prompt, the ticker will fire again and potentially double-enqueue. The plan doesn't address idleness checking — the CLI's `LoopScheduler` uses `is_idle` callback (line 245 of `loop.py`), but the plan's gateway ticker has no such guard. This could cause rapid-fire duplicate prompts.

### C4. `_handle_message_with_agent` is async
The actual `_handle_message_with_agent` at line 7077 is `async def`. The plan correctly calls it with `await` inside the inner `_run()` coroutine. ✅

---

## §D — Complete Verdict Matrix

| # | Plan Claim | Plan Line(s) | Live Ref | Verdict |
|---|-----------|-------------|----------|---------|
| 1 | `_event_loop` attribute exists | 205, 560 | 1224, 3280 | **❌ WRONG** — is `_gateway_loop` |
| 2 | `call_soon_threadsafe` + `ensure_future` | 224 | — | **⚠️ Should use `run_coroutine_threadsafe`** |
| 3 | Daemon `_loop_ticker` exists | 59-60, 150-156 | — | **❌ NOT IN LIVE CODE** |
| 4 | `_post_turn_loop_continuation` was `pass` | 31-33 | 9800-9861 | **✅ FIXED** |
| 5a | Kickoff uses `_enqueue_fifo` | ~10596 | 10603 | **✅ CORRECT** |
| 5b | Post-turn hook uses `_enqueue_fifo` | ~9849 | 9858 | **✅ CORRECT** |
| 6a | `MIN_INTERVAL_SECONDS` at line 32 | 114 | loop.py:32 | **✅ CORRECT** |
| 6b | Interval clamping in `set()` | 117-125 | loop.py:227-229 | **✅ CORRECT** |
| 6c | `[Loop check]` removed | 128-129 | loop.py (0 hits) | **✅ CORRECT** |
| 7 | `[Loop check]` removed from cli.py | 136 | cli.py:7350 | **✅ CORRECT** |
| 8 | `MessageEvent`, `MessageType` importable | 237 | run.py:641-648 | **✅ CORRECT** |
| A | Change B "already applied" | 150-156 | run.py:10517-10607 | **❌ NOT APPLIED** |
| B1 | `_dispatch_loop_prompt` uses `_event_loop` | 205 | — | **❌ Would break** |
| B2 | `_handle_message_with_agent(event, source)` — 2 args | 219 | run.py:7077 (4 args) | **❌ Would raise TypeError** |

---

## §E — Required Plan Corrections

1. **Fix attribute name**: Change `self._event_loop` → `self._gateway_loop` everywhere in the plan (line 205, 560).

2. **Fix `_handle_message_with_agent` call**: Add `_quick_key` and `run_generation` parameters:
   ```python
   _qk = self._session_key_for_source(source)
   await self._handle_message_with_agent(event, source, _qk, 1)
   ```

3. **Switch to `run_coroutine_threadsafe`** for consistency with the codebase:
   ```python
   asyncio.run_coroutine_threadsafe(_run(), loop)
   ```

4. **Do not claim Change B (daemon thread) is already applied**. Task 2 should be restructured to first **add** the daemon thread, then update it to use `_dispatch_loop_prompt`.

5. **Add idleness guard** to the daemon thread to prevent double-enqueue during long agent turns.

6. **Fix confirmation message** to show the clamped `state.interval_seconds` instead of raw `interval_seconds`.

---

## Bottom Line

The plan correctly diagnoses the root causes and the architecture is sound, but it contains **3 critical errors** that would prevent the fix from working:

1. Wrong attribute name (`_event_loop` → `_gateway_loop`)
2. Wrong method signature (2 args → 4 args for `_handle_message_with_agent`)
3. Assumes daemon thread already exists when it doesn't

Once corrected, the plan is viable.
