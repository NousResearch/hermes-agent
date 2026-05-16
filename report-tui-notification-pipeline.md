# TUI Notification Delivery Pipeline — Investigation Report

## 1. GitHub Issues Summary

### Issue #15248 — "TUI: notify_on_complete silently drops"
- **Reported by**: yonefive71 (Apr 2026)
- **Bug**: `terminal(background=true, notify_on_complete=true)` in TUI mode produces zero delivery. `grep "injecting agent notification" agent.log` returns nothing for TUI sessions, vs working entries for Discord.
- **Evidence**: Agent log shows no notification injection for TUI even when `process(action='poll')` confirms `status: exited`.
- **Root cause**: TUI backend `tui_gateway/server.py` had zero references to `completion_queue`, `notify_on_complete`, or any process notification machinery at the time of filing.
- **Status**: **FIXED** — the `_notification_poller_loop` was added to server.py.

### Issue #26071 — "notify_on_complete notifications silently lost in TUI mode"
- **Reported by**: alt-glitch (May 2026)
- **Bug**: Same root symptom as #15248 but filed later with deeper analysis. Documents that the `process_registry.completion_queue` is never drained in the TUI backend.
- **Proposed fix**: Describes two design options: (a) drain after every agent turn from inside `run_conversation()` by registering a callback, (b) a background poller thread in `tui_gateway/server.py`.
- **Also notes**: Gateway mode has its own secondary drain (gateway/run.py L7763-7768) that pops **all** events from `completion_queue` but discards `completion` type events with the comment `# else: completion events are handled by the watcher task`. This means `_move_to_finished()` wasted work enqueuing the event in gateway mode.
- **Status**: **FIXED** — the `_notification_poller_loop` implements option (b).

---

## 2. Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NOTIFICATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  terminal(background=true, notify_on_complete=true)                   │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐     │
│  │ terminal_tool.py │───▶│ ProcessRegistry.spawn()             │     │
│  │ sets:            │    │ Sets session.notify_on_complete=True │     │
│  │ notify_on_complete│    └────────┬────────────────────────────┘     │
│  └─────────────────┘              │                                  │
│                                   │ process exits                    │
│                                   ▼                                  │
│                    ┌──────────────────────────────┐                   │
│                    │ ProcessRegistry._move_to_finished()             │
│                    │  • session.exited = True      │                   │
│                    │  • enqueue → completion_queue │                   │
│                    │    {type:"completion", ...}   │                   │
│                    └──────────────┬───────────────┘                   │
│                                   │                                   │
│                                   ▼                                   │
│                    ┌──────────────────────────────┐                   │
│                    │     completion_queue          │                   │
│                    │  (global queue.Queue)         │                   │
│                    └──────┬───────────┬───────────┘                   │
│                           │           │                               │
│              ┌────────────┘           └────────────┐                  │
│              ▼                                     ▼                  │
│  ┌───────────────────┐              ┌─────────────────────┐           │
│  │ CLI process_loop   │              │ TUI _notification_  │           │
│  │ (daemon thread)    │              │ poller_loop         │           │
│  │                    │              │ (daemon thread)     │           │
│  │ Drain:             │              │ Drain:              │           │
│  │ 1. Pre-input check │              │ 1. queue.get(0.5)   │           │
│  │    (single evt)    │              │ 2. Check consumed   │           │
│  │ 2. Post-turn drain │              │ 3. format→status.up │           │
│  │    (all pending)   │              │ 4. If running:      │           │
│  │ 3. Idle drain      │              │    re-queue, retry  │           │
│  │    (all pending)   │              │ 5. Else: set running│           │
│  │                    │              │    → _run_prompt_   │           │
│  │ injects into       │              │    submit(notification│         │
│  │ _pending_input Q   │              │    = agent turn)    │           │
│  └───────────────────┘              └─────────────────────┘           │
│                           │                                           │
│              ┌────────────┘                                           │
│              ▼                                                        │
│  ┌──────────────────────┐                                            │
│  │ Agent turn injected  │                                            │
│  │ as user message      │                                            │
│  │ "[IMPORTANT: ...]"   │                                            │
│  └──────────────────────┘                                            │
│                                                                      │
│  ┌───────────────────────────────────────────────────────┐           │
│  │ GATEWAY path (for comparison)                         │           │
│  │                                                        │           │
│  │ 1. _run_process_watcher polls session.exited           │           │
│  │ 2. Detects exit independently via polling              │           │
│  │ 3. Injects synth_text as MessageEvent via adapter      │           │
│  │ 4. Secondary drain (after agent run): pops queue but   │           │
│  │    discards "completion" type events (handled above)   │           │
│  │ 5. Only passes watch_match/watch_disabled events       │           │
│  └───────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Analysis

### 3a. `tools/process_registry.py` — Event Production

**`_move_to_finished(session)`** (line 797):
- Guarded by idempotency check: only enqueues if `was_running` (first move) AND `session.notify_on_complete`.
- Enqueues into `self.completion_queue` with fields: `{type:"completion", session_id, command, exit_code, output}`.
- `_completion_consumed` set prevents duplicate delivery when user already called `poll()`/`wait()`/`log()` — those methods add the session ID to this set.

**`format_process_notification(evt)`** (line 1410, module-level function):
- Returns `[IMPORTANT: ...]` formatted string.
- Handles types: `"completion"`, `"watch_match"`, `"watch_disabled"`.
- Default type is `"completion"`.

**`drain_notifications()`** (line 829):
- Pops all events from `completion_queue`, filters consumed ones, returns list of `(event, formatted_text)`.
- Used by CLI process_loop (idle drain + post-turn drain).

### 3b. `tui_gateway/server.py` — TUI Poller

**`_notification_poller_loop(stop_event, sid, session)`** (line 3070):
- Thread loop: waits 0.5s on `completion_queue.get()`.
- **Step 1**: Check if event already consumed via `is_completion_consumed()`.
- **Step 2**: Format via `format_process_notification()`.
- **Step 3**: Emit `status.update(kind="process")` for UI visibility.
- **Step 4**: If session is `running` → **re-queue** the event and `continue` (defer until agent idle).
- **Step 5**: Set `session["running"] = True`, emit `message.start`, and call `_run_prompt_submit()` with the notification text as user message.
- On shutdown: drains any remaining events (with re-queue guard).
- Stop event is stored as `session["_notif_stop"]` and signaled in `_finalize_session()`.

**`_run_prompt_submit(rid, sid, session, text)`** (line 3169):
- Snapshot: copies `history`, `history_version`, `images` under `history_lock`.
- Spawns inner `run()` function that calls `agent.run_conversation()` with the notification text.
- Stream callback emits `message.delta` events to the TUI.
- On completion: updates history, emits `message.complete`.

### 3c. `cli.py` — Classic CLI Drain

**`process_loop()`** (line 13660):
- **Pre-input drain** (line 13666-13678): Pops one event from `completion_queue` before checking for pending user input. **BUG**: calls `_format_process_notification(evt)` (undefined name) instead of `format_process_notification` from `tools.process_registry`. Wrapped in `except Exception: pass`, so this path silently fails.
- **Idle drain** (line 13689-13692): When agent is idle and no user input pending, calls `process_registry.drain_notifications()` (correct) and injects into `_pending_input`.
- **Post-turn drain** (line 13801-13806): After each agent turn (`self.chat()` completes), calls `process_registry.drain_notifications()` (correct).
- **Kanban drain** (line 13811-13852): Separate thread polls `kanban_db` for task events and injects formatted notifications into `_pending_input`. Uses `_kanban_cli_cursors` dict for cursor persistence.

### 3d. `gateway/run.py` — Gateway Watcher

**`_run_process_watcher(watcher)`** (line 13947):
- Independent asyncio task per background process.
- Polls `process_registry.get(session_id)` every `interval` seconds.
- On exit detected (session.exited): generates its own `[IMPORTANT: ...]` string independent of `completion_queue` — does NOT consume from the queue.
- Injects via `adapter.handle_message(synth_event)` as a `MessageEvent` with `internal=True`.
- **Secondary drain** (line 7963-7978): After each agent run, drains `completion_queue` but **filters out** `"completion"` type events (they're handled by the watcher task). Only passes `watch_match` and `watch_disabled` events.

---

## 4. Identified Gaps & Bugs

### GAP 1: CLI `_format_process_notification` undefined reference (MEDIUM)
- **File**: `cli.py`, line 13674
- **Bug**: Calls `_format_process_notification(evt)` — a function that **does not exist** anywhere in the codebase. The correct name is `format_process_notification` from `tools.process_registry`.
- **Impact**: The pre-input drain silently skips all events via `except Exception: pass`. The idle drain and post-turn drain still work (they use `drain_notifications()` correctly), so this is a partial loss — but still means a notification arriving while the agent is idle (between user inputs) might be missed.
- **Fix**: Change `_format_process_notification` → `format_process_notification` and add proper import. Or just remove this redundant pre-input drain and let the idle drain handle it.

### GAP 2: TUI event re-queueing — ordering & potential loss (LOW-MEDIUM)
- **File**: `tui_gateway/server.py`, lines 3102-3105
- **Behavior**: When agent is running, the poller re-queues the event via `completion_queue.put(evt)`. This is correct for deferral, but:
  - **Re-ordering**: Multiple notifications arriving during a long agent turn are re-queued in FIFO order, but interleaved with any newly arriving notifications. A notification that has been re-queued multiple times could be processed after a later notification.
  - **Re-queue race**: Between checking `running` (line 3103) and setting `running = True` (line 3106), a second poller instance could also pop an event. Both could proceed, potentially injecting two agent turns in quick succession. The `history_lock` prevents concurrent mutations but doesn't prevent dual `_run_prompt_submit` calls.
  - **Re-queue storm**: If a process produces many events (e.g., watch matches while process is running), and the agent stays busy, the same event could be re-queued many times. But each re-queue is just a `put()/get()` cycle, so the rate is bounded by the 0.5s timeout.

### GAP 3: TUI lacks cursor persistence for notification dedup (LOW)
- The CLI kanban drain loop maintains `_kanban_cli_cursors` dict that persists event IDs per (task_id, chat_id). The TUI notification poller has no equivalent.
- For `completion_queue` events there's no cursor needed (each event is singleton per process). But for future kanban integration, the TUI would need the same cursor mechanism.
- The `_completion_consumed` set in `process_registry` provides cross-session dedup for process completions (marked when `poll()`/`wait()`/`log()` is called). The TUI correctly checks this.

### GAP 4: TUI `_notification_poller_loop` doesn't mark completions consumed on delivery (LOW)
- When the poller successfully dispatches a notification via `_run_prompt_submit`, it does NOT call `process_registry.mark_completion_consumed()`. (Note: there's no such method — the set is managed internally by `poll()`/`wait()`/`log()`.)
- This means if the agent uses `process(action='wait')` in response to a notification, the wait tool handler marks the session as consumed. But if the user never polls/wait/logs, the session stays "unconsumed" forever (harmless, since the event was already dequeued and delivered).
- Edge case: If the TUI sends a notification, then the user types something that triggers poll/log before the agent sees the notification, the consumed check prevents duplicate delivery. This is the correct behavior.

### GAP 5: Double-consumption guard redundancy in TUI poller (INFO)
- The poller at line 3093 checks `is_completion_consumed()` before formatting. This is the same check that `drain_notifications()` does internally. The poller calls `format_process_notification` directly (not via `drain_notifications`), so the explicit check is necessary — just slightly redundant with what could be a call to `drain_notifications()` for single-event extraction. Not a bug, but a readability/code-sharing opportunity.

### GAP 6: Watch pattern events delivered as agent turn in TUI but discarded in gateway (INFO)
- **TUI**: Watch pattern events trigger agent turns via `_run_prompt_submit`.
- **Gateway**: Watch pattern events drain after agent run and inject as `MessageEvent`.
- Different injection mechanisms but semantically equivalent. However, the TUI does NOT filter watch events from the poller when the gateway's `_run_process_watcher` task is active — but in TUI mode, `_run_process_watcher` is never started (TUI doesn't use the gateway adapter). So no double-delivery risk in practice.

### GAP 7: Terminal tool routing metadata not set for TUI mode (LOW)
- **File**: `tools/terminal_tool.py`, lines 1964-1976
- The `watcher_platform` metadata is populated from `HERMES_SESSION_PLATFORM` env var. In TUI mode, this env var is NOT set (the agent runs with `platform="tui"` but the session context env var is gateway-specific).
- **Impact**: The `if proc_session.watcher_platform:` guard at line 2002 prevents registering a pending_watcher entry. This is correct for TUI (no gateway watcher needed) and doesn't affect the `completion_queue` delivery path (the `notify_on_complete` flag IS set at line 1996 regardless of watcher_platform).

---

## 5. Recommendations

### Recommendation 1: Fix CLI `_format_process_notification` name
Add the proper import or rename to `format_process_notification`:
```python
# In cli.py process_loop, line 13674:
from tools.process_registry import format_process_notification
_synth = format_process_notification(evt)
```
Or better, use `process_registry.drain_notifications()` (already done for the idle and post-turn drains) and remove the redundant manual extraction.

### Recommendation 2: Add TUI kanban notification bridge
Extend `_notification_poller_loop` (or add a sibling `_kanban_notification_poller`) to drain kanban `task_events` table events, following the CLI's `_kanban_drain_loop` pattern:

```python
def _kanban_notification_poller(stop_event, sid, session):
    """Poll kanban task_events for subscribed tasks, inject as agent turns."""
    from hermes_cli import kanban_db as _kb
    _cursors = {}  # (task_id, chat_id) → max_event_id
    
    while not stop_event.is_set() and not session.get("_finalized"):
        try:
            conn = _kb.connect()
            try:
                subs = _kb.list_notify_subs(conn)
                for sub in subs:
                    if sub.get("platform") != "tui":
                        continue
                    cursor_key = (sub["task_id"], sub.get("chat_id", sid))
                    last_cursor = _cursors.get(cursor_key, 0)
                    _, events = _kb.unseen_events_for_sub(...)
                    if events:
                        _cursors[cursor_key] = max(e.id for e in events)
                    for ev in events:
                        msg = _format_kanban_notification(ev, sub)
                        if msg:
                            # Same delivery pattern: status.update + _run_prompt_submit
                            ...
            finally:
                conn.close()
        except Exception:
            pass
        try:
            stop_event.wait(2.0)  # poll every 2s
        except Exception:
            break
```

**Design notes**:
- Reuse `_run_prompt_submit` for agent turn injection (same as process notifications).
- Cursor persistence: store cursors in-memory per TUI session (volatile, lost on TUI restart) or in `_sessions[sid]`. For persistence across TUI restarts, store in the kanban DB itself.
- Rate limiting: Expose a `--kanban-notify-interval` or default to 2s.

### Recommendation 3: Single-event extraction helper for TUI poller
The poller's "consume-or-re-queue" logic (lines 3092-3111 and 3128-3154) is duplicated in the normal and shutdown drain paths. Extract a helper:

```python
def _try_dispatch_notification(evt, sid, session) -> bool:
    """Dispatch a notification event if session is idle. Returns True if consumed."""
    if evt.get("type") == "completion" and process_registry.is_completion_consumed(evt.get("session_id", "")):
        return True
    text = format_process_notification(evt)
    if not text:
        return True
    _emit("status.update", sid, {"kind": "process", "text": text})
    with session["history_lock"]:
        if session.get("running"):
            process_registry.completion_queue.put(evt)
            return False  # not consumed; requeued
        session["running"] = True
    rid = f"__notif__{int(time.time() * 1000)}"
    try:
        _emit("message.start", sid)
        _run_prompt_submit(rid, sid, session, text)
    except Exception:
        with session["history_lock"]:
            session["running"] = False
    return True
```

### Recommendation 4: Consider adding `register_completion_consumed()` method
To `ProcessRegistry`:
```python
def mark_completion_consumed(self, session_id: str) -> None:
    """Explicitly mark a session as consumed (e.g., after TUI notification delivery)."""
    self._completion_consumed.add(session_id)
```
This would let the TUI poller mark sessions consumed after successful delivery, tightening the dedup guard for edge cases where a `poll()` race happens after delivery but before the user sees the notification. Currently this race is handled by the exists-at-all check, but marking consumed explicitly would be cleaner.

### Recommendation 5: Test watch-match delivery in TUI
The existing tests cover only `type: "completion"` events. Add tests for:
- `type: "watch_match"` → triggers agent turn
- `type: "watch_disabled"` → triggers agent turn  
- Multiple events arriving while agent is busy → all re-queued, none lost
- Cancel event (re-queued) when agent later becomes free → delivered

---

## 6. Summary Table

| # | Component | Status | Issue |
|---|-----------|--------|-------|
| 1 | TUI notification poller | ✅ Implemented with tests | `_notification_poller_loop` drains `completion_queue` and injects agent turns |
| 2 | CLI pre-input drain | ❌ BUG | `_format_process_notification` undefined — silently skips notifications |
| 3 | CLI idle/post-turn drain | ✅ Correct | Uses `drain_notifications()` properly |
| 4 | Gateway watcher | ✅ Correct | Independent polling; secondary queue drain filters completions (intentional) |
| 5 | Event re-queueing (TUI) | ⚠️ Minor risk | Ordering not guaranteed; race if multiple poller instances run |
| 6 | Kanban → TUI notification | ❌ Missing | Need new poller thread following `_kanban_drain_loop` pattern |
| 7 | Cursor persistence | ✅ For process events via `_completion_consumed`; ❌ Missing for kanban events in TUI |
