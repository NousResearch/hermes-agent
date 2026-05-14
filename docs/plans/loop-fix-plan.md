# /loop Fix: Complete Implementation Plan

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task.

**Goal:** Fix the `/loop` slash command in the TUI/Gateway path so looped prompts fire reliably every N seconds with a minimum interval of 60s, no popup notifications, and an immediate first tick on command entry.

**Architecture:** Three-layer fix spanning the loop engine (`hermes_cli/loop.py`), CLI handler (`cli.py`), and gateway handler + post-turn hook (`gateway/run.py`). The gateway/TUI path requires a fundamentally different injection mechanism from the CLI path because the TUI has no platform adapter with `_pending_messages`.

**Tech Stack:** Python 3.11, threading (daemon threads), asyncio (gateway event loop), SQLite (SessionDB persistence), Hermes Agent internal APIs.

---

## Full Bug Context & Root Cause Analysis

### The Original Requirement

Three things the `/loop` slash command needed:

1. **Minimum interval of 60 seconds** (1 minute) — no sub-minute loops allowed.
2. **No popup confirmation** — the loop tick should silently enqueue the prompt. The prompt appearing as the next user input IS the validation it's working.
3. **First tick fires immediately** — when the user types `/loop 1m <prompt>`, the prompt should fire right away as the first "tick."

### The Three Bugs Found (In Order of Discovery)

#### Bug 1: Gateway Post-Turn Hook Was a No-Op

**File:** `gateway/run.py`, method `_post_turn_loop_continuation` (line ~9746)

```python
# BEFORE (shipped code):
async def _post_turn_loop_continuation(self, ...):
    """No-op — loop is driven by background scheduler, not post-turn hook."""
    pass  # ← literally nothing!
```

The gateway's loop handler (`_handle_loop_command`) fires the kickoff prompt, the agent responds, then `_post_turn_loop_continuation` runs — and does `pass`. Nothing. The loop silently dies after one tick.

**Fix applied:** Replaced `pass` with real timer logic that calls `load_loop()`, checks `elapsed >= interval_seconds`, updates `last_fired_at`/`turns_completed`, and enqueues through the adapter FIFO.

#### Bug 2: Post-Turn Hook Cannot Drive a Timer

Even with the fix from Bug 1, there's a deeper architectural problem:

```
Timeline:
  t=0:   /loop 1m testing → kickoff fires via _enqueue_fifo
  t=0:   agent responds
  t=5s:  _post_turn_loop_continuation runs → elapsed=5s < 60s → return
  t=6s:  system is IDLE — no user input, no agent turn
         ┌─────────────────────────────────────────┐
         │ NOTHING EVER CALLS THE HOOK AGAIN!      │
         │ The hook only runs during agent turns.   │
         │ No turn = no hook = no tick = dead loop. │
         └─────────────────────────────────────────┘
```

The CLI works because it spawns a `LoopScheduler` daemon thread that ticks every second independently of agent turns. The gateway doesn't start this thread because `mgr.set()` is called without `pending_input`/`is_idle` callbacks.

**Fix applied:** Added a background daemon thread (`_loop_ticker`) inside `_handle_loop_command` that ticks every second, checks the timer, and enqueues via `_enqueue_fifo` — mirroring the CLI's `LoopScheduler` but adapted for the gateway's adapter system.

#### Bug 3: The Two-Copy Trap

**Critical discovery:** Hermes Agent code lives in TWO places on disk:

| Location | Purpose | Loaded at runtime? |
|----------|---------|-------------------|
| `/home/c/Desktop/agenda/hermes-agent/` | Development fork (source of truth) | ❌ No |
| `~/.hermes/hermes-agent/` | Installation copy (pip-installed) | ✅ Yes |

Patches applied only to the fork are **invisible** to the running process. The gateway, TUI, and CLI all load from `~/.hermes/hermes-agent/`. This caused multiple false-negative test cycles — code looked fixed on disk but the old version was still in RAM.

**Prevention:** Created `hermes-agent-editing` skill documenting the sync workflow. All three files must be synced to `~/.hermes/` after every edit cycle.

#### Bug 4: `sid` Scope in Thread Closure

The daemon thread `_loop_ticker` references `sid` in its closure, but `sid` wasn't defined before the thread was created.

**Fix applied:** Added `sid = session_entry.session_id if session_entry else None` before the `if adapter and _quick_key:` guard, and changed the guard to `if adapter and _quick_key and sid:`.

### The Remaining Gap — Bug 5: TUI Has No Platform Adapter

**This is the active bug that prevents `/loop` from working in the TUI even after all above fixes.**

**Root cause:** The gateway's `_enqueue_fifo` method (line 2039-2053) is the injection point used by both the kickoff and the daemon ticker. It works as follows:

```python
def _enqueue_fifo(self, session_key, queued_event, adapter):
    pending_slot = getattr(adapter, "_pending_messages", None)
    if pending_slot is None:
        return  # ← SILENT NO-OP if adapter doesn't have _pending_messages!
    # ... put event in pending_messages dict ...
```

`_pending_messages` is defined in `BasePlatformAdapter` (`gateway/platforms/base.py:1288`). It's populated for platforms like Telegram, Discord, WhatsApp, etc. — all adapters that extend `BasePlatformAdapter`.

**The TUI uses `Platform.LOCAL` (value `"local"`, mapped to `"cli"` in the gateway). There is NO `BasePlatformAdapter` subclass for the TUI.** The gateway does not register an adapter for `Platform.LOCAL` — it's excluded from the platform adapter initialization.

This means:
- `self.adapters.get(event.source.platform)` returns `None` for TUI events
- `_enqueue_fifo` sees `adapter=None` → returns at line 2041
- OR `_enqueue_fifo` sees `pending_slot=None` → returns at line 2045
- Either way: **silent no-op. The kickoff prompt and all subsequent ticks are swallowed.**

**Why the confirmation line still appears:** The `_handle_loop_command` method returns a string (`"⊙ Loop set (60s interval): testing"`) which the gateway delivers as the command response. The loop is *set* in the database — the `LoopManager.set()` call succeeds. The daemon thread starts and ticks. But every enqueue attempt silently fails. The database shows an active loop, but no prompt ever reaches the agent.

---

## Changes Already Applied (Fork → Install Synced)

### 1. `hermes_cli/loop.py` — Three Changes

```python
# Change 1: New constant (line 32)
MIN_INTERVAL_SECONDS = 60       # 1 minute minimum

# Change 2: Interval clamping in LoopManager.set() (lines 227-233)
effective_interval = max(
    int(interval_seconds) if interval_seconds else 300,
    MIN_INTERVAL_SECONDS,
)
state = LoopState(
    prompt=prompt,
    interval_seconds=effective_interval,  # was: interval_seconds=...
    ...
)

# Change 3: Removed [Loop check] prefix and on_message popup from tick (line 396)
prompt = state.prompt  # was: f"[Loop check] {state.prompt}"
# Removed entire on_message callback block (lines 398-405 in old code)
```

### 2. `cli.py` — One Change

```python
# In _handle_loop_command (line ~7310):
self._pending_input.put(state.prompt)  # was: f"[Loop check] {state.prompt}"
```

### 3. `gateway/run.py` — Two Changes

**Change A: `_post_turn_loop_continuation` rewritten (line ~9746)**

The old `pass` was replaced with a full timer check that:
- Imports `load_loop`/`save_loop`
- Gets `sid` from `session_entry`
- Loads loop state, checks `status == "active"`
- Checks `elapsed >= interval_seconds`
- If ready: updates `last_fired_at`/`turns_completed`, saves, enqueues kickoff event

**Change B: Daemon thread + immediate kickoff in `_handle_loop_command` (line ~10532)**

After `mgr.set()`:
- Defines `sid = session_entry.session_id`
- Starts a daemon `_loop_ticker` thread (identical logic to CLI's `LoopScheduler`)
- Fires immediate kickoff via `_enqueue_fifo`

The post-turn hook acts as a **backup** for the daemon thread, not the primary driver.

**⚠️ Both Change A and Change B use `_enqueue_fifo` — which is the broken path for TUI.**

---

## Required Fix: TUI-Compatible Loop Injection

### Architecture Decision

The gateway's `_handle_loop_command` runs in response to a slash command from the TUI. After it returns the confirmation string, the gateway loop is free. To inject a synthetic "user message" (the loop prompt) into the TUI session's processing pipeline, we need to bypass `_enqueue_fifo` and directly schedule message dispatch on the gateway's asyncio event loop.

### Two Approaches Considered

| Approach | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **A: Direct Agent Dispatch** | `asyncio.run_coroutine_threadsafe(_handle_message_with_agent(...), loop)` | Clean, no adapter dependency | Daemon thread needs access to event loop ref |
| **B: Synthetic MessageEvent via Session** | Create `MessageEvent`, inject into session's pending queue | Reuses existing gateway infrastructure | Still needs adapter or different queue |

**Recommended: Approach A** — it's the simplest correct path that doesn't pretend the TUI has a platform adapter.

### Implementation Plan

#### Task 1: Add helper method `_dispatch_loop_prompt` to GatewayRunner

**Objective:** Create a reusable method that dispatches a loop prompt as a synthetic user message directly through the agent, bypassing adapter/`_enqueue_fifo`.

**File:** Modify `~/.hermes/hermes-agent/gateway/run.py`

**Step 1: Add the method to GatewayRunner class**

Insert after `_enqueue_fifo` (around line 2054) or near `_handle_loop_command`:

```python
def _dispatch_loop_prompt(
    self,
    prompt: str,
    source: "SessionSource",
) -> None:
    """Dispatch a loop prompt as a synthetic user message.

    Schedules the prompt for agent processing on the gateway's asyncio
    event loop.  This bypasses ``_enqueue_fifo`` because the TUI and
    other non-adapter platforms don't have ``_pending_messages``.

    Called from the daemon ticker thread (non-async context), so uses
    ``call_soon_threadsafe`` to hand off to the event loop.
    """
    loop = getattr(self, "_event_loop", None)
    if loop is None:
        logger.debug("loop dispatch: no event loop available")
        return

    async def _run():
        try:
            event = MessageEvent(
                text=prompt,
                message_type=MessageType.TEXT,
                source=source,
                message_id=None,
                channel_prompt=None,
            )
            await self._handle_message_with_agent(event, source)
        except Exception as exc:
            logger.debug("loop dispatch: agent run failed: %s", exc)

    try:
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_run()))
    except Exception as exc:
        logger.debug("loop dispatch: schedule failed: %s", exc)
```

**Step 2: Verify imports are available**

Ensure these are imported at the top of `gateway/run.py` (they should already be):
```python
import asyncio
import threading
```

If `MessageEvent` is not directly imported, it's available as a type annotation — verify it's accessible in method context.

**Step 3: Commit**

```bash
git add gateway/run.py
git commit -m "feat: add _dispatch_loop_prompt for adapter-free loop injection"
```

#### Task 2: Update daemon ticker to use `_dispatch_loop_prompt`

**Objective:** Replace the `_enqueue_fifo` call in the daemon ticker with `_dispatch_loop_prompt`.

**File:** Modify `~/.hermes/hermes-agent/gateway/run.py`

**Step 1: Update the `_loop_ticker` closure in `_handle_loop_command`**

Find the daemon thread definition (around line 10549). Replace:

```python
# OLD (broken — _enqueue_fifo silently fails for TUI):
try:
    ke = MessageEvent(
        text=s.prompt,
        message_type=MessageType.TEXT,
        source=event.source,
        message_id=None,
        channel_prompt=None,
    )
    self._enqueue_fifo(_quick_key, ke, adapter)
except Exception:
    pass
```

With:

```python
# NEW (works for all platforms):
try:
    self._dispatch_loop_prompt(
        prompt=s.prompt,
        source=event.source,
    )
except Exception:
    pass
```

**Step 2: Update the immediate kickoff**

In the same method, find the kickoff block (around line 10582). Replace:

```python
# OLD:
kickoff_event = MessageEvent(
    text=prompt,
    message_type=MessageType.TEXT,
    source=event.source,
    message_id=None,
    channel_prompt=None,
)
self._enqueue_fifo(_quick_key, kickoff_event, adapter)
```

With:

```python
# NEW:
self._dispatch_loop_prompt(
    prompt=prompt,
    source=event.source,
)
```

**Step 3: Remove unused adapter/quick_key references from loop set block**

The `adapter`, `_quick_key`, and `sid` variables are now only needed for the guard check and thread naming. The guard can be simplified to `if sid:`:

```python
sid = session_entry.session_id if session_entry else None
if sid:
    from hermes_cli.loop import load_loop, save_loop
    import time as _t
    import threading

    def _loop_ticker():
        """Background daemon that ticks every second."""
        while True:
            _t.sleep(1)
            s = load_loop(sid)
            if s is None or s.status != "active":
                break
            now = _t.time()
            elapsed = now - s.last_fired_at
            if s.last_fired_at > 0 and elapsed < s.interval_seconds:
                continue
            s.last_fired_at = now
            s.turns_completed += 1
            save_loop(sid, s)
            try:
                self._dispatch_loop_prompt(
                    prompt=s.prompt,
                    source=event.source,
                )
            except Exception:
                pass

    threading.Thread(
        target=_loop_ticker,
        name=f"loop-ticker-{sid[:8]}",
        daemon=True,
    ).start()

    # Launch the first tick immediately
    try:
        self._dispatch_loop_prompt(
            prompt=prompt,
            source=event.source,
        )
    except Exception as exc:
        logger.debug("loop kickoff dispatch failed: %s", exc)
```

**Step 4: Commit**

```bash
git add gateway/run.py
git commit -m "fix: use _dispatch_loop_prompt in daemon ticker and kickoff"
```

#### Task 3: Update `_post_turn_loop_continuation` to use `_dispatch_loop_prompt`

**Objective:** Fix the post-turn hook's enqueue path to also use `_dispatch_loop_prompt` for consistency and reliability.

**File:** Modify `~/.hermes/hermes-agent/gateway/run.py`

**Step 1: Replace the enqueue block in `_post_turn_loop_continuation`**

Find the post-turn hook (around line 9746). Replace the adapter/enqueue block:

```python
# OLD:
adapter = (
    self.adapters.get(source.platform)
    if source else None
)
_quick_key = (
    self._session_key_for_source(source)
    if source else None
)
if adapter and _quick_key:
    try:
        kickoff_event = MessageEvent(
            text=state.prompt,
            message_type=MessageType.TEXT,
            source=source,
            message_id=None,
            channel_prompt=None,
        )
        self._enqueue_fifo(_quick_key, kickoff_event, adapter)
    except Exception as exc:
        logger.debug("loop continuation enqueue failed: %s", exc)
```

With:

```python
# NEW:
try:
    self._dispatch_loop_prompt(
        prompt=state.prompt,
        source=source,
    )
except Exception as exc:
    logger.debug("loop continuation dispatch failed: %s", exc)
```

**Step 2: Commit**

```bash
git add gateway/run.py
git commit -m "fix: use _dispatch_loop_prompt in post-turn loop continuation"
```

#### Task 4: Sync all three files to the install location

**Objective:** Copy the fork's modified files to `~/.hermes/hermes-agent/` so the running process picks up all fixes.

**Step 1: Sync files**

```bash
cp /home/c/Desktop/agenda/hermes-agent/hermes_cli/loop.py \
   ~/.hermes/hermes-agent/hermes_cli/loop.py

cp /home/c/Desktop/agenda/hermes-agent/cli.py \
   ~/.hermes/hermes-agent/cli.py

cp /home/c/Desktop/agenda/hermes-agent/gateway/run.py \
   ~/.hermes/hermes-agent/gateway/run.py
```

**Step 2: Verify sync**

```bash
diff /home/c/Desktop/agenda/hermes-agent/hermes_cli/loop.py \
     ~/.hermes/hermes-agent/hermes_cli/loop.py \
     && echo "loop.py: synced ✓" || echo "loop.py: MISMATCH ✗"

diff /home/c/Desktop/agenda/hermes-agent/cli.py \
     ~/.hermes/hermes-agent/cli.py \
     && echo "cli.py: synced ✓" || echo "cli.py: MISMATCH ✗"

diff /home/c/Desktop/agenda/hermes-agent/gateway/run.py \
     ~/.hermes/hermes-agent/gateway/run.py \
     && echo "gateway/run.py: synced ✓" || echo "gateway/run.py: MISMATCH ✗"
```

Expected output: all three files show `synced ✓`.

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: sync all /loop fixes from fork to install"
```

#### Task 5: Restart and verify

**Objective:** Restart the gateway and TUI, then test `/loop 1m testing` end-to-end.

**Step 1: Stop running gateway and TUI**

```bash
# Find and kill any running hermes gateway processes
pkill -f "hermes.*gateway" 2>/dev/null
pkill -f "hermes.*tui" 2>/dev/null

# Verify nothing is running
pgrep -f "hermes.*gateway" && echo "Gateway still running!" || echo "Gateway stopped ✓"
pgrep -f "hermes.*tui" && echo "TUI still running!" || echo "TUI stopped ✓"
```

**Step 2: Start gateway**

In one terminal/pane:
```bash
hermes gateway
```

**Step 3: Start TUI**

In another terminal/pane:
```bash
hermes --tui
```

**Step 4: Run the test**

In the TUI, type:
```
/loop 1m testing
```

**Step 5: Verify immediate kickoff**

The confirmation line `⊙ Loop set: every 60s → testing` should appear. Wait for the agent to process `testing` as if you typed it. This should happen within seconds — the kickoff fires immediately.

**Step 6: Verify second tick at 60s**

After the agent responds to the first `testing`, wait 60 seconds. A new `testing` prompt should appear automatically — no popup, no `[Loop check]` prefix, just the prompt.

**Step 7: Verify continuous ticking**

Wait another 60 seconds. A third `testing` should appear. If all three ticks land, the fix is complete.

**Step 8: Verify minimum interval enforcement**

Type:
```
/loop 10s quick test
```

The confirmation should show `every 60s` (not 10s) — proving the 60-second clamp is working.

**Step 9: Clean up**

```bash
/loop clear
```

---

## Verification Checklist

| # | What to verify | Expected result | How to check |
|---|---------------|-----------------|--------------|
| 1 | Minimum interval enforced | `/loop 10s ping` → confirms "every 60s" | Confirmation line shows 60s |
| 2 | No `[Loop check]` prefix | Prompt appears raw | No `[Loop check]` in agent log or TUI |
| 3 | No popup notification | No `↻ Loop check (N): ...` | No on_message output |
| 4 | First tick immediate | Prompt fires within seconds of `/loop` | Agent processes prompt right after confirmation |
| 5 | Second tick at interval | Prompt fires at t+60s | Wait 60s, prompt appears |
| 6 | Continuous ticking | Third tick at t+120s | Wait another 60s, prompt appears |
| 7 | `/loop clear` works | No more ticks | Wait 60s after clear, nothing appears |
| 8 | Files synced | Fork == install | `diff` shows no differences |
| 9 | No crashes in logs | gateway.log and agent.log clean | Check logs after test |

---

## Files Summary

| File | Changes | Status |
|------|---------|--------|
| `hermes_cli/loop.py` | MIN_INTERVAL_SECONDS, interval clamp, remove [Loop check], remove on_message | ✅ Applied & synced |
| `cli.py` | Remove [Loop check] from kickoff | ✅ Applied & synced |
| `gateway/run.py` | `_post_turn_loop_continuation` rewrite, daemon ticker, `_dispatch_loop_prompt` | ⚠️ Tasks 1-3 needed |
| `~/.hermes/skills/software-development/hermes-agent-editing/SKILL.md` | Two-copy trap documentation | ✅ Created |

---

## Edge Cases & Notes

- **Gateway restart required after every code sync.** Gateway loads Python modules at startup; file changes on disk are invisible until restart.
- **Only one loop per session.** The `LoopManager` is keyed by `session_id`. Setting a new loop overwrites the previous one.
- **Loop state persists across restarts.** The state is stored in SessionDB's `state_meta` table. After a gateway restart, the loop is still marked active but the daemon thread is gone — the user needs to `/loop resume` or re-set.
- **Gateway event loop must be accessible.** The `_event_loop` attribute on `GatewayRunner` is set during initialization. Verify it's available before `_dispatch_loop_prompt` is called (the guard `if loop is None: return` handles the safety case).
- **The `_post_turn_loop_continuation` hook serves as a backup.** If the daemon thread crashes or is delayed, the post-turn hook ensures the loop prompt eventually fires. Both mechanisms should use `_dispatch_loop_prompt`.
- **`call_soon_threadsafe` is necessary.** The daemon ticker runs in a non-asyncio thread. All asyncio event loop operations must be scheduled via `call_soon_threadsafe` to avoid race conditions.
