# PR Spec: Automatic Session Resume After Gateway Restart

**Date:** 2026-04-17

> **Intent:** Fix the terrible current UX where Hermes says an interrupted task will resume after restart, but a forced/interrupted restart often converts that thread into a fresh session and tells the user to `/resume` manually.

## TL;DR

Hermes should treat **restart interruption** as a **resumable session state**, not as a **session reset**.

Today, when gateway shutdown cannot drain active work within `agent.restart_drain_timeout`, startup falls back to `suspend_recently_active()`. That marks recently-active sessions as `suspended`, and the next message in the same thread causes `SessionStore.get_or_create_session()` to create a **new session ID** with `auto_reset_reason="suspended"`. The user sees:

- shutdown banner: "Send any message after restart to resume where it left off."
- then next-turn banner: "Session automatically reset (previous session was stopped or interrupted). Use /resume..."

That is the exact wrong behavior for the common case of **same thread, same user, same restart, still wants same task**.

**Recommendation:** introduce a distinct persisted state like `resume_pending` / `interrupted_by_restart` and keep the existing `session_id` on the next message with the same `session_key`. Reuse the existing transcript reload and auto-continue logic in `gateway/run.py` instead of creating a new session. Escalation should reuse the existing `.restart_failure_counts` / stuck-loop detection path rather than adding a parallel counter on `SessionEntry`.

---

## Problem statement

### Current user experience

Current behavior is incoherent:

1. Hermes sends an optimistic restart notice.
2. Gateway restart times out draining active agents.
3. Startup suspends recently-active sessions.
4. The user's next message on the same `session_key` lands in a fresh session.
5. Hermes tells the user to browse `/resume` manually.

This is a terrible experience because:

- the product promises one behavior and delivers the opposite,
- the user is forced to understand internal session mechanics,
- the resume path is thread-local and obvious to the system but not automatic,
- the current fallback destroys continuity even though transcript history still exists.

### Root cause in current code

Relevant current behavior:

- shutdown banner text in `gateway/run.py`
  - `_notify_active_sessions_of_shutdown()`
  - says: `Send any message after restart to resume where it left off.`
- forced-interrupt path in `gateway/run.py`
  - if drain times out, gateway interrupts active agents
  - skips `.clean_shutdown` marker so next startup treats the prior run as unsafe
- startup recovery in `gateway/run.py`
  - calls `self.session_store.suspend_recently_active()` when `.clean_shutdown` is absent
- session reset behavior in `gateway/session.py`
  - `get_or_create_session()` checks `entry.suspended`
  - suspended sessions are turned into a **new session ID** with `auto_reset_reason="suspended"`
- reset notice in `gateway/run.py`
  - emits: `Session automatically reset (previous session was stopped or interrupted)...`
- transcript continuation logic already exists in `gateway/run.py`
  - if loaded history ends with a `tool` message, Hermes prepends a system note telling the model to finish the interrupted work

**Important observation:** Hermes already has part of the resume mechanism. The main thing preventing automatic resume is that forced restart currently turns the session into a fresh session instead of preserving the old one.

---

## Product goal

When a user restarts Hermes and then sends the next message that resolves to the **same `session_key`** (same chat/thread/topic lane), Hermes should, by default:

1. preserve the same conversation lane,
2. preserve the same `session_id`,
3. reload the same transcript,
4. inform the model that the previous turn was interrupted by restart,
5. continue/resume automatically,
6. avoid making the user manually browse `/resume` unless recovery has actually failed.

### Desired UX

For the normal case:

- user starts a long task in thread X
- gateway restarts
- user returns to the same lane and says anything
- Hermes continues from the interrupted session on that same `session_key`

For the pathological case:

- same session repeatedly hangs across multiple restarts
- Hermes eventually abandons auto-resume for that session and gives the user a clean slate

---

## Non-goals

This PR should **not** try to:

- implement fully autonomous resume with **no** user follow-up message,
- merge different threads/chats/topics into one session,
- auto-resume across a different thread than the one that was interrupted,
- invent a generic distributed job recovery layer,
- remove existing stuck-loop safety mechanisms entirely,
- change normal idle/daily `session_reset` policy semantics.

This is specifically about **same-lane restart continuity** after gateway interruption.

---

## Design principles

1. **Restart interruption is not the same as intentional reset.**
2. **Same thread should keep same session unless proven unsafe.**
3. **Safety escalation should be progressive, not immediate.**
4. **User-visible messages must describe the actual recovery semantics.**
5. **Reuse existing transcript + auto-continue machinery instead of inventing new prompt plumbing.**

---

## Recommendation

## Introduce a resumable restart-interruption state

Add a persisted session state that is distinct from `suspended`.

### New state

Recommended `SessionEntry` fields in `gateway/session.py`:

```python
resume_pending: bool = False
resume_reason: Optional[str] = None  # e.g. "restart_timeout", "crash_recovery"
last_resume_marked_at: Optional[datetime] = None
```

### Meaning of states

- `suspended=True`
  - do **not** resume automatically
  - next access should create a fresh session
  - used for known-poisoned sessions / explicit stuck-loop protection
- `resume_pending=True`
  - user should stay on the same session
  - next access should preserve the existing session ID
  - used when a restart interrupted in-flight work but we still expect same-thread continuation to succeed

This is the key architectural distinction missing today.

---

## High-level behavior change

### Current behavior

```text
restart timed out
  -> skip .clean_shutdown
  -> startup: suspend_recently_active()
  -> session.suspended = True
  -> next message => new session_id
  -> user gets reset notice + /resume guidance
```

### Proposed behavior

```text
restart timed out
  -> mark active session(s) as resume_pending=True
  -> next startup preserves mapping
  -> next message on the same `session_key` returns existing session entry
  -> transcript reloads from same session_id
  -> model gets interruption note
  -> Hermes continues automatically
```

### Escalation path

```text
restart timed out repeatedly for same session
  -> increment existing .restart_failure_counts counter
  -> _suspend_stuck_loop_sessions() suspends once threshold is exceeded
  -> only then force fresh-session fallback
```

---

## Detailed design

## 1) Persist `resume_pending` on interrupted restart

### Where to mark it

During shutdown in `gateway/run.py`, after drain timeout is detected and the gateway force-interrupts active agents, mark the active session keys as `resume_pending=True`.

This should happen **instead of** relying on the startup-wide "recently active means suspend" fallback for these sessions.

### Why here

At shutdown time, the gateway knows:

- which sessions were actually running,
- that the interruption came from restart/shutdown,
- that this is not an idle/daily reset,
- that the user did not ask for `/new`.

That is the correct moment to record resumable interruption state.

### Proposed helper

Add a method to `SessionStore` in `gateway/session.py`:

```python
def mark_resume_pending(
    self,
    session_key: str,
    *,
    reason: str = "restart_timeout",
) -> bool:
    ...
```

Responsibilities:

- set `resume_pending=True`
- set `resume_reason`
- set `last_resume_marked_at`
- persist metadata in `sessions.json`

---

## 2) Do not auto-reset `resume_pending` sessions on next access

### Current bad behavior

`SessionStore.get_or_create_session()` currently treats `suspended` as "auto-reset on next access".

### Proposed behavior

Extend `get_or_create_session()` logic:

- if `entry.suspended` → current reset behavior stays
- if `entry.resume_pending` → **return the existing entry** while the recovery window is still fresh, and only clear the marker after a successful turn completes

Pseudo-shape:

```python
if entry.suspended:
    reset_reason = "suspended"
elif entry.resume_pending:
    entry.updated_at = now
    self._save()
    return entry
else:
    reset_reason = self._should_reset(entry, source)
```

This is the core functional fix.

---

## 3) Reuse existing transcript reload and auto-continue logic

This PR should explicitly lean on behavior that already exists.

### Existing asset

`gateway/run.py` already prepends an interruption note when the loaded history ends in a `tool` result:

- if transcript ends with role `tool`, Hermes tells the model to finish processing interrupted tool results before addressing the new user message.

### Extend the system note behavior

If `session_entry.resume_pending` is set, prepend a stronger note such as:

> `[System note: Your previous turn in this same session was interrupted by a gateway restart. Continue from the existing transcript. If there are unfinished tool results, process them first, summarize what was accomplished, then answer the user's new message.]`

This should work whether the transcript ended with:

- a `tool` message,
- an interrupted assistant turn,
- or a partially completed tool-heavy exchange.

### Why this is enough for v1

We do **not** need a brand-new recovery engine for the first version.

Preserving the same session ID plus transcript reload plus better interruption note gets the common case back to a sane product experience.

---

## 4) Keep stuck-loop protection, but reuse the existing restart-failure mechanism

We should not regress the original safety intent behind the stuck-loop work.

### Proposed rule

- first interrupted restart for a session → auto-resume
- second interrupted restart for the same `session_key` → still auto-resume
- third interrupted restart for the same `session_key` → let the existing stuck-loop path suspend it

This keeps safety without making the default path destructive.

### Recommended implementation

Reuse the existing gateway-level `.restart_failure_counts` file and `_suspend_stuck_loop_sessions()` flow:

- shutdown-time drain timeout still calls `mark_resume_pending(...)` for the interrupted `session_key`s
- successful turn completion clears the restart-failure count for that `session_key`
- repeated interrupted restarts are counted in `.restart_failure_counts`
- `_suspend_stuck_loop_sessions()` flips the session to `suspended=True` once the existing threshold is exceeded

Do **not** add or maintain a parallel `resume_attempts` counter on `SessionEntry`.

---

## 5) Narrow the role of `suspend_recently_active()`

`suspend_recently_active()` is too blunt as the generic fallback for restart interruption.

### Current role

It treats "recently active at startup after unclean shutdown" as a reason to force clean-slate behavior.

### Proposed role after this PR

Reserve it for narrower cases, such as:

- startup crash recovery when explicit `resume_pending` metadata is absent,
- legacy upgrade path / backward compatibility,
- emergency fallback for clearly unsafe sessions.

Normal interrupted-restart recovery with explicit `resume_pending` metadata should not be suspended by this helper.

### Important outcome

This means **restart interruption should no longer immediately flow through the same code path as 'known stuck session'.**

That separation is the real product fix.

---

## 6) Fix user-facing messaging

### Current messaging is misleading

#### Shutdown banner
Current wording:

> `Send any message after restart to resume where it left off.`

This is too absolute.

#### Reset notice
Current wording points users to `session_reset` config even when the real cause is startup suspension after interrupted restart.

### Proposed messaging

#### Shutdown banner
For resumable restart:

- non-threaded chat:
  - `Gateway restarting — I'll try to resume this session after restart. Send a message in this chat to continue.`
- threaded/topic chat:
  - `Gateway restarting — I'll try to resume this session after restart. Send a message in this thread/topic to continue.`

#### If the system escalates to forced clean slate
Only after repeated failure should the user see something like:

- `This session was interrupted repeatedly during restart recovery, so Hermes started a fresh session to avoid getting stuck. Use /resume if you want the old transcript.`

### Message principle

Only mention `/resume` when Hermes has actually decided **not** to auto-resume.

---

## State machine

```text
ACTIVE
  -> clean restart/shutdown drains successfully
     -> ACTIVE (same session preserved)

ACTIVE
  -> restart/crash interrupts in-flight work
     -> RESUME_PENDING

RESUME_PENDING
  -> next message in same thread/topic
     -> ACTIVE (same session_id, transcript reloaded)

RESUME_PENDING
  -> repeated interrupted restarts exceed threshold
     -> SUSPENDED

SUSPENDED
  -> next message
     -> NEW_SESSION (fresh session_id, old transcript still available via /resume)
```

---

## File-level implementation plan

## Primary files

### `gateway/session.py`

Add persisted session fields and APIs:

- new `SessionEntry` fields:
  - `resume_pending`
  - `resume_reason`
  - `last_resume_marked_at`
- serialization/deserialization support in `to_dict()` / `from_dict()`
- helper methods:
  - `mark_resume_pending(...)`
  - `clear_resume_pending(...)`
- update `get_or_create_session()` so `resume_pending` returns existing session instead of resetting

### `gateway/run.py`

Update gateway behavior:

- after drain timeout, mark active sessions `resume_pending=True`
- on resumed turn, inject restart-interruption system note when `session_entry.resume_pending`
- clear `resume_pending` only after a successful resumed turn completes
- update shutdown banner wording to promise attempted recovery, not guaranteed recovery
- stop routing normal interrupted restart recovery through immediate clean-slate semantics

### `tests/gateway/`

Add or update tests for:

- same-session resume after interrupted restart
- transcript preserved across restart timeout
- tool-result auto-continue still works on resumed session
- repeated recovery failure escalates through `.restart_failure_counts` / `_suspend_stuck_loop_sessions()`
- shutdown banner wording is no longer misleading
- `/resume` guidance only appears when clean-slate fallback actually occurs

---

## Test plan

## Unit tests

### `tests/gateway/test_restart_resume_pending.py` (new)

Suggested cases:

1. **mark resume pending persists state**
   - create session
   - call `mark_resume_pending()`
   - reload store
   - assert flags persisted

2. **resume_pending does not create new session id**
   - create session
   - mark `resume_pending=True`
   - call `get_or_create_session()`
   - assert returned `session_id` is unchanged

3. **suspended still creates new session id**
   - existing current behavior regression guard

4. **clear resume pending after success**
   - mark resumable
   - simulate successful turn start or completion
   - assert `resume_pending=False`

### `tests/gateway/test_restart_recovery_flow.py` (new)

Suggested cases:

1. **interrupted restart on same session key resumes existing session**
   - transcript exists under original session id
   - next message on the same `session_key` loads the same transcript
   - no auto-reset notice

2. **tool-tail transcript triggers auto-continue note on resumed session**
   - transcript ends with `tool`
   - resumed run prepends correct system note

3. **repeated restart failures escalate to suspended**
   - simulate threshold crossings
   - assert fresh-session fallback only after threshold

4. **clean restart remains unchanged**
   - `.clean_shutdown` path still preserves session as before

### Message-copy tests

Add assertions for the updated shutdown banner and fallback text.

---

## Backward compatibility

This change should be backward-compatible with existing stored sessions.

### Migration behavior

- old `sessions.json` entries simply deserialize with default values for new fields
- no database migration should be required if session metadata remains file-backed
- existing `suspended=True` entries should keep current semantics

### Important safety note

Do **not** silently reinterpret existing `suspended=True` as resumable. That would change meaning for users who explicitly relied on the current clean-slate escape hatch.

---

## Risks

### 1. Resume loop risk
If resume is attempted too aggressively, a truly poisoned session could keep re-entering the same bad state.

**Mitigation:** keep thresholded escalation in the existing `.restart_failure_counts` / `_suspend_stuck_loop_sessions()` flow.

### 2. Partial transcript ambiguity
If a turn was interrupted mid-assistant generation, the last messages may not be perfectly shaped.

**Mitigation:** keep the recovery note explicit and rely on existing transcript loading behavior. Tests should cover common partial-tail shapes.

### 3. Messaging confusion during rollout
If message copy changes before semantics change, UX could still be misleading.

**Mitigation:** land copy changes in the same PR as behavior changes.

---

## Open questions

1. `resume_pending` should clear only after a successful completed turn, not at turn start.
2. Recovery should only be attempted on the same `session_key`. Do not cross lanes.
3. `suspend_recently_active()` should remain only as a narrower crash-recovery fallback and should not suspend explicit `resume_pending` sessions.
4. User-facing recovery should stay as an internal system note in v1 unless debugging is enabled.

---

## Recommended implementation order

1. Add `SessionEntry` resume-pending fields + serialization
2. Add `SessionStore.mark_resume_pending()` / `clear_resume_pending()`
3. Change `get_or_create_session()` to preserve same session for `resume_pending`
4. Mark active sessions resume-pending on drain-timeout shutdown
5. Inject restart-resume system note in `gateway/run.py`
6. Clear pending state after successful completion
7. Reuse existing `.restart_failure_counts` / `_suspend_stuck_loop_sessions()` escalation
8. Update shutdown/fallback copy
9. Add regression tests

---

## Success criteria

This PR is successful if all of the following are true:

1. A long-running task interrupted by gateway restart in a Discord/Telegram lane resumes automatically on the next message with the same `session_key`.
2. The resumed thread keeps the same `session_id` and transcript history.
3. Hermes no longer tells the user to `/resume` for the normal restart-recovery case.
4. Existing clean restart behavior is preserved.
5. Truly stuck sessions still have a safety escape hatch after repeated failures.
6. User-facing restart copy accurately describes attempted recovery rather than promising impossible behavior.

---

## Strong opinion

Hermes should **default to continuity** after restart and only fall back to clean-slate reset when recovery is actually failing.

Right now the system is optimized for protecting itself from stuck loops at the cost of making ordinary restart recovery feel broken. That tradeoff is backwards for user experience.

The correct product stance is:

- **same session_key + immediate post-restart message = continue automatically**
- **repeated failed recovery = fresh session as safety fallback**

That gets the common case right without giving up the escape hatch.
