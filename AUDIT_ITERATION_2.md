# Deep Audit: Async Error Handling and Agent Loop Reliability - Iteration #2

## Summary
This audit reveals **5 critical bugs** in core async handling, context compression, and cron job scheduling that can cause data loss, double execution, and incorrect conversation flow.

---

## 🔴 CRITICAL BUG #1: Role Violation After Context Compression
**File:** `/Users/enyuanzhang/Desktop/github/hermes-agent/agent/context_compressor.py`
**Lines:** 694-728 (compress method, summary role selection)
**Severity:** HIGH - Data Loss / Model Confusion

### The Bug
When a context compression creates a summary message, the compressor tries to avoid consecutive same-role messages by choosing a role that doesn't conflict with the head and tail:

```python
# Lines 700-710
if last_head_role in ("assistant", "tool"):
    summary_role = "user"
else:
    summary_role = "assistant"
# If the chosen role collides with the tail AND flipping wouldn't
# collide with the head, flip it.
if summary_role == first_tail_role:
    flipped = "assistant" if summary_role == "user" else "user"
    if flipped != last_head_role:
        summary_role = flipped
    else:
        # Both roles would create consecutive same-role messages
        # (e.g. head=assistant, tail=user — neither role works).
        # Merge the summary into the first tail message instead
        _merge_summary_into_tail = True
```

**The Problem:**
- When `_merge_summary_into_tail = True`, the code appends the summary content to the first tail message
- **But it doesn't validate the first tail message's role**
- If the first tail message is a "tool" result (which is common), prepending summary text to it violates the OpenAI API specification
- Tool messages should only contain the tool result; they cannot contain model narrative text

### Example Scenario
```
compress_end points to message[N], which is a "tool" result with tool_call_id
_merge_summary_into_tail = True (because both roles would violate alternation)
Line 726: msg["content"] = summary + "\n\n" + original
→ INVALID: Tool message now contains narrative + tool result
→ API rejects with "Invalid tool message format"
```

### Consequence
- Conversation crashes at the next API call
- Summary is lost
- Cannot recover the session without manual DB manipulation

### Fix
Before merging into tail, validate that the first tail message is not a "tool" role:
```python
if _merge_summary_into_tail and i == compress_end:
    # VALIDATION: Only merge into user/assistant, not tool results
    if msg.get("role") != "tool":
        original = msg.get("content") or ""
        msg["content"] = summary + "\n\n" + original
        _merge_summary_into_tail = False
    else:
        # First tail is a tool result — force a role that doesn't collide
        compressed.append({"role": "user", "content": summary})
        _merge_summary_into_tail = False
```

---

## 🔴 CRITICAL BUG #2: Double-Execution Race Condition in Cron Scheduler
**File:** `/Users/enyuanzhang/Desktop/github/hermes-agent/cron/scheduler.py`
**Lines:** 843-892 (tick function)
**Severity:** CRITICAL - Job Double-Execution

### The Bug
The cron scheduler has a time-of-check-time-of-use (TOCTOU) race condition:

```python
# Lines 843-860
try:
    due_jobs = get_due_jobs()  # ← Get jobs where next_run_at <= now
    
    # ... logging ...
    
    executed = 0
    for job in due_jobs:
        try:
            # Lines 856-860: For recurring jobs, ADVANCE next_run_at BEFORE execution
            advance_next_run(job["id"])
            
            success, output, final_response, error = run_job(job)  # ← SLOW: Can take minutes
            # ... (900+ lines of execution, delivery, etc.)
            mark_job_run(job["id"], success, error, delivery_error=delivery_error)
```

**The Problem:**
1. `get_due_jobs()` returns jobs where `next_run_at <= now` (e.g., 10:00:00)
2. `advance_next_run(job["id"])` updates the DB to the next occurrence (e.g., 10:10:00)
3. **Then** `run_job(job)` starts executing (can take seconds or minutes)
4. Meanwhile, **another concurrent tick (gateway + daemon, or manual trigger) runs**
5. That new tick calls `get_due_jobs()` again
6. Since `run_job()` is still executing, the original job record is not yet marked as "run"
7. **The same job is fetched again and executed twice**

### Why the Lock Doesn't Help
The file-based lock (lines 832-841) only protects the `tick()` function itself:
```python
lock_fd = open(_LOCK_FILE, "w")
fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # ← Exclusive lock acquired
```

**But the lock is RELEASED before `run_job()` completes:**
```python
finally:
    fcntl.flock(lock_fd, fcntl.LOCK_UN)  # ← Lock released while jobs still executing
    lock_fd.close()
```

The next tick (from gateway timer, daemon, or manual invocation) can now acquire the lock, call `get_due_jobs()`, and fetch the same job again.

### Consequence
- Scheduled reports delivered twice
- Duplicate data collected
- Duplicate messages sent to users
- Database inconsistency if job updates are applied twice

### Fix
Keep the lock held for the entire job execution:

```python
def tick(verbose: bool = True, adapters=None, loop=None) -> int:
    # ... setup ...
    try:
        lock_fd = open(_LOCK_FILE, "w")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # ... handle lock acquisition failure ...
        
        due_jobs = get_due_jobs()
        executed = 0
        
        for job in due_jobs:
            try:
                advance_next_run(job["id"])
                success, output, final_response, error = run_job(job)
                # ... delivery and marking ...
            except Exception as e:
                logger.error("Error processing job %s: %s", job['id'], e)
                mark_job_run(job["id"], False, str(e))
        
        return executed
    finally:
        # ← Keep lock held until ALL jobs complete
        if lock_fd:
            if fcntl:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            lock_fd.close()
```

**Alternative: Use DB-level locking**
Instead of releasing the file lock, implement a "tick_in_progress" flag in the jobs table that persists across tick() calls.

---

## 🔴 CRITICAL BUG #3: Unhandled Context Compression Exception in Main Loop
**File:** `/Users/enyuanzhang/Desktop/github/hermes-agent/run_agent.py`
**Lines:** 8200-8210, 8260-8270, 8335-8345 (three callsites)
**Severity:** HIGH - Silent Failure / Unhandled Exception

### The Bug
During the main conversation loop, three places call `_compress_context()` without proper exception handling:

```python
# Line 8204 (in _interruptible_api_call, post-4xx recovery)
messages, active_system_prompt = self._compress_context(
    messages, system_message, approx_tokens=_preflight_tokens,
    task_id=effective_task_id,
)
# ← If _compress_context raises, exception propagates and crashes the loop

# Line 8262 (in main loop, after context length error)
messages, active_system_prompt = self._compress_context(
    messages, system_message, approx_tokens=_preflight_tokens,
    task_id=effective_task_id,
)
# ← Same issue

# Line 8338 (in main loop, after invalid_request error)
messages, active_system_prompt = self._compress_context(
    messages, system_message, approx_tokens=_preflight_tokens,
    task_id=effective_task_id,
)
# ← Same issue
```

**Context:** `_compress_context()` can raise exceptions from:
- `self.context_compressor.compress()` - Summarizer API call failures
- `self._session_db` calls - SQLite errors
- `self._build_system_prompt()` - System prompt generation failures

### Consequence
- If compression fails, the entire API loop crashes
- User loses partial work
- No graceful degradation (e.g., dropping the middle turns without summary)
- Exception is logged but not recovered

### Fix
Wrap each call in try-except and gracefully degrade:

```python
try:
    messages, active_system_prompt = self._compress_context(
        messages, system_message, approx_tokens=_preflight_tokens,
        task_id=effective_task_id,
    )
except Exception as compress_err:
    logger.error("Context compression failed: %s — continuing without compression", compress_err)
    # Continue with original messages (may still fail on the API call, but we tried)
    # Optionally: truncate messages manually as a last resort
```

---

## 🟡 MAJOR BUG #4: Inconsistent Error Swallowing in auxiliary_client.py
**File:** `/Users/enyuanzhang/Desktop/github/hermes-agent/agent/auxiliary_client.py`
**Lines:** 2074-2106 (call_llm function)
**Severity:** MEDIUM - Error Recovery Issues

### The Bug
The `call_llm()` function attempts to recover from max_tokens parameter errors but has inconsistent error handling:

```python
# Lines 2075-2089
try:
    return client.chat.completions.create(**kwargs)
except Exception as first_err:
    err_str = str(first_err)
    if "max_tokens" in err_str or "unsupported_parameter" in err_str:
        kwargs.pop("max_tokens", None)
        kwargs["max_completion_tokens"] = max_tokens
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as retry_err:
            # If the max_tokens retry also hits a payment error,
            # fall through to the payment fallback below.
            if not _is_payment_error(retry_err):
                raise
            first_err = retry_err

    # ── Payment / credit exhaustion fallback ──────────────────────
    if _is_payment_error(first_err):
        fb_client, fb_model, fb_label = _try_payment_fallback(
            resolved_provider, task)
        if fb_client is not None:
            # ... fallback call ...
            return fb_client.chat.completions.create(**fb_kwargs)
    raise
```

**The Problem:**
1. After the max_tokens retry fails (line 2084), `first_err = retry_err` (line 2089) overwrites the original error
2. If `retry_err` is NOT a payment error but also NOT a max_tokens error, it's raised immediately (line 2088: `raise`)
3. **But** if `retry_err` IS a payment error, control falls through to line 2096
4. However, the original `first_err` has been overwritten, so logging loses context about what happened on line 2083

**Worse:** If the original error was a max_tokens issue and the retry introduced a NEW error type:
```
first_err = "unsupported_parameter: max_tokens"  # Original
retry_err = "401 Unauthorized"  # New error during retry
first_err = retry_err  # ← Overwrites, we lose the max_tokens context
if _is_payment_error(retry_err):  # False for 401
    raise  # ← Raises 401, not max_tokens
```

### Consequence
- Misclassified errors returned to caller
- Difficult debugging (error context lost)
- Fallback logic unreachable in some cases

### Fix
Keep both errors and log more carefully:

```python
try:
    return client.chat.completions.create(**kwargs)
except Exception as first_err:
    err_str = str(first_err)
    if "max_tokens" in err_str or "unsupported_parameter" in err_str:
        kwargs.pop("max_tokens", None)
        kwargs["max_completion_tokens"] = max_tokens
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as retry_err:
            # Keep original error for logging/context
            if _is_payment_error(retry_err):
                logger.debug(
                    "Max_tokens retry failed with payment error (original: %s): %s",
                    str(first_err)[:100], str(retry_err)[:100]
                )
                # Fall through to payment fallback with retry_err
                first_err = retry_err
            else:
                # Retry also failed but not payment-related — raise the original
                raise first_err from retry_err

    if _is_payment_error(first_err):
        # ... fallback ...
    raise first_err
```

---

## 🟡 MAJOR BUG #5: Session ID Change Without Exception Recovery
**File:** `/Users/enyuanzhang/Desktop/github/hermes-agent/run_agent.py`
**Lines:** 6041-6071 (_compress_context method)
**Severity:** MEDIUM - Session State Corruption

### The Bug
When context compression occurs, the session ID is changed (line 6051):

```python
# Line 6048-6051
self._session_db.end_session(self.session_id, "compression")
old_session_id = self.session_id
self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
```

**But this happens inside the _compress_context() method, which is called from the main conversation loop.**

If an exception occurs AFTER the session ID change but BEFORE the compressed messages are written to the DB:
```python
# Lines 6045-6071
if self._session_db:
    try:
        # ... session split ...
        self.session_id = new_id  # ← Changed
        # ...
        self._session_db.create_session(...)  # ← What if this fails?
        # Lines 6062-6066: What if _get_next_title_in_lineage() or set_session_title() fails?
        self._session_db.update_system_prompt(...)  # ← What if this fails?
    except Exception as e:
        logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)
        # ← Control returns to caller, but self.session_id is already changed!

# Caller (main loop) continues with the NEW session_id
# But the compressed messages are still in memory and not yet flushed to DB
# If an exception occurs in the API call, the messages are written to the NEW session
# But the compressor's previous summary was meant for the OLD session
```

### Consequence
- New session created but not fully initialized
- Compressed messages written to the new session without proper parent tracking
- Session lineage broken
- Database in inconsistent state

### Fix
Roll back session_id on exception:

```python
if self._session_db:
    old_session_id = self.session_id
    try:
        self._session_db.end_session(old_session_id, "compression")
        self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        # ... rest of creation ...
    except Exception as e:
        # Rollback: restore old session_id
        self.session_id = old_session_id
        logger.warning("Session DB compression split failed — continuing with original session: %s", e)
        # Do NOT end_session since we're rolling back
```

---

## Summary Table

| Bug | File | Lines | Severity | Impact |
|-----|------|-------|----------|--------|
| #1: Role Violation | context_compressor.py | 694-728 | CRITICAL | Data loss, API crash after compression |
| #2: Double Execution | scheduler.py | 843-892 | CRITICAL | Duplicate cron jobs |
| #3: Unhandled Compression Exception | run_agent.py | 8204, 8262, 8338 | HIGH | Silent crash during recovery |
| #4: Error Swallowing | auxiliary_client.py | 2074-2106 | MEDIUM | Misclassified errors, lost context |
| #5: Session ID Change | run_agent.py | 6041-6071 | MEDIUM | Session state corruption |

---

## Recommended Priority Order
1. **#1** (Role Violation) - Fix immediately, affects all compression users
2. **#2** (Double Execution) - Fix immediately, affects all cron users
3. **#3** (Unhandled Exceptions) - Add error recovery to prevent data loss
4. **#4** & **#5** - Fix to improve robustness and debuggability

