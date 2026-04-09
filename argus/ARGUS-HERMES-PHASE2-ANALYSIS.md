# ARGUS → Hermes Deep Porting Analysis (Phase 2)

**Date:** 2026-04-08
**Status:** Investigation complete, ready for implementation decisions

---

## Opportunities Found

### 1. Entropy Detection via `SessionDB.get_messages()` (HIGH VALUE)

**Current:** ARGUS maintains its own `tool_calls` table, populated by external ingestion.
**Port to:** `SessionDB.get_messages(session_id)` — tool calls are stored directly in session messages.

**Evidence:**
```
Session: "ML Data Collection and Quality Activation"
  Assistant messages with tool_calls: 89
  Total tool calls: 96
    patch: 37
    terminal: 14
    execute_code: 13
    skill_manage: 10
  Repeat sequences (3+ consecutive): 42
```

Tool calls are in `msg['tool_calls'][i]['function']['name']` on `role: assistant` messages.

**Implementation:**
```python
db = SessionDB(DEFAULT_DB_PATH)
msgs = db.get_messages(session_id)
tool_calls = []
for msg in msgs:
    if msg.get('role') == 'assistant' and msg.get('tool_calls'):
        tc = msg['tool_calls']
        if isinstance(tc, str):
            tc = json.loads(tc)
        for call in tc:
            name = call.get('function', {}).get('name')
            if name:
                tool_calls.append(name)

# Detect repeats
for i in range(len(tool_calls) - 2):
    if tool_calls[i] == tool_calls[i+1] == tool_calls[i+2]:
        # Repeat detected!
```

**Tradeoff:**
- ✓ Real session data (no ingestion lag)
- ✓ No external process needed to populate tool_calls table
- ✗ Doesn't track file changes (hash comparison)
- ✗ JSON parsing overhead per message
- ✗ Only works for sessions in SessionDB (not external processes)

**Recommendation:** Hybrid — use SessionDB for entropy detection, keep argus.db for file change tracking and ARGUS-specific metadata.

---

### 2. Corrective Prompt Injection via `SessionDB.update_system_prompt()` (HIGH VALUE)

**Current:** ARGUS records corrective prompts in `watcher_actions` table but can't actually inject them into running sessions.
**Port to:** `SessionDB.update_system_prompt(session_id, corrective_prompt)` — changes system prompt for the NEXT message.

```python
db = SessionDB(DEFAULT_DB_PATH)
db.update_system_prompt(
    session_id,
    f"ENTROPY CORRECTION: {corrective_prompt}\n\n{original_system_prompt}"
)
db.close()
```

**Impact:** ARGUS can now actually correct running sessions, not just record the intent. This turns the `inject_prompt` action from a no-op into a real intervention.

**Caveat:** Only works for sessions tracked in SessionDB. Manual sessions spawned outside the gateway won't be affected.

---

### 3. Direct Job Execution via `cron.scheduler.run_job()` (MEDIUM VALUE)

**Current:** ARGUS uses `trigger_job()` which schedules the job for the next tick.
**Port to:** `cron.scheduler.run_job(job_dict)` — executes a job immediately in-process.

```python
from cron.scheduler import run_job
job = get_job(job_id)
success, output, error, delivery_error = run_job(job)
# Record outcome
mark_job_run(job_id, success=success, error=error, delivery_error=delivery_error)
```

**Impact:** Immediate execution instead of waiting for the next scheduler tick (up to 30s delay).

---

### 4. Session Forensics via `SessionDB.export_session()` (MEDIUM VALUE)

**Current:** ARGUS's `collect_metrics` reads tool_calls and terminal_commands from its own tables.
**Port to:** `SessionDB.export_session(session_id)` — full session export with messages, metadata, token counts.

**Use case:** When ARGUS detects entropy, it can export the full session for analysis:
```python
data = db.export_session(session_id)
# Analyze: messages, tool calls, token usage, session duration
```

---

### 5. Message Search via `SessionDB.search_messages()` (LOW-MEDIUM VALUE)

**Current:** No content analysis capability.
**Port to:** `SessionDB.search_messages(query, role_filter=['tool'])` — search tool results for error patterns.

**Use case:** Detect repeated error messages as an entropy signal.

---

## Items to SKIP

| Item | Why Skip |
|------|----------|
| `TelegramAdapter.send()` | Async, requires adapter instance + connection setup. Overkill for daemon alerts. |
| `SessionDB.ensure_session()` | ARGUS tracks different metadata than SessionDB. Hybrid is better. |
| `hermes_tools.terminal()` | ARGUS IS the monitoring daemon — shouldn't use agent tools. |

---

## Recommended Implementation

| Phase | Change | Effort | Impact |
|-------|--------|--------|--------|
| 2A | Entropy detection via SessionDB | Medium | Replaces external tool_calls ingestion |
| 2B | Corrective injection via update_system_prompt | Low | Makes inject_prompt actually work |
| 2C | Direct job execution via run_job | Low | Eliminates scheduler tick delay |
| 2D | Session forensics via export_session | Low | Better debugging on entropy detection |

## File Changes Needed

| File | Change |
|------|--------|
| `argus.py` | Add `_detect_entropy_from_sessiondb()`, update `_inject_*_prompt()` to use `update_system_prompt()` |
| `test_argus.py` | Tests for SessionDB-based entropy detection |
| `config.yaml` | No changes needed |
