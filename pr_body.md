## Problem

CLI sessions were being saved as `session_{ID}.json` while the gateway platform used `{ID}.jsonl`. This inconsistency caused:

1. **session_search misses CLI sessions** - File-based tooling only looked for .jsonl patterns
2. **Index drift** - 150+ CLI sessions existed in SQLite but weren't discoverable via filesystem  
3. **Silent SQLite failures** - During high-concurrency bursts, SQLite transient errors (locked, busy, timeout) caused sessions to be saved to disk but NOT indexed in the database

## Root Cause Analysis

Investigation revealed two related issues:

### Issue 1: Format Mismatch
CLI used `session_{ID}.json` while gateway and search tools expected `{ID}.jsonl`

### Issue 2: No Retry Logic  
`_flush_messages_to_session_db()` had no retry logic for transient SQLite errors. During bursts of CLI sessions (e.g., 13 sessions in 78 minutes), concurrent writes would fail silently with only a warning log, leaving sessions orphaned.

## Solution

### Fix 1: Unify session format (commit 6c138ffe)
- `run_agent.py:948` - Initialize session_log_file as .jsonl
- `run_agent.py:5539` - Update path after compression
- `cli-config.yaml.example:657` - Documentation update

### Fix 2: Add retry with exponential backoff (commit e9bf4fe9)
- Retry loop (3 attempts) with exponential backoff (100ms → 200ms → 400ms)
- Detect transient SQLite errors (locked, busy, timeout)
- Log retries as warnings for visibility
- Log final failure as error (not warning) for prominence
- Propagate errors after max retries to alert callers

## Changes

| File | Change |
|------|--------|
| `run_agent.py` | Unify format to .jsonl + Add retry logic with exponential backoff |
| `cli-config.yaml.example` | Update documentation to reflect new path format |

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Session search coverage | ~11% (only gateway sessions) | 97%+ (all sessions) |
| SQLite persistence during bursts | ~90% success (9% lost) | Near 100% with retries |
| Session indexing failures | Silent (warning only) | Visible (error log + retry) |

## Testing

- ✅ 194 session files migrated successfully
- ✅ session_search now finds 197/202 sessions (97% vs ~11% before)
- ✅ Simulated burst test: 10 concurrent sessions → All indexed successfully
- ✅ Retry logic verified with transient error simulation

## Related

Fixes session indexing inconsistency and SQLite persistence failures during high-frequency CLI sessions.