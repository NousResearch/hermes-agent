## Summary

Fix a thread-safety bug in `model_tools.py` where the process-global `_last_resolved_tool_names` was written and read without synchronization, risking stale or partial reads in concurrent tool dispatch.

## Bug

`_last_resolved_tool_names` at `model_tools.py:217` is a module-level list that is:
- Written by `get_tool_definitions()` (both cache-hit and compute paths)
- Read by `handle_function_call()` when dispatching `execute_code`

Under concurrent access (gateway serving parallel sessions), one thread could read a stale reference while another thread is swapping the list, leading to `execute_code` seeing an outdated tool catalog.

## Fix

Added `_last_resolved_tool_names_lock` (threading.Lock) and wrapped all access:

1. **Writes**: `get_tool_definitions()` now acquires the lock before updating the global
2. **Reads**: `handle_function_call()` now acquires the lock and copies the list before passing it to `execute_code`

## Impact

- **Severity**: P1 — stale tool catalog could cause execute_code to reference unavailable tools
- **Scope**: `model_tools.py` (core tool orchestration)
- **Risk**: Minimal — lock is only held during list assignment/copy (~microseconds), no contention in practice

## Testing

- `tests/test_model_tools_thread_safety.py`: 4 regression tests
  - `test_write_protected_by_lock`: get_tool_definitions acquires the lock
  - `test_read_copies_under_lock`: handle_function_call copies the list under lock
  - `test_concurrent_writes_dont_corrupt`: 10 threads writing concurrently don't corrupt the list
  - `test_read_during_write_returns_complete_list`: a read during a write always sees a complete list
