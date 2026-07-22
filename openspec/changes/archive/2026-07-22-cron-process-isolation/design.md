# Design: Parent-owned cron supervision

1. `run_one_job` remains the authoritative execute/deliver/mark path. Built-in tick dispatch adds `_running_job_ids` and a durable execution record before submitting work.
2. Agent-backed work runs in a `spawn` child. The child creates a new process group, sends a ready handshake, waits for parent authorization, then runs the existing agent path. The parent registers the process while holding `_running_lock`, supervises the result pipe, sends SIGTERM at the wall bound, waits a bounded grace period, escalates to SIGKILL, and joins before releasing parent state.
3. Script-only jobs retain their existing script timeout path. `max_runtime_seconds: 0` remains an explicit unlimited wall-time opt-out; inactivity timeout semantics remain unchanged.
4. Parent process-global environment is never changed by an agent child. Existing child-local CWD coordination remains harmless for compatibility; the reader/writer lock notifies all waiters when a timed-out writer abandons acquisition.
5. Runtime status is held in `_runtime_states` for built-in runs and exposed as `running`, `cancelling`, `terminal`, or durable `claimed` in cron formatting. Historical `last_status` is not used as a liveness signal.

## Failure handling

Child pipe closure without a result is a failed execution. All child termination paths join/reap before the parent worker's `finally` removes `_running_job_ids`; `run_one_job` then performs the existing authoritative mark/finish transitions exactly once.
