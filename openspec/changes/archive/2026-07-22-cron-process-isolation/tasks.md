# Tasks: Cron process isolation

- [x] Inspect current scheduler, lock, claim, status, and gateway shutdown paths.
- [x] Add killable child process-group execution with startup handshake.
- [x] Bound normal termination, escalation, join, and pipe cleanup.
- [x] Wake readers when a queued writer abandons acquisition.
- [x] Expose parent-owned runtime state in cron status.
- [x] Document wall-time semantics and explicit zero opt-out.
- [x] Add subprocess-level regression coverage for timeout, reaping, and descendant cleanup.
- [x] Run full repository-native verification and archive the change.
