# Explore brief: cron process isolation

## ADR and scope baseline

- Serve ADR 0006: Spec-Driven Development via OpenSpec & Reflection Harness.
- Preserve the accepted ADR layer; this change is limited to cron agent lifecycle isolation,
  authoritative timeout cleanup, cwd/lock safety, runtime state, and evidence.
- Existing script-only (`no_agent`) jobs retain their script timeout path.

## Code paths inspected

- `cron/scheduler.py`: `run_job`, `run_one_job`, `_run_isolated_cron_job`,
  `_cron_child_entry`, `_terminate_cron_process_tree`, and `_terminal_cwd_lock`.
- `gateway/run.py`: scheduler shutdown and active-job cancellation integration.
- `tests/cron/test_terminal_cwd_lock.py` and `tests/cron/test_process_isolation.py`.
- Cron user documentation describing inactivity and wall-clock semantics.

## Required invariants

1. Agent-backed jobs execute in a killable child process group supervised by the parent.
2. The child must complete a ready/authorization handshake before agent execution.
3. Parent registration, timeout escalation, process join/reap, pipe closure, and registry removal
   are bounded and occur on every termination path.
4. A non-cooperative agent cannot retain scheduler ownership or a process-global cwd override
   indefinitely after inactivity or wall timeout.
5. `cron.max_runtime_seconds: 0` remains an explicit unlimited wall-time setting; inactivity
   termination remains authoritative and cannot wait forever for a Python worker thread.
6. Script-only jobs are not routed through the agent child-process path.
7. Runtime status and execution-ledger transitions remain parent-owned and are not inferred from
   historical `last_status`.
8. Tests exercise production supervision lifecycle, not only the low-level kill helper.

## Verification baseline

- Focused cron tests via `scripts/run_tests.sh`.
- Strict OpenSpec validation if the CLI is available.
- Governance evidence records a pre-apply reflection PASS before archive.
