# Reflection review log

## Round 1 — pre-apply

**Reviewer:** Change-Reviewer
**Baseline:** `explore-brief.md`, ADR 0006, and the accepted cron process-isolation scope.

### Checks

- ADR citation present in `proposal.md`: PASS.
- Proposal/design/tasks/spec delta describe the same implementation scope: PASS.
- Child startup handshake and parent registration race are specified: PASS.
- Parent-owned timeout escalation, join/reap, pipe failure, and registry cleanup are specified: PASS.
- `max_runtime_seconds: 0` and script-only semantics are preserved: PASS.
- Inactivity timeout is bounded even when `agent.interrupt()` is non-cooperative: PASS after repair in `cron/scheduler.py` and regression coverage in `tests/cron/test_process_isolation.py`.
- No accepted ADR contradiction identified: PASS.

**Result: PASS — apply authorized.**

## Archive verification

The implementation was applied before this evidence was restored in the preserved checkout. This log records the required pre-apply reflection result and the repair review of the archived artifact; it does not claim a new implementation scope.

## Round 2 — post-apply lifecycle regression

**Reviewer:** Local focused verification
**Scope:** Current uncommitted checkout, including `tests/cron/test_process_isolation.py`.

### Checks

- Child handshake, parent registration, result delivery, and registry removal: PASS (`test_isolated_supervisor_handshake_result_and_registry_cleanup`).
- Wall-time SIGTERM/SIGKILL escalation and reaping: PASS (`test_isolated_supervisor_escalates_and_reaps_on_wall_timeout`).
- Child pipe failure is bounded and reaped: PASS (`test_isolated_supervisor_reports_pipe_failure_and_reaps`).
- Non-cooperative inactivity path reaches the child termination boundary and parent reaps/removes registration: PASS (`test_isolated_supervisor_reaps_child_after_non_cooperative_inactivity`).
- Focused command `scripts/run_tests.sh tests/cron/test_process_isolation.py tests/cron/test_terminal_cwd_lock.py -q`: PASS (12 tests).
- Scheduler regression command `scripts/run_tests.sh tests/cron/test_scheduler.py -q`: PASS (222 tests).

**Result: PASS — production supervisor lifecycle coverage is present in the preserved checkout.**

## Round 3 — independent repair review

**Reviewer:** Codex CLI 0.144.1, `gpt-5.6-terra` (read-only; session `019f87a3-2a1c-7fc2-9926-b3afe11cb4ad`)
**Scope:** Complete current tracked and untracked diff against the cron lifecycle contract and ADR 0006.

### Findings repaired

- P1: a child leader could exit before descendants, leaving the parent to join without signalling the still-live process group. Parent final cleanup now always force-signals the process group before join, and a subprocess regression proves orphan descendants do not survive.
- P1: wall-time supervision coverage did not prove escalation. The stubborn-child fixture now ignores SIGTERM and asserts the parent calls TERM then KILL.
- P2: archive layout used an extra date directory. The change is now at `openspec/changes/archive/2026-07-22-cron-process-isolation/` as required by ADR 0006.

**Result: FAIL repaired — a fresh independent final review is required before advancement.**

## Round 4 — setsid descendant containment repair

**Reviewer:** Local implementation and focused verification
**Scope:** Current uncommitted checkout, including the cron process-isolation repair.

- Linux isolated children install a parent-death boundary and the scheduler
  enables a Linux subreaper to adopt, signal, and reap descendants that escape
  the leader's process group: PASS.
- Cleanup repeats containment after reaping the leader so descendants adopted
  during leader teardown are included: PASS.
- Focused command `scripts/run_tests.sh tests/cron/test_process_isolation.py tests/cron/test_terminal_cwd_lock.py tests/cron/test_scheduler.py -q`: PASS (235 tests).

**Result: PASS — local repair verification complete; fresh independent Terra review remains required.**

## Round 5 — independent final review

**Reviewer:** Codex Terra (`gpt-5.6-terra`)

**Result: FAIL — P1 remains.**

Terra accepted the PID-reuse guard, temporary subreaper restoration, and batched execution-state reuse, but found a race in the final cleanup: a `setsid()` descendant can spawn and outlive the leader before the parent-side watcher records its PID, then remain alive after cleanup. The focused tests pass in this environment, but this race requires a stronger supervisor-side containment design before advancement.
