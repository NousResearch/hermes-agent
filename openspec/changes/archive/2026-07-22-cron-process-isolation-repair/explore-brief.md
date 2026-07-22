# Explore brief: cron process-isolation repair

## Baseline
- ADR 0006 is the architectural anchor for this active OpenSpec change.
- `cron/scheduler.py` supervises spawned cron children and owns `_cron_processes`.
- `cron/jobs.py` stores advisory external `fire_claim` records.
- `cron/executions.py` stores durable claimed/running/terminal ownership.
- `cron/scheduler_provider.py` creates external-fire executions and invokes the shared runner.

## Preserved-worktree evidence
- The complete dirty diff was inspected before this re-base: tracked changes add isolated child supervision, durable fire ownership, runtime state, shutdown integration, CWD-lock repair, documentation, and focused regressions; untracked `cron/process_boundary.py` and `tests/cron/test_process_isolation.py` add the cgroup boundary and capability-gated supervisor/direct-boundary regressions.
- The boundary creates a unique child cgroup under the caller's discovered v2 cgroup, records parent and child inode identity, parks the multiprocessing child on a start pipe, assigns and verifies the child PID, and releases that pipe only after success. It targets `cgroup.kill`, not a reconstructed PID list.
- Allocation now proves the actual owned boundary's create/kill/remove/recreate contract before returning it; the capability-gated supervisor regression exercises `_run_isolated_cron_job`, while injected failures cover assignment, allocation cleanup, and termination ownership retention.

## Findings to close
1. Registry removal must occur only after forced process-group cleanup and confirmed `join`, so shutdown cannot miss a live child.
2. Parent result polling must never sleep beyond the remaining wall deadline.
3. External-fire claims must carry durable execution ownership; an expired advisory timestamp cannot replace a provably live owner.
4. Tests must exercise the real supervisor boundary, including a descendant and a non-cooperative worker.
5. The product validation surface includes focused cron tests, static checks, and structural OpenSpec validation.
6. `/proc` descendant polling, `PR_SET_PDEATHSIG`, and temporary process-global subreaper adoption are not authoritative containment boundaries.
7. Hard detached-session containment is available only on Linux when a dedicated, verifiably owned cgroup-v2/systemd transient unit or scope can be created before workload start. Otherwise fail closed for the hard-containment claim; process-group cleanup is documented best effort only.

## Questions resolved by the re-base
1. **Capability discovery:** Linux capability is affirmative only after discovering cgroup v2 or systemd support, a writable delegated parent/control-group path, an empty new boundary, stable identity, membership assignment/readback, and a cgroup-wide termination primitive. Absence, permission denial, delegation refusal, or ambiguous identity is not a degraded form of hard containment.
2. **Ownership:** a boundary handle is an opaque capability, not a job-id-derived path. It carries the expected parent plus immutable identity evidence and is revalidated before assignment and teardown.
3. **Launch ordering:** the child remains parked after it becomes observable and registered; no agent or tool code may begin before membership and ownership verification complete. Assignment/verification failure kills the parked child and tears down only the still-owned boundary.
4. **Termination:** only a revalidated owned cgroup/unit is targeted. Completion requires boundary emptiness and child reaping; inability to prove either retains ownership and reports `cleanup_failed`.
5. **Portability:** macOS and Windows are unsupported for detached-session hard containment. Linux without usable delegation is `unavailable`. In every such case the workload may use best-effort process-group cleanup, but status and documentation must not represent it as hard containment.

## Constraints
- Preserve existing process isolation, startup handshake, parent watchdog, timeout escalation, cwd locking, and delivery behavior.
- macOS and Windows do not claim equivalent detached-session containment; capability reporting is explicit.
