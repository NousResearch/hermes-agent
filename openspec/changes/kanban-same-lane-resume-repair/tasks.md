# Implementation tasks

The OpenSpec phase precedes tests and implementation. Completion boxes below must reflect new execution against the surviving candidate, not the earlier staged prototype.

## 1. Specification gate

- [x] Record the Full OpenSpec classification, intent, non-goals, and immutable exact-target identity.
- [x] Define card/lane/PR/worktree/upstream/substantive-outcome guards and transaction boundaries.
- [x] Define gated PID/start/session handshake, atomic publication, compensation, cleanup-failure, concurrency, observability, compatibility, and live-verification behavior.
- [x] Obtain independent read-only review/approval of this package before changing tests or controller code.

## 2. Test-first implementation

- [x] Replace or extend the staged tests so the exact required behavior fails on the pre-fix controller.
- [x] Cover target READY/unclaimed success and rejection for every stale task/claim/assignee field.
- [x] Cover strict competing nonterminal Flynn lanes for TODO, READY, RUNNING, BLOCKED, and REVIEW; terminal DONE/ARCHIVED cards do not block.
- [x] Cover same-card PR ownership and refusal when the PR belongs to another task or the target has a conflicting PR.
- [x] Cover PR state/base/head-ref/head-OID mismatches through typed observer evidence.
- [x] Cover missing/dirty/detached/wrong-branch worktree and local/upstream/OID mismatches.
- [x] Recreate the exact history: run `1144` `changes_requested`, followed by runs `1146` and `1149` as claimed/heartbeat-only `reclaimed` housekeeping.
- [x] Prove a later reclaimed run with genuine spawned-worker evidence is substantive and blocks reuse of run `1144`.
- [x] Cover two concurrent attempts with the same one-use authorization ID before, during, and after compensation; prove at most one child launch and durable receipt replay.
- [x] Cover an explicitly new authorization ID after compensation and require a fresh complete guard evaluation before retry.
- [x] Cover atomic READY/unclaimed -> BLOCKED/`resume_quarantined` admission with no run/claim, and crash recovery from quarantine before child creation.
- [x] Cover a contender arriving after publication rollback but before child cleanup; the task-scoped resume fence must prevent a second launch.
- [x] Cover launch failure, invalid PID, missing/duplicate session, child exit during handshake, and observer timeout.
- [x] Cover claim/run/handle/`spawned` write failures after process creation and prove rollback plus confirmed termination.
- [x] Add Linux-only crash-injection coverage for controller death before/after parent-death setup and before/after child-identity fsync; prove no wrapper/descendant survives. Also cover after commit/before release, after release send/before acknowledgement, and release acknowledgement failure.
- [x] Cover cleanup failure, including leader exit with a surviving process-group/tree descendant, and prove the card remains fail-closed until full group/tree extinction.
- [x] Assert no claimed-only run, heartbeat-only run, pre-commit heartbeat, or orphan child remains after compensated failure.
- [x] Cover task-level `resume_quarantined`, `spawn_failed`, and `cleanup_failed` with `run_id = null` through existing event readers, exports, hooks, serializers, and notification paths.
- [x] Assert ordinary active-PR suppression for every other task and the existing serial Node/Flynn-lane posture remain unchanged.
- [x] Reproduce READY plus null claim/current-run/PID/session and stale `worker_started_at` against an isolated board; retain the failing result before implementation.
- [x] Cover atomic stale-fingerprint normalization, durable evidence, active claim/run/PID/session refusal, and unchanged terminal heartbeat-only history.

## 3. Controller/kernel change

- [x] Add immutable request and typed observer-result contracts with a one-use resume authorization ID; do not accept a bare Boolean verifier.
- [x] Bind the initial authorization explicitly to `t_dfa23a41` / PR `#155`; any different task or PR requires a separate reviewed change.
- [x] Add strict latest-substantive-run selection that skips only proven no-launch housekeeping reclaims/spawn failures.
- [x] Add a board-and-task-scoped interprocess resume fence held across quarantine, rollback, full process-group/tree cleanup, and compensation, with atomically fsynced authorization/launch nonce/phase/deadline/PID/start/process-group metadata and bounded owner-death recovery.
- [x] Use task-level SQLite events as canonical atomic authorization receipts; reconcile sidecar process metadata against the quarantined board state after crashes.
- [x] Atomically quarantine READY/unclaimed as BLOCKED/`resume_quarantined` before spawn, with no claim/run/PID/session; revalidate every guard before publication.
- [x] Add the explicit controller-only operation without wiring it into ordinary dispatcher scanning.
- [x] Add a gated child handshake that yields a live PID, process-start fingerprint, and child-confirmed session before durable publication.
- [x] Add the hardened Linux launcher wrapper: parent-death signal plus expected-parent PID/start recheck before imports/threads/descendants/work; refuse unsupported hosts.
- [x] Make gate EOF, parent death, timeout, or invalid token terminate the child before work; make the child persist `resume_released` after committed-run readback and before acknowledgement/work; add crash recovery.
- [x] Atomically publish claim/run/PID/start/session/`claimed`/`spawned`, commit, then release the child gate.
- [x] Add bounded compensation and fail-closed cleanup handling with task-level `spawn_failed` evidence and no phantom run.
- [x] Add structured refusal/failure observability without secrets or Finance business data.
- [x] Add the narrow board-kernel lifecycle normalizer and invoke it only from the exact authorized operation before complete admission.

## 4. Verification

- [x] Run `git diff --check` for the scoped candidate.
- [x] Run the focused same-lane resume suite through `scripts/run_tests.sh`.
- [x] Run existing Kanban active-PR guard, worker-session, worker-lifecycle, reclaim, and serial dispatch suites through `scripts/run_tests.sh`.
- [x] Confirm the diff contains only the approved Hermes controller/kernel tests/code and this OpenSpec package; exclude unrelated staged files.
- [x] Obtain independent non-author review of the surviving candidate.

## 5. Safe live verification and handoff

- [ ] Wait until every other Flynn card is DONE or ARCHIVED; otherwise the mandatory lane guard must refuse.
- [ ] From a non-worker controller principal, use only approved read-only PR/worktree observers and invoke the exact target request once.
- [ ] Read back one coherent task/run/PID/start/session/`spawned` identity and verify no heartbeat predates publication.
- [ ] Confirm ordinary active-PR suppression remains effective for all other tasks and no replacement card/worktree/PR was created.
- [ ] Record immutable test/review/live evidence without mutating GitHub, deploying, starting Finance runtimes, or interacting with PR `#37`.
