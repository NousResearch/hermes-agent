# Design: guarded atomic same-lane resume

## 1. Design goals

The design provides one narrow recovery path for an existing card whose own open PR would otherwise trigger the generic duplicate-lane guard. It must preserve four invariants:

1. Same-card identity: the resumed card owns the exact PR being remediated.
2. Serial execution: no other nonterminal Flynn card exists when the live target is resumed.
3. Atomic publication: durable task/run/spawn state appears as one coherent unit only after a real child PID and session are known.
4. Fail-closed compensation: a failed launch/publication cannot leave a runnable orphan or a claimed/heartbeat-only phantom run.

## 2. Explicit request and verification types

The controller accepts an immutable `SameLaneResumeRequest` equivalent to the fields below. This OpenSpec authorizes exactly the listed target values; the implementation must reject a different task ID or PR even if all generic predicates would otherwise pass. Generalizing the operation requires a separate reviewed OpenSpec.

```text
task_id
authorized_assignee
expected_pr_url
expected_pr_number
expected_pr_base
expected_pr_head_ref
expected_pr_head_oid
expected_workspace
expected_upstream_ref
required_substantive_run_id
required_substantive_outcome
resume_authorization_id
```

For the authorized live target these values are:

```text
task_id: t_dfa23a41
authorized_assignee: flynn
expected_pr_url: https://github.com/NXE-ORG/nxe-helix-alpha/pull/155
expected_pr_number: 155
expected_pr_base: dev
expected_pr_head_ref: wt/finance-pr153-producer-output-closure
expected_pr_head_oid: 1b910d4ff9d1c9fe103da5df84d302ba81fc2cc1
expected_workspace: /home/hermes/nxe-helix-alpha/.worktrees/t_dfa23a41
expected_upstream_ref: origin/wt/finance-pr153-producer-output-closure
required_substantive_run_id: 1144
required_substantive_outcome: changes_requested
resume_authorization_id: controller-issued one-use ID for this exact approved invocation
```

The observer returns a typed result containing the observed PR URL/number/state/base/head ref/head OID and workspace path/branch/status/local OID/upstream ref/upstream OID. A bare Boolean verifier is insufficient because it cannot prove which identity was checked. The kernel compares every observed field to the immutable request while the claim lease is held.

The session ID is not supplied as trusted input. It is generated for this launch and confirmed by the child handshake.

Before entering the SQLite write lease, the controller acquires a board-and-task-scoped interprocess resume fence. Every same-lane resume attempt for that board/task must take this fence. The fence is held through admission, launch, commit or rollback, child cleanup, and compensation evidence. It is not a run handle or worker claim. Ordinary dispatch remains independently prevented by the unchanged active-PR guard. This closes the otherwise unsafe gap in which a rolled-back transaction would release SQLite while its child was still being terminated.

The fence has a crash-surviving, atomically replaced metadata record in the board's existing lock/state area (not a SQLite task/run row and not a new configuration surface). Before process creation it records owner identity, the one-use `resume_authorization_id`, a unique launch nonce, phase `launching`, and the finite gate deadline. After fork/spawn and before handshake success, the child writes and fsyncs the nonce-bound PID, process-start fingerprint, and process-group identity, then reports readiness. The controller must read back the exact fsynced identity before publication. A child that cannot persist this identity exits while still gated.

The SQLite event ledger, not the sidecar fence file, is the canonical authorization receipt store. The initial quarantine transaction appends task-level `resume_quarantined` with the authorization ID and no run ID. Success appends `spawned` with that authorization ID in the same transaction that publishes RUNNING/run state. Compensation appends task-level `spawn_failed` with that authorization ID in the same transaction that restores READY/unclaimed; uncertain cleanup appends task-level `cleanup_failed` while leaving the card quarantined. Under the board write lease, any prior receipt for the authorization ID makes the operation replay its in-progress/terminal disposition instead of spawning.

The sidecar fence metadata mirrors the authorization ID and process identity only for crash recovery; it is not the source of truth for `published`, `compensated`, or `cleanup_failed`. If sidecar and SQLite differ after a crash, recovery reconciles process liveness first and then completes one board transaction from `resume_quarantined` to either RUNNING/published or READY/`spawn_failed`; it never infers a terminal receipt from the sidecar alone. Retrying after a compensated failure requires an explicit new controller-issued authorization ID and a fresh complete guard evaluation.

The initial live path is Linux-only. The only process created before durable child identity is a hardened launcher wrapper whose first instructions, before threads, descendants, imports of the agent runtime, or worktree access, install `PR_SET_PDEATHSIG`, verify the expected parent PID/start identity did not change during setup, and enter the fail-closed gate. If the parent died before `prctl`, the post-install parent check forces immediate exit; if it dies afterward, the kernel sends the death signal. The wrapper is prohibited from creating descendants until after its own PID/start/process-group identity is fsynced and the controller validates it. Unsupported hosts refuse this operation.

Therefore a pre-identity `launching` record plus a dead recorded controller owner is proof that the wrapper either never started or is kernel-forced to exit without descendants. Recovery still waits beyond the recorded finite gate deadline and verifies the controller owner PID/start is dead before compensating or retrying. Once child identity exists, recovery verifies/terminates the exact PID/start process group/tree. Linux crash-injection tests must kill the controller before and after `prctl`, before and after identity fsync, and prove no wrapper or descendant survives.

## 3. Admission guards

All guards are evaluated while one task-scoped resume/claim lease is held. Before a child starts, one board transaction verifies every guard, changes the exact unclaimed card from READY to existing status BLOCKED with reason/kind `resume_quarantined`, and appends the task-level authorization receipt. The operation lease remains held after that commit. Immediately before publication, a second board transaction revalidates every mandatory board, remote PR, worktree, upstream, and substantive-history guard while the child is still gated. Any mismatch compensates without releasing the child.

### 3.1 Card state

- The exact task exists.
- `status == ready`.
- `claim_lock`, `claim_expires`, `current_run_id`, `worker_pid`, `worker_started_at`, and `session_id` are absent.
- The assignee is exactly `flynn` and matches the request.
- The preserved workspace and branch metadata on the card match the request.

One pre-admission normalization is permitted for the exact authorized operation: if and only if the task is READY, `worker_started_at` is non-null, every claim/current-run/PID/session field is null, and no `task_runs` row remains open (`ended_at IS NULL`), board-kernel lifecycle code may clear that fingerprint. The compare-and-swap and a task-level `lifecycle_normalized` event containing the previous value, reason `terminal_residue`, and authorization ID commit in one transaction. Any active status, claim field, current/open run, PID, or session refuses normalization and leaves the fingerprint untouched. The operation then runs the ordinary complete guard evaluation against the normalized row.

This normalizer does not infer liveness from the age or numeric value of the fingerprint, delete history, clear any other field, or relax ordinary dispatcher behavior. Operator scripts must call the kernel operation; they must never update the board database directly.

### 3.2 Serial Flynn lane

No other task assigned to `flynn` may be nonterminal. For this operation, nonterminal means every status except `done` and `archived`, including `todo`, `ready`, `running`, `blocked`, and `review`.

This is intentionally strict. The current specification and implementation cards therefore prevent a live resume until they and every other Flynn card are terminal. Tests SHALL distinguish a competing TODO/READY lane from a competing RUNNING lane; both refuse admission.

### 3.3 Same-card PR ownership

- The target card's durable comments/events/metadata identify the exact expected PR URL.
- The URL resolves to PR `#155` in the expected repository.
- No different nonterminal task identifies or owns that same PR.
- No different active PR is associated with the target card.
- The exceptional path is invoked with the target task ID; it never accepts an arbitrary active PR discovered by scanning other cards.

The ordinary comment-regex active-PR guard remains unchanged for every non-explicit dispatch.

### 3.4 Remote PR identity and state

Using the approved read-only observer while the lease is held, verify:

- state is `OPEN`;
- base ref is exactly `dev`;
- head ref is exactly `wt/finance-pr153-producer-output-closure`;
- head OID is exactly `1b910d4ff9d1c9fe103da5df84d302ba81fc2cc1`; and
- the PR URL/number match the same-card ownership record.

Mergeability and CI may be reported for observability but are not substitutes for the required identity checks and are not mutated by this operation.

### 3.5 Preserved worktree and upstream

Inside the exact preserved workspace, verify:

- the worktree path is the expected path and is still registered for the repository;
- the checked-out branch is the expected head ref;
- the index, tracked files, and untracked-file set are clean;
- local `HEAD` equals the expected OID;
- the configured upstream is the expected upstream ref; and
- the upstream OID equals local `HEAD` and the expected PR head OID.

A missing upstream, detached HEAD, dirty state, branch mismatch, or OID mismatch refuses admission.

### 3.6 Latest substantive outcome

Select the newest substantive terminal run, not simply the newest terminal row. Scan terminal runs newest-first. A run may be skipped as non-substantive only when all of these facts hold:

- its outcome is exactly `reclaimed` or `spawn_failed`;
- task/run PID, process-start fingerprint, and session fields are absent;
- it has no `spawned` or gate-release event;
- it has no worker summary, result, artifact, or worker-produced completion evidence; and
- its event kinds are a subset of controller housekeeping (`claimed`, `heartbeat`, `reclaimed`, `spawn_failed`).

Any run failing any one condition is substantive and becomes the selected candidate. The required selected candidate must have run ID `1144`, profile/reviewer provenance `crash`, outcome `changes_requested`, and matching durable review event. This rule is deterministic and testable against complete run/event rows; it does not infer provenance from timestamps or prose.

For the target, the selected run must be exactly Crash run `1144` with outcome `changes_requested`. Later runs `1146` and `1149` may be skipped only because each is proven claimed/heartbeat-only reclaimed housekeeping with no spawned-worker evidence.

## 4. Launch protocol

The launcher creates a child process group with a one-use start gate and a private handshake channel. The gate channel is fail-closed: EOF, parent death, timeout, or any token other than the one-use commit token causes the child to terminate without entering its worker loop. Where the host supports a parent-death process primitive it should reinforce, not replace, the channel contract. The child may initialize only enough runtime to report:

```text
pid
process_started_at
session_id
ready_but_gated
```

The controller validates:

- PID is a positive live process;
- `process_started_at` matches the canonical process-start fingerprint;
- the process belongs to the newly created child process group;
- session ID is non-empty and unique for the launch; and
- the child is still waiting behind the gate.

Before the gate is released, the child SHALL NOT enter the agent loop, call Kanban heartbeat, mutate the board, enter the Finance worktree, or perform external actions. The gated child also enforces a finite handshake deadline so a controller crash cannot leave it waiting indefinitely.

## 5. Transaction boundaries and ordering

### 5.1 Quarantine and publication transactions

While holding the task-scoped resume fence/claim lease:

1. In an admission transaction, read and validate every board/card/lane/history predicate and typed PR/worktree observation.
2. In that same transaction, transition the exact card from READY/unclaimed to existing status BLOCKED with `resume_quarantined`, leave claim/run/PID/start/session fields empty, append task-level `resume_quarantined` with the one-use authorization ID and `run_id = null`, and commit.
3. Start the gated child and complete the durable fence PID/start/process-group plus session handshake.
4. In a publication transaction, revalidate every mandatory guard. The target's expected state is now its own matching quarantine receipt; every external/ownership/lane/history predicate must still match.
5. Create the run and transition the same card from its matching quarantine to RUNNING.
6. Persist the claim lock/expiry, run ID, PID, process-start fingerprint, and session on task/run records as applicable.
7. Append `claimed` marked `same_lane_resume=true` and `spawned` containing the authorization ID and matching PID/start/session identity.
8. Commit successful publication once.
9. Release the child gate with the one-use commit token only after commit succeeds.
10. The child reads back its committed run identity and atomically persists `resume_released` before it enters the work loop; only then does it acknowledge release to the controller.
11. Permit heartbeat only after durable `resume_released` and child acknowledgement.

No run handle, durable claim, `spawned` event, or heartbeat may exist before step 8 commits. The earlier durable quarantine is explicitly not a run/claim; it prevents ordinary dispatch and preserves fail-closed recovery. No schema migration or direct database repair is required.

### 5.2 Compensation transaction

If pre-quarantine observer validation fails, no child starts and no durable task state changes are required beyond structured refusal telemetry. If post-handshake revalidation fails, the child remains gated and compensation proceeds from durable quarantine.

If launch or handshake fails before publication:

1. keep/close the gate so no work loop can start;
2. terminate the recorded process group/tree if one exists, escalating from graceful termination to forced kill within a bound;
3. verify both that the `(pid, process_started_at)` leader is no longer live and that no member/descendant remains in the recorded process group/tree; and
4. in one transaction transition the same card from its matching quarantine to READY/unclaimed, clear all run/worker/session fields, ensure no run row was published, and append task-level `spawn_failed` with the authorization ID and `run_id = null`.

If publication fails after child creation, roll back the publication transaction to the already committed quarantined state, not to READY. Retain the task-scoped resume fence, perform gated full-tree termination, then execute the separate compensation transaction. Release the fence only after full extinction and READY/`spawn_failed` compensation commit. A failed SQLite transaction must not be reused for evidence persistence.

If the child gate cannot be proven closed, the leader cannot be confirmed dead, or any process-group/tree member remains, safety overrides automatic readiness: append task-level `cleanup_failed` with the authorization ID in a transaction that leaves the card BLOCKED/`resume_quarantined`. Do not expose READY/unclaimed beside a possibly live leader or descendant. Once full group/tree extinction is conclusively established, the controller atomically transitions quarantine to READY/unclaimed plus `spawn_failed`. Tests must cover a descendant that survives leader exit.

### 5.3 Controller-crash and post-commit recovery

- Crash before child creation leaves a durable BLOCKED/`resume_quarantined` card plus the expiring resume fence; recovery either resumes the same authorization or restores READY with `spawn_failed` after proving no child exists.
- Crash after child creation but before publication commit leaves the card quarantined. Before identity fsync, the Linux parent-death wrapper plus expected-parent recheck guarantees no descendant can exist and forces wrapper exit; after identity fsync, the exact PID/start process group is available for termination verification. Recovery proves the controller owner dead, waits beyond the gate deadline for the pre-identity case, or verifies/terminates the recorded tree, before atomically restoring READY with task-level `spawn_failed`.
- Crash after commit but before gate release leaves a real published PID/session/`spawned` identity, not a claimed-only phantom. The gated child exits on EOF/deadline. Reclaim detects a dead PID/start pair and absence of `resume_released`/heartbeat, atomically ends that run as `spawn_failed`, and restores READY/unclaimed.
- If the controller dies after sending the commit token, the child either atomically writes `resume_released` after run readback and proceeds, or exits without entering the work loop. There is no acknowledged-but-unpersisted release state because acknowledgement follows durable release persistence.
- Gate release/acknowledgement failure after commit triggers the same termination and published-run recovery path unless durable `resume_released` already proves the child entered the normal lifecycle. It must never fabricate a heartbeat or erase the real spawned evidence.

The resume fence requires owner identity plus bounded expiry/recovery semantics so controller death cannot wedge the task forever. Recovery may steal an expired fence only after proving the recorded owner is dead and reconciling any recorded child PID/start/process-group identity. When the pre-spawn `launching` record has no child identity, recovery must also wait beyond its recorded fail-closed gate deadline.

## 6. Concurrency behavior

- Two controller principals may attempt the exact request/authorization concurrently, but the task fence plus durable authorization receipt serializes the logical invocation and permits at most one child launch for that authorization.
- The task-scoped resume fence serializes attempts across transaction rollback and compensation; contenders cannot launch while cleanup is in progress.
- Only the lease winner may reach child launch.
- A contender using the same authorization ID that waits through compensation or publication returns the durable receipt without launching, even if the card is READY again. A retry requires a newly issued authorization ID.
- Lease timeout or `database is locked` is a retryable controller failure, not permission to use an unlocked or direct-SQLite path.
- The operation must not start a second dispatcher thread, Node runtime, or Flynn lane.
- Ordinary dispatcher scans continue to treat every active PR as guarded; no automatic scan calls this operation.

Holding the task-scoped operation lease across bounded observer checks, quarantine, gated launch, revalidation, and compensation increases operation duration without holding one SQLite transaction open throughout. Observer/handshake timeouts must be finite; every board transition remains a short explicit transaction under the operation lease.

## 7. Failure modes

| Failure | Required result |
|---|---|
| Card missing, stale, claimed, or not READY | Refuse; no child/run |
| Any competing nonterminal Flynn card | Refuse; no child/run |
| PR belongs to another task or another PR belongs to target | Refuse; no child/run |
| PR closed, base/ref/OID mismatch | Refuse; no child/run |
| Worktree/upstream dirty, missing, or mismatched | Refuse; no child/run |
| Latest substantive outcome is not exact run 1144 `changes_requested` | Refuse; no child/run |
| Observer timeout/error | Refuse; no child/run; structured reason |
| Child returns no valid PID/start/session or exits during handshake | Keep quarantine; terminate full tree; atomically restore READY/unclaimed + `spawn_failed`; no run |
| Claim/run/handle/`spawned` persistence fails | Roll back to quarantine; terminate full tree; compensation transaction; no run |
| Gate release fails after commit | Immediately fence/terminate child; use existing lifecycle recovery to end the published run with explicit failure; never fabricate heartbeat |
| Controller crashes before commit | Fail-closed gate EOF/deadline ends child; durable fence identity is reconciled, or no-identity `launching` deadline elapses, before task-level `spawn_failed` |
| Host lacks the required parent-death primitive | Refuse before quarantine or process creation; no weaker live fallback |
| Controller crashes after commit before release | Dead gated child plus missing `resume_released` is reclaimed as a real spawn failure; READY is restored after liveness check |
| Controller crashes after sending release | Child durably writes `resume_released` before acknowledgement/work or exits; recovery follows the durable branch |
| Termination cannot be confirmed | Fail closed/non-runnable with `cleanup_failed`; reconcile before READY |
| Leader exits but descendant/group member survives | Continue group/tree kill and fail closed; no READY/retry until full extinction |
| Concurrent attempt loses lease | Refuse/retry without launch |
| Concurrent attempt reuses in-progress/terminal authorization ID | Wait/fail fast or return durable receipt; never launch another child |

## 8. Observability

Structured logs and events use `same_lane_resume=true` and include only non-secret identifiers:

- request task ID and assignee;
- one-use resume authorization ID and terminal receipt disposition;
- expected/observed PR number, base, ref, and OID;
- expected/observed workspace branch and OIDs;
- selected substantive run ID/outcome;
- refusal reason code;
- child PID/start fingerprint/session only after handshake;
- launch/publication/termination phase;
- `resume_quarantined`, `spawn_failed`, and `cleanup_failed` task-level receipts with authorization ID and `run_id=null`;
- `spawn_failed` failure phase and full-tree-termination-confirmed flag; and
- `cleanup_failed` when liveness is uncertain.

Never include credentials, token values, full environment variables, or Finance business data. A successful readback must show one matching task/run/`spawned` identity, durable `resume_released` preceding release acknowledgement/heartbeat, and no pre-commit heartbeat. Existing event readers, exporters, hooks, dashboard serializers, and notification paths must be regression-tested with the nullable task-level quarantine/failure receipts before compatibility is claimed.

## 9. Compatibility and migration

- No SQLite schema migration.
- No configuration flag, environment variable, profile change, or service restart is introduced by the contract.
- Existing task/run/event readers remain compatible; new event payload keys are additive. Compatibility must be demonstrated for task-level `resume_quarantined`, `spawn_failed`, and `cleanup_failed` with nullable `run_id`, not assumed from schema shape alone.
- Existing ordinary dispatch, active-PR suppression, reclaim, and serial Node behavior remain unchanged.
- Legacy phantom runs are interpreted for admission only through the narrow substantive-outcome rule; they are not rewritten or deleted.
- The operation is internal/controller-only and opt-in by explicit typed request. It is not exposed as a general worker tool.

## 10. Safe verification plan

1. Run static checks and focused tests in the Hermes repository only; do not edit or run Finance application/OpenSpec/tests.
2. In an isolated temporary board, prove every guard, exact legacy history (`1144` followed by phantom reclaims `1146`/`1149`), same-authorization idempotency before/during/after compensation, explicit-new-authorization retry, handshake ordering, crash injection before and after durable child-identity recording, controller crashes before commit/after commit/after release send, post-commit gate failure, publication rollback, surviving-descendant cleanup failure, nullable-event compatibility, and unchanged ordinary active-PR suppression.
3. Use an inert handshake-capable child in integration tests. It must not enter a Finance worktree or perform external effects.
4. Before any exact live resume, require all other Flynn cards—including specification, implementation, and review cards—to be `done` or `archived`.
5. From a non-worker controller principal, re-run the approved read-only PR observer and local worktree checks under the operation's lease.
6. Invoke the exact immutable target request once. Read back the card, run, and events to prove one coherent PID/start/session/`spawned` identity and no earlier heartbeat.
7. The resumed worker remains constrained by its own remediation card. This verification itself does not mutate GitHub, start Finance application runtimes, deploy, access providers/customers, or interact with PR `#37`.
8. If any guard is false, stop with refusal evidence; do not create a replacement lane or repair state manually.
