# Kanban guarded same-lane resume

## ADDED Requirements

### Requirement: Explicit exact-card recovery operation

The controller SHALL expose a controller-only operation authorized by this change only for task `t_dfa23a41` remediating PR `#155`. Each explicit invocation SHALL carry a controller-issued one-use resume authorization ID. A different task or PR SHALL refuse even if generic predicates match; widening authorization requires a separate reviewed change. The operation SHALL NOT be called by ordinary dispatcher scanning and SHALL NOT create or substitute a task, worktree, branch, or PR.

#### Scenario: exact existing card is selected

- GIVEN an immutable request naming one existing task, assignee, PR identity, worktree, upstream, and required substantive review outcome
- WHEN the controller evaluates same-lane resume
- THEN every observed identity SHALL match the request exactly
- AND the operation SHALL apply only to that task

#### Scenario: another task owns the PR

- GIVEN the expected PR is durably associated with a different task
- WHEN same-lane resume is evaluated for the target
- THEN the operation SHALL refuse before launch
- AND no task or run state SHALL change

#### Scenario: a different task and PR otherwise satisfy generic guards

- GIVEN a request does not identify `t_dfa23a41` and PR `#155`
- WHEN same-lane resume is evaluated
- THEN the operation SHALL refuse before launch

### Requirement: Ordinary active-PR suppression remains universal

The existing ordinary active-PR duplicate-lane guard SHALL remain unchanged for every task that is not the exact explicit same-lane request. The new operation SHALL NOT weaken, bypass, or reinterpret ordinary suppression.

#### Scenario: unrelated active PR remains suppressed

- GIVEN any other ready task has an active PR
- WHEN the ordinary dispatcher evaluates it
- THEN it SHALL remain guarded with the existing `active_pr` behavior
- AND no same-lane exception SHALL be inferred

#### Scenario: target is seen by ordinary scanning

- GIVEN the exact target is READY and its PR is active
- WHEN ordinary dispatcher scanning evaluates it without an explicit request
- THEN the existing active-PR guard SHALL suppress it

### Requirement: All admission guards hold under one claim lease

The controller SHALL acquire a board-and-task-scoped resume/claim lease before authoritative admission and SHALL retain it through initial validation, durable quarantine, child handshake, complete guard revalidation, publication or rollback, cleanup, and compensation. Board transitions SHALL use short explicit SQLite transactions under that operation lease. A lease timeout SHALL refuse or retry; it SHALL NOT fall back to unlocked or direct-SQLite mutation.

#### Scenario: target state is eligible

- GIVEN the exact task is READY
- AND claim, current-run, PID/start, and session fields are empty
- AND its assignee/workspace/branch metadata match the request
- WHEN admission is evaluated under the lease
- THEN card-state validation SHALL pass

#### Scenario: target is stale or claimed

- GIVEN any required task field is stale, missing, claimed, or not READY
- WHEN admission is evaluated
- THEN the operation SHALL refuse before launch

### Requirement: Isolated terminal lifecycle residue is normalized atomically

For the exact authorized same-lane request only, board-kernel lifecycle code MAY clear a non-null `worker_started_at` value before admission when the target is READY and has no claim lock/expiry, current run, open run row, worker PID, or session. Clearing the residue and appending task-level `lifecycle_normalized` evidence SHALL occur in one transaction. The evidence SHALL identify the field, prior value, reason `terminal_residue`, authorization ID, and same-lane scope. No operator script SHALL directly update the board database.

#### Scenario: stale fingerprint is the only lifecycle residue

- GIVEN the exact target is READY
- AND claim/current-run/PID/session fields are null
- AND no task run remains open
- AND only `worker_started_at` remains from a terminal prior attempt
- WHEN the authorized same-lane operation begins
- THEN the kernel SHALL atomically clear the fingerprint and append `lifecycle_normalized`
- AND complete admission SHALL run afterward

#### Scenario: an active lifecycle handle remains

- GIVEN `worker_started_at` is non-null
- AND any claim field, current/open run, worker PID, session, or non-READY status remains
- WHEN normalization is evaluated
- THEN normalization SHALL refuse without clearing any field
- AND no child SHALL launch

#### Scenario: heartbeat-only terminal history exists

- GIVEN later claimed/heartbeat-only runs are terminal and have `ended_at`
- AND no open run or active task handle remains
- WHEN isolated fingerprint residue is normalized
- THEN terminal history SHALL remain unchanged
- AND no phantom run or heartbeat SHALL be created

### Requirement: Eligible target is durably quarantined before launch

After every initial guard passes, one transaction SHALL change the exact card from READY/unclaimed to existing status BLOCKED with kind/reason `resume_quarantined`, keep claim/run/PID/start/session fields empty, and append task-level `resume_quarantined` containing the one-use authorization ID and `run_id = null`. The operation lease SHALL remain held. Immediately before publication, every mandatory guard SHALL be revalidated while the child remains gated.

#### Scenario: admission enters quarantine

- GIVEN every initial guard passes
- WHEN the admission transaction commits
- THEN the target SHALL be BLOCKED/`resume_quarantined` with no claim, run, PID, start fingerprint, or session
- AND ordinary dispatch SHALL not consider it runnable

#### Scenario: controller crashes before child creation

- GIVEN the target has a committed matching quarantine receipt
- WHEN the controller dies before launch
- THEN the target SHALL remain non-runnable
- AND recovery SHALL either continue the same authorization or atomically restore READY with `spawn_failed` after proving no child exists

#### Scenario: a guard drifts after quarantine

- GIVEN a gated child exists
- AND any mandatory card, lane, PR, worktree, upstream, or substantive-history guard no longer matches at publication revalidation
- WHEN publication is evaluated
- THEN no run/claim/`spawned` SHALL be published
- AND compensation SHALL proceed from quarantine without releasing the child

### Requirement: Strict serial Flynn lane

The operation SHALL refuse while any other task assigned to Flynn is nonterminal. Nonterminal SHALL include TODO, READY, RUNNING, BLOCKED, and REVIEW; only DONE and ARCHIVED SHALL be terminal for this guard. The existing serial Node posture SHALL remain unchanged.

#### Scenario: competing TODO card exists

- GIVEN the target is otherwise eligible
- AND another Flynn task is TODO
- WHEN admission is evaluated
- THEN the operation SHALL refuse before launch

#### Scenario: competing active card exists

- GIVEN another Flynn task is READY, RUNNING, BLOCKED, or REVIEW
- WHEN admission is evaluated
- THEN the operation SHALL refuse before launch

#### Scenario: only terminal sibling cards exist

- GIVEN every other Flynn task is DONE or ARCHIVED
- WHEN admission is evaluated
- THEN the serial-lane guard SHALL not by itself refuse the target

### Requirement: Same-card active-PR ownership

The controller SHALL verify that the exact active PR is durably associated with the target card, that no different nonterminal task owns the PR, and that no conflicting active PR is associated with the target.

#### Scenario: ownership matches

- GIVEN the target's durable board evidence identifies the exact expected PR URL
- AND no other nonterminal task identifies that PR
- WHEN admission is evaluated
- THEN PR ownership SHALL pass

#### Scenario: ownership is ambiguous or conflicting

- GIVEN PR ownership is missing, duplicated, or conflicts with another PR/task association
- WHEN admission is evaluated
- THEN the operation SHALL refuse before launch

### Requirement: Exact remote PR guard

While the claim lease is held, the approved read-only observer SHALL return typed evidence for the expected PR. State SHALL be OPEN, base SHALL be `dev`, head ref SHALL be `wt/finance-pr153-producer-output-closure`, and head OID SHALL be `1b910d4ff9d1c9fe103da5df84d302ba81fc2cc1`. The observed URL/number SHALL match the target's ownership evidence.

#### Scenario: exact PR remains open

- GIVEN typed observer evidence matches every expected PR field
- WHEN admission is evaluated
- THEN the remote PR guard SHALL pass

#### Scenario: PR identity or state changed

- GIVEN the PR is not OPEN or its URL, number, base, head ref, or head OID differs
- WHEN admission is evaluated
- THEN the operation SHALL refuse before launch

#### Scenario: observer fails

- GIVEN the observer times out, errors, or returns incomplete evidence
- WHEN admission is evaluated
- THEN the operation SHALL refuse before launch
- AND SHALL NOT treat cached Boolean success as authoritative

### Requirement: Exact clean worktree and upstream guard

While the claim lease is held, the controller SHALL verify that the expected preserved worktree is registered, on the expected branch, clean in index/tracked/untracked state, and at the expected local OID; its configured expected upstream SHALL resolve to the same OID.

#### Scenario: preserved worktree matches

- GIVEN workspace path, branch, cleanliness, local OID, upstream ref, and upstream OID all match
- WHEN admission is evaluated
- THEN the worktree guard SHALL pass

#### Scenario: worktree or upstream drifted

- GIVEN the worktree is missing, detached, dirty, on another branch, or has a local/upstream/ref/OID mismatch
- WHEN admission is evaluated
- THEN the operation SHALL refuse before launch

### Requirement: Latest substantive outcome selects the reviewed remediation point

The controller SHALL scan terminal runs newest-first and select the first substantive run. It MAY skip a run only when its outcome is `reclaimed` or `spawn_failed`, PID/start/session fields are absent, no `spawned` or gate-release event exists, no worker summary/result/artifact exists, and every event is controller housekeeping (`claimed`, `heartbeat`, `reclaimed`, or `spawn_failed`). The selected run for the exact target SHALL be run `1144`, attributed to profile/reviewer `crash`, with outcome and durable review event `changes_requested`.

#### Scenario: phantom reclaims follow requested changes

- GIVEN run `1144` ended `changes_requested`
- AND later runs `1146` and `1149` are reclaimed claimed/heartbeat-only attempts with no spawned-worker evidence
- WHEN the latest substantive outcome is selected
- THEN run `1144` SHALL be selected

#### Scenario: later genuine worker outcome exists

- GIVEN a later terminal run has spawned-worker evidence or a worker-produced outcome
- WHEN substantive history is selected
- THEN that later run SHALL be substantive
- AND the request requiring run `1144` SHALL be refused

#### Scenario: required review outcome differs

- GIVEN the selected substantive run ID or outcome is not exactly `1144` / `changes_requested`
- WHEN admission is evaluated
- THEN the operation SHALL refuse before launch

### Requirement: Real gated child identity precedes durable run publication

The initial live operation SHALL be Linux-only. The launcher SHALL create a hardened wrapper behind a start gate. Before process creation, the task-scoped resume fence SHALL durably record controller owner PID/start, unique launch nonce, `launching` phase, and a finite gate deadline. Before imports/threads/descendants/work, the wrapper SHALL install `PR_SET_PDEATHSIG`, recheck the expected controller PID/start after installation, and exit on mismatch. Before any claim/run handle, `spawned` event, or heartbeat is persisted, the wrapper SHALL atomically persist and fsync nonce-bound PID, canonical process-start fingerprint, and process-group identity to the fence; then it SHALL report those values plus a unique child-confirmed session ID and gated-ready state. It SHALL NOT create descendants before that identity is durable and validated. Unsupported hosts SHALL refuse before quarantine/process creation.

#### Scenario: valid handshake

- GIVEN every admission guard passes
- WHEN the child reports a valid PID/start/session while still gated
- THEN publication MAY proceed
- AND the child SHALL NOT enter its work loop before commit

#### Scenario: child identity is invalid

- GIVEN the child returns no PID, a stale PID/start pair, no session, a duplicate session, or exits during handshake
- WHEN launch is evaluated
- THEN publication SHALL NOT occur
- AND compensation SHALL run

#### Scenario: controller crashes before child identity is recorded

- GIVEN a durable `launching` fence record exists
- AND the process was created but no PID/start/process-group identity was persisted
- WHEN the controller dies
- THEN the Linux parent-death signal or post-install expected-parent mismatch SHALL terminate the wrapper before work
- AND no descendant SHALL exist because descendant creation was not yet permitted
- AND recovery SHALL prove the recorded controller owner is dead and wait beyond the gate deadline before compensation or retry

#### Scenario: controller crashes after child identity is recorded

- GIVEN the fence durably identifies the child PID/start/process group
- WHEN the controller dies before publication
- THEN recovery SHALL reconcile and terminate that exact identity before compensation or retry

#### Scenario: required host primitive is unavailable

- GIVEN the host cannot provide the required Linux parent-death contract
- WHEN same-lane resume is invoked
- THEN the operation SHALL refuse before quarantine or process creation
- AND SHALL NOT use a weaker fallback

#### Scenario: heartbeat attempts before release

- GIVEN the child remains behind the gate
- WHEN it attempts to heartbeat or enter the worker loop
- THEN the attempt SHALL be prevented
- AND no heartbeat SHALL be durable before publication commits

### Requirement: Successful publication is atomic

After a valid gated handshake and complete guard revalidation, one transaction SHALL transition the exact card from its matching quarantine to RUNNING, open its run, persist the same PID/start/session identity on task/run records, and append matching `claimed` and `spawned` events including the authorization ID. The controller SHALL commit once and only then release the child gate. After receiving the commit token, the child SHALL read back the committed run and atomically persist `resume_released` before acknowledging release, entering the worker loop, or heartbeating.

#### Scenario: publication commits

- GIVEN a valid request and gated child handshake
- WHEN publication succeeds
- THEN task, run, claim, PID/start/session, `claimed`, and `spawned` evidence SHALL become visible together
- AND the released child SHALL read back the same run identity and persist `resume_released` before acknowledgement or heartbeat

#### Scenario: concurrent attempts

- GIVEN two controllers attempt the exact request with the same one-use authorization ID
- WHEN the task fence serializes them before, during, or after compensation/publication
- THEN at most one SHALL launch a child
- AND the other SHALL wait/fail fast or return the durable terminal receipt without launching

#### Scenario: retry after compensated failure

- GIVEN an authorization ID has a terminal `compensated` receipt
- WHEN a controller retries with that same ID
- THEN it SHALL return the recorded disposition without launching
- AND a new launch SHALL require a newly issued authorization ID plus a fresh complete guard evaluation

### Requirement: Failed launch or publication leaves no phantom run

If launch/handshake or run-handle/event publication fails, the controller SHALL keep or roll back to the already committed quarantine, prevent gate release, terminate the complete partial-child process group/tree, and confirm the PID/start leader is no longer live and no group/tree member or descendant remains. Only then SHALL one transaction change the same card from quarantine to READY/unclaimed with worker/run/session fields clear and append task-level `spawn_failed` containing the authorization ID and `run_id = null`. The operation fence SHALL remain held through that commit. No claimed-only run, heartbeat-only run, or orphan leader/descendant execution SHALL remain.

#### Scenario: launch fails before a PID exists

- GIVEN all guards pass
- WHEN child launch fails before a PID is obtained
- THEN the target SHALL remain quarantined until cleanup proves no child exists
- AND `spawn_failed` SHALL identify the launch phase
- AND one compensation transaction SHALL restore READY/unclaimed with no run

#### Scenario: persistence fails after process creation

- GIVEN the child completed a gated handshake
- WHEN claim/run/handle/`spawned` persistence fails
- THEN the publication transaction SHALL roll back to committed quarantine
- AND the child SHALL be terminated and confirmed dead
- AND a separate compensation transaction SHALL atomically restore READY/unclaimed plus `spawn_failed`
- AND no run or heartbeat SHALL remain

#### Scenario: cleanup cannot be confirmed

- GIVEN a partial child may still be live
- WHEN termination cannot be confirmed
- THEN the card SHALL remain BLOCKED/`resume_quarantined`
- AND task-level `cleanup_failed` with the authorization ID SHALL commit without changing it to READY
- AND READY/unclaimed compensation SHALL complete only after liveness is conclusively absent

#### Scenario: leader exits but a descendant survives

- GIVEN the recorded leader PID/start identity is dead
- AND a process-group/tree member or descendant remains live
- WHEN cleanup evaluates liveness
- THEN cleanup SHALL continue escalation or fail closed
- AND the card SHALL NOT become READY and no retry SHALL launch until full group/tree extinction is proven

#### Scenario: a contender arrives during compensation

- GIVEN publication rolled back and the first child is still being terminated
- WHEN another controller attempts the same resume
- THEN the task-scoped resume fence SHALL prevent a second launch
- AND the fence SHALL release only after cleanup and compensation complete

#### Scenario: a contender arrives after compensation with the same authorization

- GIVEN the first attempt restored READY and retained a terminal authorization receipt
- WHEN a contender presents the same authorization ID
- THEN it SHALL return the recorded disposition without launching a sequential child

### Requirement: Controller crash and gate-release recovery are fail-closed

The start gate SHALL terminate the child on channel EOF, parent death, deadline expiry, or an invalid token. The resume fence SHALL have bounded owner-death recovery and crash-surviving launch nonce/phase/deadline/PID/start/process-group metadata. A crash before commit SHALL leave no published run; a crash after commit but before gate release SHALL be recoverable as a real spawned failure with no heartbeat. Release acknowledgement SHALL never precede durable `resume_released` persistence.

#### Scenario: controller crashes before publication commit

- GIVEN a gated child exists and publication is uncommitted
- WHEN the controller dies
- THEN gate EOF or deadline SHALL terminate the child before work
- AND fence recovery SHALL confirm death and retain task-level `spawn_failed`
- AND no run or heartbeat SHALL remain

#### Scenario: controller crashes after commit before gate release

- GIVEN task/run/PID/start/session/`spawned` committed
- AND no release acknowledgement or heartbeat exists
- WHEN the controller dies and the gated child exits
- THEN reclaim SHALL end the real run as `spawn_failed`
- AND SHALL restore READY/unclaimed only after liveness is absent

#### Scenario: gate release acknowledgement fails

- GIVEN publication committed but release cannot be acknowledged
- WHEN recovery runs
- THEN the controller SHALL terminate the child and end the published run explicitly
- AND SHALL NOT fabricate a heartbeat or erase real `spawned` evidence

#### Scenario: controller crashes after sending release

- GIVEN publication committed and the controller sent the commit token
- WHEN the controller dies before receiving acknowledgement
- THEN the child SHALL either persist `resume_released` before work or exit without work
- AND recovery SHALL use durable `resume_released` to choose the normal-lifecycle or spawn-failure branch

### Requirement: Failure and refusal are observable without sensitive data

The controller SHALL record structured phase/reason telemetry sufficient to distinguish guard refusal, observer failure, launch failure, publication failure, termination success, and cleanup uncertainty. Telemetry SHALL NOT contain credentials, full environments, or Finance business data.

#### Scenario: guard refusal is diagnosed

- GIVEN one admission guard fails
- WHEN the operation returns refusal
- THEN logs or bounded task evidence SHALL identify the failed guard and expected/observed non-secret identity
- AND no child SHALL start

#### Scenario: compensated failure is diagnosed

- GIVEN launch or publication fails
- WHEN compensation completes
- THEN `spawn_failed` SHALL include the failure phase and termination-confirmed state
- AND SHALL not reference a phantom run

#### Scenario: nullable task-level failure event is consumed

- GIVEN `resume_quarantined`, `spawn_failed`, or `cleanup_failed` is stored with `run_id = null`
- WHEN existing readers, exporters, hooks, serializers, and notification paths consume it
- THEN each SHALL handle the event without error or misattributing it to a run

### Requirement: No migration, configuration, Finance, or external side effects

The capability SHALL require no database migration, configuration/environment change, profile/gateway/cron/systemd/host/network/Vault change, deployment, GitHub mutation, Finance application/OpenSpec/test change, Finance runtime/provider/customer action, business-data mutation, replacement card/worktree/PR, direct SQLite repair, or PR `#37` interaction.

#### Scenario: implementation and verification remain bounded

- GIVEN the package is implemented and tested
- WHEN its diff and verification effects are inspected
- THEN only approved Hermes controller/kernel code, focused Hermes tests, and this package SHALL change
- AND all prohibited surfaces SHALL remain untouched
