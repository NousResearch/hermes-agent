# Hermes Mission Ownership — Phase 1 Integration and Recovery Review

## Candidate identity and scope

- Branch: `codex/mission-ownership-phase1`.
- The starting implementation commit `672658051d8e694804417c700c232ac27e81e61d` exists on this branch.
- Prior candidate head at the start of this review: `8fba08637115a4d585cc975d71d5222b2c14f37c`.
- Validated harness and probe implementation: `a52a6473b378b491121018c57121659d03f74d3c`.
- This review commit documents that evidence. The branch also retains the earlier Phase 1 candidate commits. No Phase 2 routing or budget features were added.
- The canonical `/home/ftr3d/X` checkout, live Hermes profiles, original Kanban database, running Fred/H.A.M. gateways, and their configuration were not modified or restarted. No production credential values were read or passed to the isolated processes.

## Original board: verified state and limits

The live `/home/ftr3d/.hermes/kanban.db` was opened only as a one-time read-only SQLite backup input. The isolated online backup contained seven user tables, seven tasks, eleven task runs, 63 task events, two task comments, and no mission tables. All existing table definitions and row fingerprints matched between the source snapshot, the preserved baseline copy, and the working copy. SQLite's `schema_version` counter differed after backup; it is a per-database cache counter, not a table or row difference.

The live database, WAL, and SHM SHA-256 values were the same before and after the backup and subsequent testing:

```text
kanban.db      81cfd6261a25c422004947eff5957434d084b582ca244ec3645d355f67aff930
kanban.db-wal  c0cf7050726ff68ac3fb4dd79e0d0ed8ee1f7312643cebd7374f23163f3a2154
kanban.db-shm  0397911fe68821a262d5495a50bd3ba3ba008a713b6174c1afad8a933f406bdb
```

The final baseline backup file SHA-256 was `7192c46e7cafbc6bf65f09695f92b240fb862a5a0ccaf4354a4f2d8b80c19c4d`. The live board still has no mission tables, as supported by the unchanged source hashes and the read-only backup fingerprint.

The earlier five empty mission tables reported as accidentally created and then removed are not reconstructible from available evidence. The October 15 snapshot predates that event, has schema version 1 and zero task rows, and cannot establish the exact board state immediately before the incident. It is therefore not evidence that no other transient state existed. A prior read-only inspection of that old snapshot created empty `-wal` and `-shm` sidecars; those exact inspection-created files were removed, and the snapshot's main-file bytes were unchanged. No such sidecar was created on the current board by this review.

## Test isolation

The reusable harness is `scripts/mission_phase1_isolation.py`. It refuses an unsandboxed run and checks the full set of runtime state-path variables, the profile config, root overlap, symlink escapes, and command arguments. `self-test` deliberately supplies live Hermes, board, and X paths to every declared runtime path variable and to command arguments; each is refused. It also refuses an external config and an isolation root overlapping the source or protected paths.

Runtime commands use bubblewrap with the host root mounted read-only and only the dedicated isolation root writable. `/tmp`, `/var/tmp`, and `/dev/shm` map into that root. The live Hermes, Codex, SSH, cloud, XDG credential/config directories, and canonical X checkout are covered by empty read-only overlays. The only host-data exception is the exact original Kanban database plus existing WAL/SHM files, mounted read-only into an internal path for the explicit backup command. Normal runtime commands cannot see those host contents. Runtime networking is isolated; only the PM dependency build opts into host networking.

The isolation proof passed after these protections were enabled. It confirmed the host Hermes and X directories were hidden, read-only, and unwritable; `/tmp` and `/var/tmp` writes resolved to the isolation root; and the runtime network namespace differed from the host. The initial PM build used the project's `pm.build_env` lock/setup with `dev` and `test` groups inside the sandbox and installed `python-dotenv` into the isolated virtualenv. No package was installed into the live Hermes environment.

The test runner itself writes a duration cache to the source checkout, so its full canonical wrapper cannot complete with the source checkout mounted read-only. A first canonical run also revealed that its workers use hard-coded `/var/tmp`; the harness now maps that path into the test root. Direct pytest runs below disable pytest's cache and pin `--basetemp` inside the isolation root. The source checkout is never writable to a test process.

## Mission and Kanban relationship

A mission is the coordination record for a root request and must exist before a dispatcher task is runnable. Native Kanban tasks remain the work queue and represent the root worker only after handoff; child tasks remain ordinary Kanban tasks. Keeping the root mission separate avoids making an admitted interactive request dispatchable before ownership transfer. The migration check added the five mission tables to a copy while preserving all seven pre-existing Kanban tables and rows.

The candidate couples root task creation/promotion and owner changes in transactions, but this proves consistency only through the candidate APIs. The child list remains an owner-supplied snapshot, direct SQL writers can bypass it, and unrecorded child work is not ruled out. This is application-level consistency, not a database-level complete ownership graph.

## Evidence

### Unit-tested

Under the isolated PM-built virtualenv:

```text
pytest mission_ownership_tests/test_mission_ownership.py -q -p no:cacheprovider \
  -k 'not test_valid_receipt_completes_and_requires_existing_code_artifact and not test_stale_artifact_hash_and_fake_empty_check_are_refused and not test_mission_completion_waits_for_recorded_native_child_tasks and not test_foreground_background_handoff_and_completion_guard_use_native_kanban'
26 passed, 4 deselected in 3.08s
```

The four excluded cases were run by the full file and failed. They pass an unchanged, already-committed test file as a validation artifact. `mission_validation.validate` rejects it because its bytes equal `HEAD`, so the cases cannot test completion from a clean committed checkout. This is a candidate limitation, not a passing completion test. The other unit cases cover delivery identity, owner fencing, lease blocking and cleanup, abrupt worker exit, recovery generation, completion evidence rejection, and Kanban coupling.

The canonical `scripts/run_tests.sh` ran 30 tests: 26 passed and those same 4 failed. A focused canonical-runner selection reported four tests passed, then the runner exited nonzero while trying to write `test_durations.json` into the read-only source checkout. The direct pytest command above is the successful hermetic result; it is not represented as a successful canonical-runner exit.

Ruff passed for all six new harness/probe scripts. `git diff --check` passed before the harness commit.

### Integration-tested in an isolated profile and gateway

- A real Hermes gateway started with an isolated Fred profile, no messaging tokens or provider credentials, `gateway.standalone: true`, and the isolated dispatcher configured on. It logged `No messaging platforms enabled`, did not connect to a platform or model provider, and was stopped by a bounded timeout.
- In an earlier run with an isolated ready task, the embedded dispatcher claimed the task and launched a worker. On gateway shutdown the worker PID disappeared while the task/run remained `running`; the isolated database accurately exposed that stale state. The same exact gateway command did not use the original board or original profile.
- The recovery probe verified the PID was no longer alive, then explicitly recovered after the owner lease time. It moved the mission to generation 3, changed the task to `blocked`, marked the task run `crashed`, and transferred ownership to the recovery session. There is no automatic takeover based only on elapsed time.
- The admission probe invoked the real gateway `run_post_admission_hook` and discovered plugin hook registry with synthetic Telegram-shaped events. A repeated delivery ID produced one mission for that session. Identical wording under a separate delivery and session produced a second mission. This exercised the gateway consumer and plugin hook, but not Telegram polling or a live Telegram account.
- The old installed plugin files were copied read-only into the isolated profile using the normalized `mission_admission` directory. Plugin discovery selected legacy user version 0.1.0; replacing that isolated user plugin with candidate version 0.2.0 selected the candidate. The real admission probe then passed. The installed legacy directory is named `mission-admission` with a hyphen; plugin discovery warns and ignores that directory because the package key expects `mission_admission`. Staging migration must explicitly remove or rename that legacy directory and verify the selected version. No live profile was changed.
- The copied-board migration probe added mission schema to the disposable working copy, verified every original user-table schema and row fingerprint stayed the same, and confirmed the seven legacy tasks remained queryable. It added no mission rows. Rollback evidence is limited to additive compatibility: legacy Kanban tables remain readable and mission tables are retained. No full old-binary rollback, restore, or newer-state rollback drill was performed.

### Controlled subprocess and recovery-tested

`scripts/mission_phase1_detached_probe.py` launched a real isolated detached child from inside an execution scope. After the wrapped handler returned, the lease count was zero, the child was still alive, and handoff committed. The child wrote its marker after the ownership generation changed. This is a reproduced unsafe handoff, not a theoretical risk.

Focused unit recovery cases passed for normal drain, interruption, hard worker exit, live worker refusal, and dead worker recovery. The isolated gateway shutdown also left a dead worker PID and running task state; explicit recovery fenced and parked it. Gateway restart during the middle of the SQLite transfer transaction, arbitrary plugin subprocesses, and external remote operations were not exhaustively tested.

### Not verified

- Real Telegram, CLI interactive, desktop, and dashboard/web admission routes were not run end to end. Telegram-shaped events were simulated through the actual gateway consumer. No production credentials or accounts were available or used.
- Actual model/API tool calls, terminal/code execution backends, inline registry tools, delegated workers, nested/proxied tools, browser automation, and plugin failures that create work outside the shared executor were not exercised end to end.
- Other live profiles were not started or altered. The isolated gateway reported itself standalone and stated that `default` was not served; this is not a coexistence test with Fred and H.A.M. profiles.
- Effective host-level dispatch was observed in the earlier isolated gateway run that claimed the ready task. The configured false case was observed in a prior isolated run. No host service manager or live dispatcher was queried or changed.
- No full Hermes suite, real transport, real API completion, full rollback, or production overhead benchmark was run.

## Enforcement, advisory behavior, and remaining blockers

### Genuinely enforced within candidate application paths

- Message delivery identifiers deduplicate identical deliveries; prompt similarity does not attach a new session to an existing mission. With no transport ID, the fallback is scoped to the specific session/turn and cannot reliably identify a replay.
- Handoff fences later dispatches routed through the shared executor. An active execution lease or recorded running child blocks transfer. Hard process exit leaves a durable lease that fails closed rather than expiring automatically.
- Task-owner completion requires a trusted stored receipt bound to mission generation, contract, validator, checkout revision, artifact hashes, and check results. Model-provided command status is not accepted. This check remains application-level, and current clean-commit artifact behavior failed four tests as described above.
- Candidate APIs transactionally couple root handoff and task state changes. Additive schema migration preserved the tested copy's existing rows.

### Application-level only or unsafe

- **Detached/outliving work bypasses the lease.** A subprocess or remote operation can survive handler return, release the lease, and continue after successful handoff. This was reproduced. No reliable universal cancellation/join boundary currently wraps terminal, browser, connector, delegated, and plugin effects. Strong exclusivity needs backend-specific process ownership and wait/cancel semantics, likely a stronger OS process boundary. Do not claim exclusive ownership or stage this candidate as a safety fence.
- **Crash leases fail closed but cannot self-heal.** A hard worker crash leaves a lease that blocks transfer/recovery until an operator resolves it. Automatically deleting it based on age is unsafe. Current recovery also waits for owner lease expiry and verifies the recorded worker is not alive before moving a task owner.
- **Unwrapped work remains outside the fence.** Direct SQL, other Hermes builds, nested workers, and paths that bypass the shared executor can mutate without a mission lease.
- **Completion verifier has an artifact policy gap.** Clean committed artifacts are rejected by the existing changed-from-`HEAD` rule in four tests; reviewer identity is recorded by the application and is not independently authenticated.
- **Old plugin path needs a migration action.** The hyphenated installed directory is ignored by plugin discovery; staging must explicitly remove/rename it and verify the normalized candidate user/bundled version wins.
- **Native child graph is incomplete.** Only recorded child IDs are checked; direct DB/API bypasses or unrecorded descendants can diverge.

## Runtime overhead

No production benchmark was run. The ownership path adds SQLite lease acquire/release transactions around covered tool dispatches and indexed owner/child reads. The trusted validator adds Python parse and Ruff subprocess work only when completion validation is requested. Isolation/test harness overhead is test-only. Claims of specific latency percentages are unsupported.

## Migration and rollback readiness

The additive schema migration is validated on a read-only backup and separate disposable working copy. Existing table definitions/rows matched before and after migration. Legacy Kanban reads remained available while new mission tables were retained. The original database was not migrated. A deployable backup/restore procedure, full old-code rollback, recovery from newer writes during rollback, and multiple-profile migration are still unverified. Do not use this result as migration approval.

## Deployment recommendation

**Blocked; continue hardening. Do not proceed to supervised staging yet.** The candidate cannot guarantee single active mutation when a real detached operation survives handler return, crash leases have no automated safe recovery, and completion validation has four clean-checkout test failures. Real Telegram/CLI/desktop/dashboard routes, backend subprocess trees, and complete rollback remain unverified. Address the bounded process ownership/cancellation contract and completion artifact policy, then rerun the same isolated harness before asking for a staging review.

## Git and worktree state

- Implementation commit: `a52a6473b378b491121018c57121659d03f74d3c` (`Add isolated Phase 1 integration harness`).
- This report is committed after the implementation commit above; it contains evidence for that exact code revision. The final branch head is the following documentation-only commit.
- Branch: `codex/mission-ownership-phase1`.
- The starting candidate commits remain in history. Git status is clean after the implementation and documentation commits.
