# Pi Harness Upstream Reconciliation

## Decision

Option B is the sole durable target: Hermes Kanban gains a deterministic native
`HarnessExecutor` / `HarnessDriver` seam. Executor identity is separate from
human assignee and profile ownership. The dispatcher and Kanban lifecycle
kernel remain the only components allowed to claim, finalize, and accept a
task. Each attempt also has one immutable invocation identity covering the
exact prompt/instruction bytes and ordered deterministic launch inputs supplied
to Pi. The executor invokes a driver and returns structured evidence; neither
a harness process nor a model worker receives a terminal Kanban mutation
surface.

This rejects an external watcher, fake profile, nested Hermes worker,
DB-writing bridge, worker-owned completion or acceptance, and any second
lifecycle ledger. Legacy profile spawning remains unchanged for every
non-harness card.

The frozen source identities are:

- pinned main: `26480e6c57c3558442a73c2dffe313996b19417f`;
- PR #67718: `6e682eaefdb1147e7e64a255483967e81460e658`,
  measured at 3 commits ahead of and 17 commits behind pinned main;
- PR #63297: `ac6550e487e88e17a59a8454720eecacbd341709`,
  measured at 1 commit ahead of and 1101 commits behind pinned main.

These are evidence inputs, not merge targets. The recommendation is to extract
the classified invariants onto pinned/current main. Do not wait for either PR
and do not merge or rebase either PR wholesale.

## Primitive reconciliation

Each primitive has exactly one disposition. “Reuse unchanged” means reuse the
invariant and its tests in a native contract, not import the PR's external
worker API or claim/finalization ownership.

| Primitive | Classification | Ownership rationale and exact-source evidence | Required test evidence to port or preserve |
|---|---|---|---|
| Immutable spec identity | **reuse unchanged** | Preserve `(attachment_id, SHA-256(exact raw bytes), schema_version)` and re-read/rehash before execution. It is Kanban-owned input identity, not an external submission protocol. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:SpecIdentity/L327-L339`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:submit/L1387-L1535`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:read_submitted_attachment/L1538-L1597` | Preserve candidate-race rollback and raw-byte tamper detection: `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_submit_expected_hash_rejects_candidate_mutation_atomically/L258-L300`, `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_read_submitted_attachment_detects_byte_mutation/L443-L471`. |
| Strict JSON | **reuse unchanged** | Preserve bytes-only strict UTF-8 decoding, rejection of BOM, duplicate keys, NaN and trailing data, exact key sets, and integer-token typing. The canonical native TaskSpec/Result schemas replace external-worker-specific envelopes. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:_decode_strict_json/L577-L630`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:_validate_taskspec/L678-L734` | Port the negative parser matrix: `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_strict_json_parser_negatives/L479-L516`. |
| Lease identity | **adapt into native `HarnessExecutor`** | Replace the bearer lease with a trusted, Kanban-issued harness attempt identity containing task id, run id, immutable spec identity, attempt/generation, and executor identity. Validate the complete tuple on every write; do not accept worker-asserted state or secrets. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:Lease/L342-L363`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:_check_lease_identity/L1056-L1111`, `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:claim_task/L3484-L3603` | Adapt stale-token/state/expiry and heartbeat identity coverage, including `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_heartbeat_returns_updated_lease/L740-L795`; add native stale-run and executor/profile-separation assertions. |
| Process binding | **adapt into native `HarnessExecutor`** | The native driver records identity for the child it actually spawned before work is enabled. Keep same-identity replay and reject divergent binding; never accept a supervisor-supplied process claim. Existing native PID/run binding remains the lifecycle foundation. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:BoundProcess/L367-L389`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:bind_process/L1816-L1907`, `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:_set_worker_pid/L7204-L7222` | Adapt durable/lost-response replay coverage from `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_bind_process_records_durable_bound_substate/L676-L718`; add process-absence-before-release coverage. |
| Result staging/finalization | **adapt into native `HarnessExecutor`** | Preserve exact result bytes, module-computed hash, stage-before-commit, complete-tuple replay, and divergent-result rejection. The driver only returns/stages evidence. Hermes lifecycle code alone applies `COMPLETE`, `REQUEUE`, or `BLOCK` using native completion/block semantics. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:put_result/L2186-L2288`, `ac6550e487e88e17a59a8454720eecacbd341709:hermes_cli/kanban_db.py:terminal_handoff/L3004-L3120`, `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:complete_task/L4164-L4300` | Port exact-byte, stage-without-finalize, identical replay, divergent replay, rollback, concurrency, and post-commit recovery tests. Starting evidence: `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_finalize_complete_persists_exact_result_bytes/L1183-L1204`, `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_put_result_persists_without_finalizing/L1398-L1449`, `ac6550e487e88e17a59a8454720eecacbd341709:tests/hermes_cli/test_kanban_terminal_handoff.py:L41-L130`. |
| Artifacts | **adapt into native `HarnessExecutor`** | Preserve safe names, size bounds, SHA-256 over exact bytes, `(run,name)` idempotency, divergent-byte rejection, and a pre-commit check that every result-declared artifact is durably present and byte-identical. Use a native run-owned attachment/repository path, not the external lease API/table; pinned main stores attachments by path and already has completion artifact behavior that must not be bypassed. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:_validate_result_artifacts/L1170-L1242`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:put_artifact/L2780-L2870`, `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:complete_task/L4134-L4219` | Port byte-collision and declared-durability tests: `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_put_artifact_idempotent_for_same_name_same_bytes/L1709-L1778`, `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_finalize_rejects_declared_artifact_that_is_not_durable/L2127-L2173`; add crash/orphan cleanup tests. |
| Recovery | **keep only for external workers** | PR #67718's public `list_active`, caller-supplied absence/no-start proofs, recovery holds, and two-hold ceiling solve out-of-process supervisor uncertainty. Native recovery stays in Hermes' stale-claim, maximum-runtime, stale-running, and crashed-worker paths. The native driver must reap/confirm process absence, but must not import this public recovery state machine. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:recover_expired/L2476-L2690`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:hold_for_recovery/L1994-L2123`, `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:release_stale_claims/L3712-L3855` | Keep PR recovery tests with the external design only. Native HARN tests instead preserve Hermes-owned crash, stale, runtime, restart, and live-process deferral behavior from `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:_pid_alive,_terminate_reclaimed_worker,_defer_reclaim_for_live_worker/L6186-L6366`, tested at `26480e6c57c3558442a73c2dffe313996b19417f:tests/hermes_cli/test_kanban_db.py:test_stale_claim_with_live_pid_extends_instead_of_reclaiming,test_stale_claim_deferred_when_live_worker_survives_termination,test_stale_claim_reclaimed_when_termination_succeeds/L456-L612`. |
| Atomic handoff | **reject because it would introduce Option C or duplicate lifecycle ownership** | PR #67718's `claim_external()` and `finalize()` let an external actor enter and terminate lifecycle state. PR #63297's DB-writing CLI/model/MCP handoff similarly lets the worker accept itself. Preserve atomicity only inside the existing native claim and kernel-owned terminal transaction. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:claim_external/L1645-L1813`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:finalize/L2291-L2468`, `ac6550e487e88e17a59a8454720eecacbd341709:hermes_cli/kanban.py:_cmd_handoff/L1887-L1910`, `ac6550e487e88e17a59a8454720eecacbd341709:tools/kanban_tools.py:_handle_handoff/L834-L893` | Do not port API tests as native acceptance tests. Add negative tests proving the harness process has no terminal tools and cannot mutate task/run/event terminal state. Existing PR bundle/replay tests are evidence for the internal finalizer only: `ac6550e487e88e17a59a8454720eecacbd341709:tests/hermes_cli/test_kanban_terminal_handoff.py:L41-L130`. |
| Dispatcher filtering | **keep only for external workers** | Filtering `external_spec_hash IS NULL` reserves cards for an external puller. A native harness card must stay visible to the Hermes dispatcher, which selects an executor discriminator and performs the ordinary native claim. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_db.py:has_spawnable_ready/L7869-L7898`, `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_db.py:_dispatch_once_locked/L8078-L8083` | Keep the external exclusion test only with that external feature: `6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py:test_dispatcher_does_not_spawn_external_ready_task/L1679-L1691`. Native tests must instead prove harness selection plus unchanged legacy profile selection. |

## Ownership and lifecycle boundary

Pinned main already creates claim state, run state, the task run pointer, and a
run-scoped event atomically under `BEGIN IMMEDIATE`; review claims use the same
ownership model. Completion is likewise a Kanban lifecycle operation.
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:claim_task,claim_review_task/L3484-L3678`,
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:complete_task/L4164-L4261`.
The dashboard rejects direct entry into `running`, consistent with this kernel
ownership. `26480e6c57c3558442a73c2dffe313996b19417f:plugins/kanban/dashboard/plugin_api.py:update_task/L840-L895`.

Accordingly:

- the dispatcher identifies a harness card, claims it through the native path,
  and creates a fresh run/attempt;
- `HarnessExecutor` receives trusted identity and invokes the selected
  `HarnessDriver`;
- the driver spawns and observes its own child and returns exact evidence; it
  does not own Kanban claim, acceptance, or terminal state;
- a Hermes/Kanban finalizer verifies attempt, process, hashes, result, and
  artifacts and alone commits task/run/event lifecycle state;
- assignee/profile remains human or routing metadata; executor identity is an
  independent field/value and cannot be encoded as a fake profile.

PR #63297's `kanban_handoff` model tool, CLI bridge, MCP/ACP exposure, and
goal-mode judge are prohibited surfaces, not shortcuts. In particular, the
goal judge is nondeterministic and fails open on judge failure.
`ac6550e487e88e17a59a8454720eecacbd341709:tools/kanban_tools.py:_check_kanban_mode,_handle_handoff/L65-L79,L834-L893`,
`ac6550e487e88e17a59a8454720eecacbd341709:toolsets.py:_HERMES_CORE_TOOLS,TOOLSETS/L70-L77,L260-L275`,
`ac6550e487e88e17a59a8454720eecacbd341709:agent/transports/hermes_tools_mcp_server.py:EXPOSED_TOOLS/L86-L105`,
`ac6550e487e88e17a59a8454720eecacbd341709:acp_adapter/tools.py:_POLISHED_TOOLS/L59-L78`.

## Schema and migration collision inventory

### Pinned main

Pinned main has seven application tables: `tasks`, `task_links`,
`task_comments`, `task_events`, `task_runs`, `task_attachments`, and
`kanban_notify_subs`. `tasks` holds current claim/PID/runtime/heartbeat and
`current_run_id`; `task_runs` holds attempt history. Neither has a native
executor type, executor configuration, or executor identity.
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:Task,Run/L840-L1056`,
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:SCHEMA_SQL/L1096-L1265`.

The schema declares no foreign keys and no application triggers. Base indexes
cover task assignee/status and status, link directions, comment/event time,
run task/start and status, attachment task/time, and notification task.
Additive indexes cover task tenant, idempotency key and session id plus event
run/id. `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:SCHEMA_SQL/L1267-L1276`,
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:_migrate_add_optional_columns/L1853-L2030`.

Additive migration is deliberately tolerant: it adds missing task/event/
notification columns and copies legacy names instead of renaming around
potential dependent views/triggers. It also backfills active runs and repairs
legacy event names. Any HARN columns must be idempotently additive and must not
assume a fresh database.
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:_migrate_add_optional_columns/L1853-L2101`.

`_REBUILD_SPECS` rebuilds drifted `task_events`, `task_comments`, `task_runs`,
and `kanban_notify_subs`, copies shared columns, regenerates identifiers where
specified, and recreates only the canonical indexes enumerated for each table.
Any additional/custom index attached to the renamed legacy table is removed
when that table is dropped and is not recreated. A trigger owned by the rebuilt
table is likewise not recreated. Views and triggers owned elsewhere but
dependent on the rebuilt table are not explicitly migrated or recreated; their
rename/drop behavior must not be claimed deterministic without controlling the
SQLite version and settings. If compatibility with dependent views or triggers
is promised, migration tests must create and exercise each such object before
and after rebuild. A new run column or index therefore has to be represented
consistently in fresh `SCHEMA_SQL`, additive migration, and the `task_runs`
rebuild specification. Structural fresh/legacy parity is a release requirement.
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:_REBUILD_SPECS,_rebuild_drifted_tables/L2104-L2226`.

Both pinned-main hard-delete paths omit `task_attachments`: they delete the
other enumerated child rows and the task but neither delete attachment rows nor
unlink the files named by `stored_path`. Only explicit single-attachment
deletion removes the row and then best-effort unlinks its file. Hard deletion
can therefore leave ordinary attachment rows and stored files orphaned.
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:delete_attachment/L3164-L3185`,
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:delete_archived_task,delete_task/L5568-L5614`.

### PR #67718

PR #67718 adds four spec-lock columns to `tasks`; sixteen `worker_kind` and
`external_*` lease/process/spec/attempt/recovery/result columns to `task_runs`;
`spec_hash` to `task_attachments`; and a new `task_external_artifacts` table.
It also adds singleton `kanban_board_identity` state to invalidate connections
that outlive a named board archive/delete.
It adds `idx_runs_worker_kind`, `idx_tasks_external_spec`, and
`idx_external_artifacts_run`; the last overlaps the table's
`UNIQUE(run_id,name)`. The artifact table has no foreign keys.
`6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_db.py:SCHEMA_SQL/L1286-L1428`,
`6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_db.py:_migrate_add_optional_columns/L2308-L2372`.

Its rebuilt `task_runs` specification repeats the external columns and worker
index. That parity obligation is correct, but HARN must not layer generic
harness columns next to synonymous `external_*` columns or inherit an
external-worker state machine. `6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_db.py:_REBUILD_SPECS,_rebuild_drifted_tables/L2478-L2580`.
Fresh/legacy parity tests worth adapting are
`6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_db_init.py:test_rebuilt_schema_matches_fresh_db/L138-L150`
and
`6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_db_init.py:test_external_worker_schema_is_present_on_fresh_and_legacy_boards/L180-L233`.

Artifact bytes are written before the SQL row commits, creating a crash window
for orphan files. The PR's two hard-delete paths explicitly delete
`task_external_artifacts` rows before `task_runs`, compensating for the missing
foreign keys, but do not unlink those rows' `stored_path` files. Both paths also
omit `task_attachments`, so ordinary attachment rows and stored files survive
task deletion. The locked spec is itself a `task_attachments` row, and explicit
single-attachment deletion rejects any attachment whose `spec_hash` is set;
hard deletion therefore also strands the locked spec row and file.
`6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:put_artifact/L2838-L2864`,
`6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_db.py:delete_attachment/L3529-L3553`,
`6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_db.py:delete_archived_task,delete_task/L6026-L6080`.

### PR #63297

PR #63297 adds `terminal_handoffs`, keyed by
`(task_id,idempotency_key)`, with no foreign key and no separate index. Because
connection initialization runs `SCHEMA_SQL`, the table would appear on both
fresh and existing boards. `ac6550e487e88e17a59a8454720eecacbd341709:hermes_cli/kanban_db.py:SCHEMA_SQL/L1195-L1204`,
`ac6550e487e88e17a59a8454720eecacbd341709:hermes_cli/kanban_db.py:connect/L1782-L1791`.
Its two hard-delete paths omit both `task_attachments` and `terminal_handoffs`,
so ordinary attachment rows/stored files and replay records can become orphans.
As at pinned main, only explicit single-attachment deletion removes the row and
then best-effort unlinks its file.
`ac6550e487e88e17a59a8454720eecacbd341709:hermes_cli/kanban_db.py:delete_attachment/L3225-L3246`,
`ac6550e487e88e17a59a8454720eecacbd341709:hermes_cli/kanban_db.py:delete_archived_task,delete_task/L5406-L5451`.

### Native deletion and orphan contract

Native HARN persistence must make deletion complete across both SQL and the
filesystem. Before either hard-delete transaction commits, it must identify all
task-owned ordinary attachments, the locked spec attachment, and all native
run-owned artifact paths; the transaction must delete their metadata rows and
every HARN run/attempt ledger row along with the existing child rows. After
commit, it must best-effort unlink only those validated, board-owned paths.
Because SQLite cannot atomically commit filesystem changes, interrupted writes
or interrupted post-commit unlinks must be recoverable: an idempotent orphan
sweeper must reconcile metadata and files under the trusted board storage roots,
remove unreferenced owned blobs, and handle rows whose blobs are missing under
an explicit fail-closed policy. It must never infer that SQL rollback removed a
file.

Focused tests must exercise both `delete_archived_task` and `delete_task` with
ordinary attachments, a locked spec attachment, multiple runs, and multiple
artifacts, asserting deletion of all SQL rows and stored files. Fault-injection
tests must interrupt after file write/before row commit and after row
deletion/before unlink, then run the sweeper twice to prove safe idempotent
cleanup. Migration coverage must also seed the PR-shaped orphan classes --
`task_external_artifacts` rows/files and `terminal_handoffs` rows -- so the
native migration or explicit rejection policy cannot silently retain them.

There is no literal pinned-main table/index name collision, and the two PRs do
not literally collide with each other's DDL names. The semantic collision is
more serious: `terminal_handoffs` would be a second terminal idempotency ledger
beside PR #67718's run-bound exact result hash/JSON/disposition fields, while
both duplicate pinned main's lifecycle transaction. That creates competing
answers to whether an attempt was accepted. Do not add `terminal_handoffs`, do
not carry the external-prefixed schema wholesale, and do not maintain parallel
terminal ledgers.
`6e682eaefdb1147e7e64a255483967e81460e658:hermes_cli/kanban_external_worker.py:finalize/L2351-L2444`,
`ac6550e487e88e17a59a8454720eecacbd341709:hermes_cli/kanban_db.py:terminal_handoff/L3004-L3120`.

Any minimal native persistence must also preserve pinned main's current
completion behavior, including artifact/scratch handling, dependency
promotion, run/event closure, failure counters, cleanup, and hooks. A copied
terminal SQL bundle from an old branch is not parity.
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:complete_task/L4134-L4300`,
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:block_task/L4876-L5065`.

## Merge and rebase risk

PR #67718 is close textually but far from the target semantically. Its three
dependent commits branch before only 17 pinned-main commits, and the scoped
pre-existing files have matching merge-base/main blobs, so a textual rebase is
likely clean. That cleanliness is dangerous: it would silently add external
claim/finalization APIs, exclude those cards from native dispatcher/recovery,
and wire a second lifecycle owner into dashboard and documentation.

PR #63297 is both textually and semantically high risk. It is 1101 commits
behind pinned main; seven of nine touched paths changed on main, with known
three-way conflicts in `hermes_cli/kanban.py`, `tools/kanban_tools.py`, and
`tests/tools/test_kanban_tools.py`. Auto-merging `kanban_db.py`, toolsets, MCP,
or ACP would be worse than an obvious conflict because it could silently expose
the duplicate handoff authority while bypassing later completion/artifact
semantics.

For both PRs, extract only the table classifications and invariant-level tests
onto pinned/current main. Re-derive every integration against current native
claim, dispatcher, recovery, completion/block, attachment, deletion, and
migration code.

## Fixed implementation touchpoints

Pinned main demonstrates the boundary that the native design must replace: the
legacy worker constructs a prompt in the spawn path, then assembles launch
arguments from task/profile/config state and passes them to `Popen`.
`26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:_default_spawn/L8195-L8195,L8286-L8343`.
The repository doctrine independently requires prompt-cache preservation and a
byte-stable system prompt.
`26480e6c57c3558442a73c2dffe313996b19417f:AGENTS.md:L19-L22,L88-L91`.

### HARN-01

- Add a focused canonical four-layer contract/hashing module under
  `hermes_cli/`: exact transport bytes, strict decoded JSON, validated canonical
  value, and domain object/result. Define which hash belongs to each layer and
  never silently substitute canonical reserialization for raw bytes.
- In addition, define one canonical, attempt-bound `HarnessInvocation`
  contract. Its exact transport bytes contain the byte-for-byte ordered
  prompt/instruction payload supplied to Pi and an unambiguous ordered list of
  every deterministic launch input that can affect it: invocation
  contract/schema version, driver identity/version, model identifier, ordered
  argv and non-secret environment/config inputs, and ordered tool-schema,
  skill-content, and other behavior-affecting configuration fingerprints.
  Credentials, tokens, secret values, and secret-derived fingerprints are not
  part of this envelope or its persistence.
- Persist the exact invocation bytes as an immutable native attachment, or as
  a content-addressed immutable attachment reference, together with its
  SHA-256. Bind that identity to the trusted task, run, immutable spec,
  attempt/generation, and executor tuple; retain it with the attempt for audit
  and replay. Re-read the immutable bytes and recompute their SHA-256
  immediately before process spawn. Attachment identity, stored hash,
  recomputed hash, and the complete trusted tuple must all agree or the attempt
  fails before work is enabled. This is one immutable input identity on the
  native attempt, not another lifecycle ledger.
- Add only justified persistence/migration hooks in
  `hermes_cli/kanban_db.py`. Use native names, distinguish executor identity
  from assignee/profile, and update fresh DDL, additive migration, rebuild DDL,
  indexes, deletion, and orphan policy together.
- Focus tests on strict JSON negatives, exact raw-byte identity, all four
  spec/result contract layers, invocation exact-byte identity, single-byte
  prompt mutation, launch-input reordering, tamper detection,
  fresh/existing/rebuilt schema parity, retention, and cleanup.

### HARN-02

- Add a focused `HarnessExecutor` / `HarnessDriver` seam and deterministic fake
  driver under `hermes_cli/`; do not add a core model tool.
- `HarnessExecutor` alone constructs the canonical invocation from already
  frozen attempt inputs, computes and persists its identity, re-reads and
  verifies it at the pre-spawn boundary, and passes the verified bytes/envelope
  unchanged to `HarnessDriver`. The driver may consume those bytes but may not
  regenerate, append to, reorder, or ad-lib the prompt/instructions from
  mutable task, profile, global configuration, tool, or skill state.
- Integrate dispatcher selection and Hermes-owned lifecycle in
  `hermes_cli/kanban_db.py`. A harness discriminator selects the executor;
  ordinary non-harness cards retain the exact legacy profile-spawn path.
  The baseline selection/recovery loop and profile spawn are
  `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:_dispatch_once_locked/L7505-L7807`
  and
  `26480e6c57c3558442a73c2dffe313996b19417f:hermes_cli/kanban_db.py:_default_spawn/L8169-L8355`.
- Use existing configuration validation surfaces for driver selection and
  options. Do not add a non-secret `.env` setting.
- Add focused contract, fake-driver, selection, identity, process binding,
  replay/divergence, restart, concurrency, recovery, and lifecycle tests. The
  fake driver must prove it receives the same already-verified invocation bytes
  across the seam and is never called after a prompt-byte, launch-order, or
  complete-identity mismatch. Assert that executor identity never changes
  profile ownership and that legacy dispatch output and behavior remain
  unchanged.

### HARN-06

- Implement the native Pi driver through `HarnessExecutor`; it is not a nested
  Hermes worker and receives no worker terminal tools.
- Immediately before spawn, separately verify the exact immutable invocation
  bytes and prompt/instruction payload against the attempt-bound invocation
  SHA-256 and complete invocation identity. This is a distinct pre-spawn
  invocation/prompt boundary, not one of the following four data/evidence
  boundaries; it must not be conflated with or silently inserted into their
  numbering.
- Perform four explicit SHA-256 checks: (1) caller-expected hash versus the
  submitted raw spec bytes at lock time, (2) locked and attachment-record
  hashes versus freshly re-read raw spec bytes immediately before execution,
  (3) exact result bytes versus the run's staged/replayed terminal hash and
  tuple, and (4) every declared artifact hash versus its exact durable bytes.
  Use a fresh isolated worktree and fresh attempt/session identity, and persist
  exact deterministic evidence and lifecycle results.
- Replay is permitted only when the complete invocation identity and SHA-256
  match the retained attempt record. Any prompt/instruction mutation, changed
  deterministic fingerprint, reordered launch input, or missing immutable
  invocation bytes is divergence and must fail before work is enabled.
- Hermes alone claims, heartbeats, verifies process exit, finalizes, and accepts
  the run. The Pi process cannot call complete, block, handoff, requeue, or a
  DB-writing bridge.
- Preserve legacy profile behavior unchanged for every non-harness card.

## Affected migration and test surfaces

Migration surfaces in `hermes_cli/kanban_db.py` are `SCHEMA_SQL`, model/row
mapping, `_migrate_add_optional_columns`, initialization order,
`_REBUILD_SPECS`, `_rebuild_drifted_tables`, named index recreation, legacy
active-run backfill, and both task deletion paths. Filesystem artifact layout,
attachment deletion, orphan cleanup, and crash recovery are part of migration
parity even though they are not SQLite DDL.

Focused test surfaces are:

- `tests/hermes_cli/test_kanban_db_init.py`: fresh versus legacy/rebuilt schema,
  additive idempotence, canonical/custom indexes, table-owned triggers, and
  explicit dependent-view/trigger behavior for every compatibility guarantee;
- existing `kanban_db` claim, dispatch, heartbeat, stale claim, crash, maximum
  runtime, completion, block, attachment, deletion, dependency, recurrence,
  and hook tests;
- new four-layer spec/result contract/hash and strict-decoder tests;
- new invocation-contract tests for exact prompt/instruction bytes, ordered
  launch fingerprints, mutation and reordering rejection, identical replay,
  divergent replay, immutable-attachment retention, and restart re-read/rehash;
- new executor/fake-driver tests for routing, trusted attempt identity,
  complete invocation identity, unchanged verified-byte delivery,
  executor/profile separation, process binding and absence, exact result and
  artifact bytes, replay/divergence, rollback, two-connection concurrency,
  restart/post-commit recovery, pre-spawn fail-closed behavior, and worker
  self-finalization rejection;
- legacy dispatcher/profile-spawn regression tests proving non-harness behavior
  is byte/behavior compatible;
- HARN-06 native Pi integration tests in a fresh isolated attempt, including
  tamper at each of the four data/evidence hash boundaries, separate
  invocation/prompt-hash tamper before spawn, and terminal-tool absence.

The PR test files are sources to port selectively, not suites that establish
the native architecture:
`6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_external_worker.py`,
`6e682eaefdb1147e7e64a255483967e81460e658:tests/hermes_cli/test_kanban_db_init.py`,
and
`ac6550e487e88e17a59a8454720eecacbd341709:tests/hermes_cli/test_kanban_terminal_handoff.py`.

## Downstream guardrails

- [ ] Use exact commit-object evidence only; record the complete SHA beside
      every source/test claim and never substitute a symbolic ref.
- [ ] Keep exactly one lifecycle owner: Hermes Kanban claims, heartbeats,
      finalizes, accepts, emits terminal events, and promotes dependents.
- [ ] Keep executor identity distinct from human/profile ownership; never
      represent Pi as a fake profile.
- [ ] Start each integration proof in a fresh isolated worktree and fresh
      ephemeral session/attempt identity.
- [ ] Disable compaction for the proof and fail closed if compaction occurs.
- [ ] Use no native subagents in the harness attempt and no nested Hermes
      worker/watch process.
- [ ] Remove terminal mutation tools/MCP/CLI bridges from the harness process;
      attempted self-finalization must fail.
- [ ] Bind the actual spawned process before enabling work and affirmatively
      establish process absence/reaping before release, retry, or acceptance.
- [ ] Compare and persist exact result bytes and exact artifact bytes; verify
      declared hashes and reject same-name/same-attempt divergent content.
- [ ] Freeze and persist the exact ordered prompt/instruction bytes and all
      deterministic non-secret launch fingerprints in an immutable
      attempt-bound invocation attachment; re-read and re-hash it immediately
      before spawn, and reject complete-identity mismatch, mutation,
      reordering, or missing retained bytes before enabling work.
- [ ] Perform all four SHA-256 checks: expected/submitted raw spec,
      locked/pre-execution re-read raw spec, exact staged/replayed result bytes,
      and every declared/durable artifact byte sequence. Keep the separate
      pre-spawn invocation/prompt SHA-256 boundary explicit and do not renumber
      these four data/evidence checks.
- [ ] Fence every mutation with the complete trusted task/run/spec/attempt/
      executor/invocation identity and assert guarded update row counts.
- [ ] Exercise identical replay, divergent replay, mid-transaction rollback,
      concurrent finalization, restart, and post-commit recovery.
- [ ] Maintain fresh/additive/rebuilt schema and canonical-index parity; include
      both deletion paths, filesystem orphan cleanup, custom-index and
      table-owned-trigger loss, and explicit dependent-view/trigger migration
      coverage for every compatibility guarantee.
- [ ] Prove legacy profile selection/spawn/recovery behavior is unchanged for
      non-harness cards.

## Uncertainties and proof limits

- The source and tests cited here were inspected at the exact commit objects;
  they were not executed as part of this reconciliation. No test result is
  claimed for either unmerged PR.
- Neither PR is claimed to be present on pinned main. Their schemas, APIs, and
  tests are research evidence only.
- PR #67718 has no demonstrated production supervisor/launcher E2E consumer;
  its process-binding and restart coverage is kernel-level. Its
  `ProcessAbsenceProof.evidence` is caller-supplied text, not independent OS
  verification.
- PR #67718's artifact write/SQL-commit crash window lacks fault-injection proof,
  and its artifact rows lack foreign keys.
- PR #63297 lacks relevant divergent-key, expired-claim, rollback,
  two-connection finalization, guarded-rowcount, migration/delete,
  post-commit-recovery, authorization, and pinned-main artifact-parity coverage.
- Exact field names and the minimum native persistence set remain HARN-01/HARN-02
  design decisions. They must be chosen from the canonical contract and native
  lifecycle needs, not copied from either external-worker schema.
- The stated ahead/behind counts are measurements at the freeze identities;
  they are not statements about later branch state.
