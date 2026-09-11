# Phase 3 authority history — continuation checkpoint (NOT ACCEPTED)

Actor: Hermes CLI worker, not BUREAU/CLIENT.
Scope: only C:/Users/sibag/hermes-phase3-authority-history was edited. No staging, commit, push, merge, activation, live-board/profile mutation, worker launch, scheduler creation, runtime install or dependency install was performed.

Status: substantial real-tested candidate implementation, but the full brief is NOT complete. This checkpoint is not deployment approval or proof of complete lifecycle coverage. Independent review is pending. The older execution-tree-exclusion limitation is unchanged.

## Candidate changes

- hermes_cli/kanban_history.py: additive owned schema and integrity guards; permanent opt-in board incarnation; prospective immutable repo/project/task bindings and consumer/runtime/instance/non-secret-owner bindings; private token digest to run binding; durable token-free versioned snapshots; strict cursor reader; transaction net-ownership audit; pre-migration integrity check; irreversible board-removal reservation.
- hermes_cli/kanban_db.py: owned migration hook and Python API facade; transaction validation/audit; same-transaction capture at _append_event; claim-renewal event; deletion tombstones; board-removal enrollment fence/refusal.
- plugins/kanban/dashboard/plugin_api.py: direct status and dependent-child demotion now use the owned event writer rather than raw event SQL.
- tests/hermes_cli/test_kanban_authority_history.py: real imported candidate SQLite contracts, process contention, abrupt exit boundary probes, retention and fail-closed tests.
- tests/hermes_cli/test_kanban_boards.py: one Windows handle-lifetime correction (connect_closing before filesystem removal).
- tests/hermes_cli/test_kanban_write_txn_busy_retry.py: boundary-only fake answers schema SELECT with empty schema; production integrity checks not disabled.
- .phase3-evidence/: strict isolated launcher, subprocess contention/crash probe, read-only AST inventory, verification helper, full RED/GREEN logs and manifest. state/ is disposable and ignored.
- PHASE3-HISTORY-PROGRESS.md and this result: observable progress and exact continuation. Supplied brief remains unchanged and untracked.

## Isolation and evidence

Use only this command pattern from the candidate root:

    python -B .phase3-evidence/run_isolated.py <unique-log-label> <explicit-test-path-or-nodeid> -q --tb=short

The launcher runs installed Python 3.11.4 read-only, clears non-allowlisted environment, preserves Windows essentials, sets HOME/USERPROFILE/HERMES_HOME/APPDATA/LOCALAPPDATA/TEMP/TMP/TMPDIR inside .phase3-evidence/state, resets tempfile cache, and proves candidate import location before pytest. No dependency install was needed; the existing installed interpreter was used for the real tests. Existing scripts/run_tests.sh was inspected but not used unchanged because its env -i treatment does not preserve the required Windows setup/local tempfile routing.

Evidence logs preserve intermediate failures, including two corrected test-author API mistakes (create_task returns an ID; complete_task has no cleanup keyword). No failed tests were deleted or xfailed to make a run green. Only explicitly selected safe suites were run, not the unreviewed whole repository suite, which includes git/worker operations outside the permitted test activity.

Verified before the final dashboard child fix:
- green-unemitted.log: 38 passed.
- final-focused.log: 39 passed (includes board-removal race regression).
- final-boards.log: 56 passed.
- final-init.log: 5 passed.
- adjacent-txn-audited.log: 8 passed.
- red-dashboard.log: 1 failed, 39 passed; real dashboard writer was rejected as unjournaled.
- red-dashboard-child.log: 1 failed, 39 passed; parent status was fixed, dependent child raw SQL event still rejected. Both source event writes are now routed through the owned writer; final rerun pending below.

Process tests import real candidate runtime in two independent subprocesses, verify exactly one claim succeeds, and use explicit fault injection at the real COMMIT boundary with os._exit(23). They prove rollback-before-commit and durable-after-commit behavior. They do not launch task workers or LLM agents. SQL fault triggers test journal-insert rollback. GC and hard-delete preserve retained records. Cursor mismatches and malformed/versioned capability, metadata, guards and records have focused tests.

## Acceptance gaps / exact continuation

1. Finish the lifecycle matrix against actual imported methods. Already covered: ordinary/review claims, renewal, manual/stale reclaim, heartbeat, spawn failure/give-up, complete, block, archive, active and archived deletion, GC, dashboard direct status and reopened-parent demotion. Still missing explicit exercised cases: live-worker extension and reclaim deferral, timeout enforcement, stale-runtime detection, crash/rate-limit/protocol-violation outcomes, typed dependency blocks/unblocks, and legitimate legacy schema variants. Never signal existing PIDs or launch workers; inject only process-liveness/termination boundaries while exercising real DB implementations.
2. Complete strict-reader scrutiny: duplicate JSON keys, non-finite/nonnumeric values, nested extra fields, binding version corruption, sequence gaps/source-event reuse, wrong task/run binding and malformed run ownership. Current validation is NOT proven comprehensive. Public reader must never return partial data on a malformed page; never silently skip unknown versions.
3. Source inventory is .phase3-evidence/lifecycle-inventory.txt, generated by read-only AST inspection. Refresh after edits with `python -B .phase3-evidence/inventory.py`. Audit f-string/dynamic SQL and callers as well; the inventory is an aid, not exhaustive proof. Direct status writers were found in plugins/kanban/dashboard/plugin_api.py and fixed through RED tests. Other raw event inserts there concern prose/priority and do not currently change tracked ownership fields; decide/document coverage rather than imply all event kinds are retained.
4. Transaction audit refuses un-emitted NET ownership changes inside write_txn. It is not a general SQLite authorization boundary. Direct autocommit writes / code bypassing write_txn can evade it; prove all authority writers participate or add a fail-closed enforcement boundary. Existing validate_existing catches inconsistent running/run pointers but not every coherent out-of-band mutation. This remains an acceptance blocker.
5. Active hard-delete currently emits a deleted tombstone after the task row is removed and does not carry a closing run snapshot. Prior claim/run history survives, but terminal owner/run disposition must be reviewed/tested before claiming complete transition payloads.
6. Prospectivity is conservative: task enrollment refuses any existing runs; no historical backfill or automatic repair. Board removal now reserves an immutable marker in SQLite before filesystem action, so enrollment cannot race removal. Failed filesystem removal leaves the board usable as ordinary but permanently ineligible for enrollment; this fail-closed behavior needs review. No clone/restore/new-incarnation workflow is implemented; do not claim clone continuity or activate one.
7. No CLI enrollment surface was added: Python API facade is the candidate interface. Confirm whether this satisfies the brief's explicit opt-in API contract during independent review. Do not add activation or installed-runtime work.
8. Rerun final safe suites after any change and refresh manifest/verification using `python -B .phase3-evidence/verify.py`. The helper compiles without pyc, checks git diff --check and empty index/unchanged HEAD, and compares three installed-source hashes to the saved baseline. That comparison covers exactly those three files, not a whole-install attestation.
9. Independent review/sign-off remains PENDING. No BUREAU/CLIENT identity or delivery is claimed. Save exact outstanding failures, commands and status in these files before another turn limit. No commit/push/merge/activation is authorized.

## Final rerun / verification

Final executed evidence (each log contains exact interpreter command, import/temp proof, full pytest output and exit):

- `python -B .phase3-evidence/run_isolated.py final-dashboard-focused tests/hermes_cli/test_kanban_authority_history.py -q --tb=short` -> 40 passed in 12.72s, exit 0.
- `python -B .phase3-evidence/run_isolated.py final-boards tests/hermes_cli/test_kanban_boards.py -q --tb=short` -> 56 passed in 29.32s, exit 0. This ran after final core DB/removal edits; subsequent edits only affected dashboard and its focused test.
- `python -B .phase3-evidence/run_isolated.py final-init tests/hermes_cli/test_kanban_db_init.py -q --tb=short` -> 5 passed in 2.65s, exit 0.
- `python -B .phase3-evidence/run_isolated.py final-txn tests/hermes_cli/test_kanban_write_txn_busy_retry.py -q --tb=short` -> 8 passed in 1.87s, exit 0.
- `python -B .phase3-evidence/run_isolated.py final-plugin-smoke tests/plugins/test_kanban_dashboard_plugin.py::test_board_empty tests/plugins/test_kanban_dashboard_plugin.py::test_create_task_appears_on_board -q --tb=short` -> 2 passed in 3.82s, exit 0.

Final selected runs: zero failures, zero skips/xfails reported. This is NOT the complete repository test suite or complete lifecycle acceptance matrix.

`python -B .phase3-evidence/verify.py` -> exit 0: 10 touched/new Python files compile without bytecode; git diff --check passes; no staged paths; HEAD unchanged at 09109fec98016ffd7fef8622223073d296c02fa4; all three baseline installed hashes match exactly. No changes to code after these passing runs.

Independent parent-Hermes review: PENDING. Build acceptance: INCOMPLETE for the explicit gaps above. No runtime activation or safe old-worker recovery is implied.

## Manifest

Complete scoped nonignored changed/untracked manifest follows (also generated in .phase3-evidence/manifest.txt). Disposable state is excluded by .phase3-evidence/.gitignore. Exact hashes: .phase3-evidence/installed-before.json and installed-after.json. Verification output: .phase3-evidence/verification.log.

    .phase3-evidence/.gitignore
    .phase3-evidence/adjacent-boards-fixed.log
    .phase3-evidence/adjacent-boards.log
    .phase3-evidence/adjacent-init.log
    .phase3-evidence/adjacent-txn-audited.log
    .phase3-evidence/adjacent-txn-fixed.log
    .phase3-evidence/adjacent-txn.log
    .phase3-evidence/claim_probe.py
    .phase3-evidence/final-boards.log
    .phase3-evidence/final-dashboard-focused.log
    .phase3-evidence/final-focused.log
    .phase3-evidence/final-init.log
    .phase3-evidence/final-plugin-smoke.log
    .phase3-evidence/final-txn.log
    .phase3-evidence/green-bindings-attempt.log
    .phase3-evidence/green-bindings.log
    .phase3-evidence/green-enrollment-fixed.log
    .phase3-evidence/green-enrollment.log
    .phase3-evidence/green-integrity-process.log
    .phase3-evidence/green-unemitted.log
    .phase3-evidence/installed-after.json
    .phase3-evidence/installed-before.json
    .phase3-evidence/inventory-command.log
    .phase3-evidence/inventory.py
    .phase3-evidence/lifecycle-inventory.txt
    .phase3-evidence/lifecycle-matrix.log
    .phase3-evidence/manifest.txt
    .phase3-evidence/red-bindings-retention.log
    .phase3-evidence/red-dashboard-child.log
    .phase3-evidence/red-dashboard.log
    .phase3-evidence/red-enrollment.log
    .phase3-evidence/red-integrity-process.log
    .phase3-evidence/red-removal-race.log
    .phase3-evidence/red-unemitted.log
    .phase3-evidence/run_isolated.py
    .phase3-evidence/verification.log
    .phase3-evidence/verify.py
    PHASE3-HISTORY-BRIEF.md
    PHASE3-HISTORY-PROGRESS.md
    PHASE3-HISTORY-RESULT.md
    hermes_cli/kanban_db.py
    hermes_cli/kanban_history.py
    plugins/kanban/dashboard/plugin_api.py
    tests/hermes_cli/test_kanban_authority_history.py
    tests/hermes_cli/test_kanban_boards.py
    tests/hermes_cli/test_kanban_write_txn_busy_retry.py
