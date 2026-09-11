# Phase 3 history reservation retry result

## Outcome

Bounded reservation retry defect repaired with actual candidate RED/GREEN. Frozen candidate is ready for independent spec re-acceptance, then the parent's quality-review gate; NOT self-approved or activated. Generic Hermes worker, not BUREAU/CLIENT.

Checkout: `C:/Users/sibag/hermes-phase3-authority-history`; branch `feat/phase3-authority-history`; HEAD `09109fec98016ffd7fef8622223073d296c02fa4` unchanged. Read FINAL-SPEC-REVIEW fully, original/finish briefs, finish result and both launcher/verification helpers before execution.

## Exactly what changed

- `hermes_cli/kanban_history.py`, only `reserve_removal`: after the existing capability refusal, inside the existing serialized audited `write_txn`, return if the immutable reservation exists; otherwise plain INSERT. No guard, enrollment fence, schema or transaction changes. Reconstructing the former one-line INSERT from the final bytes exactly matched its baseline SHA-256.
- `tests/hermes_cli/test_kanban_history_lifecycle.py`: replace the one archive-failure-only case with two parameterized archive/destructive retry cases. Only the filesystem rename/rmtree is fault-injected; real candidate board creation, SQLite reservation, enrollment refusal, ordinary task creation/read, and subsequent successful removal execute. Archive also proves retained task data and enrollment refusal after reopening the archived database. Destructive failure is injected before rmtree deletes anything; partial filesystem destruction is not claimed.
- New evidence only: `.phase3-evidence/retry_verify.py`, `retry-baseline.json`, `retry-red-freeze.json`, `retry-freeze.json`, `retry-verification.json`, six `retry-*.log` files, and this report. All other existing candidate edits and prior evidence preserved. No prior finish manifest/log/report overwritten. `verify_finish.py` was inspected but not run.

## Real execution

Every runtime command used the unchanged reviewed launcher, with fresh labels checked absent before use:

    python -B .phase3-evidence/run_isolated.py LABEL PYTEST_ARGS

Exact resolved child command, interpreter, candidate path, tempfile and exit are recorded in each log. The adjacent/profiles/imports arguments were replayed from the existing `finish-frozen-*.log` command arrays through the launcher, changing only the evidence label. These are NEW current-source executions, not historical green claims.

| Evidence under `.phase3-evidence/` | Result |
|---|---|
| `retry-red.log` | 2 failed, 23 deselected; exit 1. Both retry variants reached the second real remove_board call and failed in reserve_removal with `sqlite3.IntegrityError: authority history is immutable`. First filesystem failure, enrollment refusal and ordinary task behavior had already executed. Production not yet changed. |
| `retry-green.log` | Same tests: 2 passed, 23 deselected; exit 0. |
| `retry-frozen-history.log` | All five history files: 121 passed; exit 0. |
| `retry-frozen-adjacent.log` | Boards, init, transaction retry, selected dashboard, backup import/roundtrip and quick restore: 96 passed, 1 skipped; exit 0. Same exact selection as finish frozen adjacent. |
| `retry-frozen-profiles.log` | Same `test_profiles.py -k 'clone_all or import'`: 21 passed, 1 failed, 133 deselected; exit 1. |
| `retry-frozen-imports.log` | Unchanged finish_imports_test.py: 1 passed; exit 0. Candidate backup, profiles, kanban_db, kanban_history and constants origins confirmed. |

Final non-duplicated frozen total: **239 passed, 1 failed, 1 skipped, 133 deselected**. Focused RED/GREEN not added to that total. NOT ALL GREEN.

Focused arguments: `tests/hermes_cli/test_kanban_history_lifecycle.py -k failed_board_removal -q --tb=short`.
History arguments: `tests/hermes_cli/test_kanban_authority_history.py tests/hermes_cli/test_kanban_history_pass2.py tests/hermes_cli/test_kanban_history_strict.py tests/hermes_cli/test_kanban_history_lifecycle.py tests/hermes_cli/test_kanban_history_profile_copy.py -q --tb=short`.
All adjacent exact node IDs are in the logged COMMAND and unchanged prior FINISH-RESULT command listing.

## Freeze, preservation and checks

Static commands executed before/after mutation and runtime verification:

    python -B .phase3-evidence/retry_verify.py retry-baseline
    python -B .phase3-evidence/retry_verify.py retry-red-freeze
    python -B .phase3-evidence/retry_verify.py retry-freeze
    python -B .phase3-evidence/retry_verify.py retry-verification

Final manifest: `.phase3-evidence/retry-verification.json`. It binds this report, every retry evidence file preceding the final manifest, 29 explicit source/test/support inputs (the prior 28 plus the retry static helper), full nonignored modified/untracked path listing, branch/HEAD/status and installed hashes. Its source map must equal retry-freeze. Baseline preservation has 103 paths; only the two authorized source/test paths may differ, all remaining entries must exactly match. This includes original finish freezes/logs and spec review. This is an explicit input/preservation map, not an exhaustive transitive-import or whole-repository attestation. Ignored disposable state is excluded.

`retry-freeze.json` SHA-256: `73d07bdedfb2671400dc98b1c51e29aeeb5220e8a8a7356aad298adc2c13cc1b`.
`retry-red-freeze.json` binds pre-fix source and the same regression tests used for GREEN. Final helper also checks compilation without runtime imports or pyc, git diff --check, empty index, expected branch/base, and the exact three installed source hashes against installed-before.json. No whole-install/live-state attestation.

All runtime state uses launcher-local HOME/USERPROFILE/HERMES_HOME/APPDATA/LOCALAPPDATA and TEMP/TMP/TMPDIR under `.phase3-evidence/state`; child tempfile and candidate import are asserted before tests. Windows essentials retained, credentials/board environment scrubbed. Read-only interpreter reuse: installed venv Python 3.11.4. Cooperative test containment, not an OS sandbox.

One initial static baseline attempt timed out while unnecessarily hashing all tracked repository files. No baseline artifact existed afterward. The new retry helper was narrowed before baseline creation to the nonignored changed/untracked preservation set plus prior explicit frozen inputs, then succeeded. No runtime test or product edit preceded the successful baseline.

## Remaining gates and limitations

1. Independent spec acceptance of this corrected frozen candidate, then parent quality review. No self-approval or quality substitute; no full repository suite or pinned full quality/lint gate claimed.
2. Current-source Windows failure remains `TestExportImport.test_export_default_handles_broken_symlinks` at test_profiles.py:1439, WinError 1314 creating fixture symlink before export production code. No privilege change or skip masking. `test_backup.py:846` remains the existing POSIX-permissions skip.
3. Prior FINAL-SPEC-REVIEW and FINISH-API-COVERAGE scope limits remain: no exhaustive dynamic SQL/legacy variants, full enrolled HTTP/browser validation, arbitrary-copy/custom-location/rollback continuity, coherent administrator corruption proof, power-loss/stress or surviving execution-tree exclusion. Durable history does not authorize recovery/replacement work.
4. No real live boards/PIDs, worker/gateway launches, profiles/runtime/config edits, installs, skill writes, staging, commits, pushes, sends, merge, enrollment of live authority or activation. All task-authored files and disposable test writes confined to this checkout. Existing dirty candidate changes retained.
