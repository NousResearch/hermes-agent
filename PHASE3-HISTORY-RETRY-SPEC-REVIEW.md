# Phase 3 authority-history — reservation retry spec re-acceptance

## Verdict: PASS — bounded spec acceptance only

The remaining reservation retry blocker in `PHASE3-HISTORY-FINAL-SPEC-REVIEW.md` is resolved. The correction implements that review's smallest requested change and adds real candidate regression evidence for both archive and destructive retries. Prior blocker dispositions remain accepted only within the previously documented bounded contract; this review does not reopen or expand them.

Generic independent Hermes reviewer, not BUREAU/CLIENT. Candidate: `C:/Users/sibag/hermes-phase3-authority-history`, branch `feat/phase3-authority-history`, HEAD `09109fec98016ffd7fef8622223073d296c02fa4`. **This is not quality approval, activation approval, or an all-green verification claim.** The parent's independent quality-review gate remains next.

## Correction inspected against the requested behavior

- `hermes_cli/kanban_history.py:215–228`: `reserve_removal` retains the enrolled-capability refusal first, then checks for the immutable reservation inside `write_txn`; an existing marker returns successfully, otherwise a plain INSERT creates it. It does not attempt a conflicting insert on retry.
- `hermes_cli/kanban_db.py:2316–2354`: the existing owned-writer context and `BEGIN IMMEDIATE` serialize the check/insert and retain before/after audits. Returning from inside the context still exits through its audit and commit; this is not an unaudited early return.
- `kanban_history.py:49–79,231–239`: immutability, conflict-insert and owned-insert guards remain intact, as does permanent enrollment refusal for a reserved board. No schema or guard relaxation is part of the fix.
- `kanban_db.py:808–836`: each supported removal still reserves before filesystem mutation. The already-reserved ordinary board can now reach rename/rmtree on a subsequent attempt, without making it eligible for enrollment.
- `tests/hermes_cli/test_kanban_history_lifecycle.py:146–186`: both parameterized cases create a real board, inject only the relevant filesystem failure, establish the retained reservation and enrollment refusal, and create/read an ordinary task. After restoring the filesystem operation, the second real `remove_board` succeeds. Archive additionally reopens the moved database, verifies retained task data and exactly one marker, and checks enrollment refusal again. Destructive removal verifies the original directory is absent and the deletion result is returned.

The destructive fault occurs before any rmtree deletion; partial filesystem destruction/recovery is not covered or accepted. No new decommissioning, authority continuity, or worker-exclusion contract is implied.

## RED/GREEN and frozen execution evidence

I read `PHASE3-HISTORY-RETRY-RESULT.md`, the final spec review, `PHASE3-HISTORY-FINISH-API-COVERAGE.md`, the actual changed production function and test, relevant caller/transaction/guard code, the isolation launcher/static helper, and all six retry logs. **I did not rerun tests or import candidate runtime code.** The following are inspected recorded executions, not executions performed by this reviewer.

| Evidence under `.phase3-evidence/` | Recorded result |
|---|---|
| `retry-red.log` | 2 failed, 23 deselected; exit 1. Both cases reach the second removal at lifecycle test line 174 and fail at the old `INSERT OR IGNORE` with `sqlite3.IntegrityError: authority history is immutable`. |
| `retry-green.log` | Same focused command and tests: 2 passed, 23 deselected; exit 0. |
| `retry-frozen-history.log` | Five history files: 121 passed; exit 0. |
| `retry-frozen-adjacent.log` | 96 passed, 1 skipped; exit 0. Skip is `test_backup.py:846`, POSIX file permissions only. |
| `retry-frozen-profiles.log` | 21 passed, 1 failed, 133 deselected; exit 1. |
| `retry-frozen-imports.log` | 1 passed; exit 0. Candidate backup, Kanban DB, history, profiles and constants origins printed. |

Non-duplicated frozen total: **239 passed, 1 failed, 1 skipped, 133 deselected**. Focused RED/GREEN results are not added again. The remaining failure is `TestExportImport.test_export_default_handles_broken_symlinks`, WinError 1314 at fixture symlink creation (`test_profiles.py:1439`), before production `export_profile` at line 1448. It remains a failed/unverified platform case, not a successful export test and not a retry acceptance blocker.

Logs identify the installed Python 3.11.4 interpreter used read-only, checkout-local tempfile and candidate import. The inspected launcher sets local HOME/USERPROFILE/HERMES_HOME/APPDATA/LOCALAPPDATA and TEMP/TMP/TMPDIR before imports, strips inherited credentials/board environment, and asserts candidate and tempfile origins. This is cooperative test containment, not an OS sandbox or deployed runtime proof. Adjacent, profiles and imports logged command arrays independently compare equal to their respective prior finish-frozen selections.

## Independently recomputed freeze and preservation checks

A read-only standard-library verification command completed with exit 0; it did not run the artifact-writing helpers. I independently recomputed hashes rather than relying on their PASS strings:

- All **29** explicit source/test/support inputs match current bytes; retry-freeze and retry-verification source maps are identical. All **28** Python inputs compile without imports or bytecode writes.
- Retry baseline's prior source entries match finish-freeze. Baseline to RED changes only the lifecycle test input; RED to frozen GREEN changes only `kanban_history.py`. Thus the same regression test bytes are bound across RED and GREEN.
- Replacing only the new reservation SELECT/return/plain INSERT block in memory with the former `INSERT OR IGNORE` reproduces the baseline production file SHA-256 exactly. This independently establishes that the production delta is confined to that function, including preservation of guards and enrollment logic elsewhere in the file.
- The **103-path** baseline preservation map has exactly the two authorized current-byte differences: `hermes_cli/kanban_history.py` and `tests/hermes_cli/test_kanban_history_lifecycle.py`. All other entries match baseline, including prior finish evidence and the final spec review. Every current preservation entry also matches retry-verification.
- All **10** retry evidence entries and the **one** retry result report bound by retry-verification match current bytes, including all six logs. All **114** listed manifest paths exist. Before writing this review, the only current nonignored changed/untracked path beyond the manifest was `retry-verification.json` itself, written after its path inventory. There were no missing manifest paths.
- Expected branch/HEAD, empty index and clean `git diff --check` independently confirmed.
- Recomputed **exactly three installed-source hashes**, matching `installed-before.json` and retry-verification: `hermes_cli/kanban_db.py`, `hermes_cli/kanban.py`, and `hermes_constants.py`. This is **not a whole-install or live-state attestation**.

Recomputed manifest anchors:

- `retry-red-freeze.json`: `d3cb0227f076a543cbf542464cac750ccfc63ade831228e1b44db76a680091d1`
- `retry-freeze.json`: `73d07bdedfb2671400dc98b1c51e29aeeb5220e8a8a7356aad298adc2c13cc1b`
- `retry-verification.json`: `65f6e959120f68e72be30ccef06b56c41c2c9019975a5d1c0e7d6094f3c743b0`

These explicit input/preservation maps are not exhaustive repository, transitive-import, ignored-state or installed-environment attestations. This new review report postdates the frozen manifests and is not claimed to be included in them.

## Scope and remaining gates

No remaining finding blocks this reservation retry spec acceptance. All limitations from the final spec review and API coverage contract remain: no full enrolled HTTP/browser matrix, exhaustive dynamic SQL or legacy variants, arbitrary-copy/custom-location/rollback continuity, comprehensive coherent administrator-corruption proof, power-loss/stress proof, surviving execution-tree exclusion, full repository suite, or full pinned quality/lint gate. Durable history does not authorize recovery or replacement work.

Proceed only to the parent's quality-review gate. No merge, installation, live enrollment, integration, publication or activation is authorized by this verdict.

**Only authored file:** `PHASE3-HISTORY-RETRY-SPEC-REVIEW.md`. No source/test/frozen-evidence edits, runtime/live-board/profile/config writes, tests, worker/gateway launches, PID signals, installs, sends, staging, commits or pushes were performed by this reviewer.
