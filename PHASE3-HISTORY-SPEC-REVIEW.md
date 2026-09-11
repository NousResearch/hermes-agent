# Phase 3 authority-history — independent SPEC / architecture review

## Verdict: REQUEST_CHANGES (accept the architectural direction, not the capability)

Reviewed the isolated `feat/phase3-authority-history` candidate at HEAD `09109fec98016ffd7fef8622223073d296c02fa4`. Generic independent reviewer; not BUREAU/CLIENT. The builder correctly labels this **incomplete**. Keep ownership integration, activation and recovery blocked.

**Partial acceptance:** authority-owned additive schema, same-board journal, existing claim arbitration and transaction boundary, prospective immutable scope/owner bindings, token-free snapshot intent, retained history independent of operational GC, explicit reader incarnation/cursor, and enrolled-board removal refusal are the right small amendment. No second scheduler, after-commit publication substitute, or Bureau monkeypatch is needed. Python enrollment/reader APIs are acceptable for this candidate; a CLI is not a prerequisite to fixing correctness.

## Evidence checked (not a fresh test run)

Read BRIEF, RESULT, PROGRESS, repository AGENTS.md, the read-only OWNERSHIP-HISTORY-DESIGN.md, new implementation/tests, tracked diffs, launcher, process probe, lifecycle inventory and verification artifacts.

| Preserved log under `.phase3-evidence/` | Actual recorded result |
|---|---|
| `final-dashboard-focused.log` | 40 passed, exit 0 |
| `final-boards.log` | 56 passed, exit 0 |
| `final-init.log` | 5 passed, exit 0 |
| `final-txn.log` | 8 passed, exit 0 |
| `final-plugin-smoke.log` | 2 passed, exit 0 |

These logs include the installed Python 3.11.4 executable, candidate `kanban_db.py` import path and checkout-local TEMP proof. No failures/skips/xfails are reported in those selected final runs. Historical RED logs exist, including missing audit (2 failures), integrity (4 failures), and dashboard child capture (1 failure); their failure bodies support the stated iteration rather than an extracted-model-only proof. All paths in the saved manifest exist. The board/init evidence is not a full legacy-schema or lifecycle matrix.

`claim_probe.py:17–28` calls the real candidate claim and interrupts the real COMMIT boundary; tests at `test_kanban_authority_history.py:164–201` check one winning process and pre/post-commit effects. This is useful real-runtime evidence, not power-loss testing, a synchronized contention stress test, receipt-reconciliation coverage, or execution-tree exclusion.

Independently ran read-only git checks: expected branch/base, empty index, `git diff --check` clean. Recomputed exactly the three installed-source baseline hashes (`hermes_cli/kanban_db.py`, `hermes_cli/kanban.py`, `hermes_constants.py`): all match `installed-before.json`; saved before/after JSON also matches. **Not a whole-install attestation.** No tests rerun: static findings below are sufficient to reject full acceptance and are not represented as newly executed failures. No runtime imports, live boards, installs, signals, workers, profile/config changes, sends or commits were needed. Only this review file was authored.

## Prioritized changes

### P0 — Authority mutation audit does not establish exact ownership, even inside `write_txn`

**Pointers:** `kanban_db.py:2329–2333`; `kanban_history.py:220–235,244–265,268–317`.

The acknowledged autocommit bypass remains: guards protect journal UPDATE/DELETE, not enrolled task/run changes, and an out-of-transaction coherent ownership mutation is invisible to the net audit. But the problem is wider than bypassing `write_txn`:

- Snapshots include `claim_lock`, yet the final comparison deliberately removes it (`303–305,314–316`). Change both live task/run credentials inside `write_txn`, append a known event such as heartbeat, and the comparisons have no check that the new credential hashes to the immutable run binding. `capture` only derives/verifies the credential at `kind == 'claimed'`; later events resolve owner solely from the old run binding. This can attribute a changed owner to the old public identity.
- `validate_existing` checks starting state only and does not compare live credentials to `authority_run_bindings`, validate all open/orphan runs, or validate final invariants. A known status event with `running` and no run/owner can pass capture/audit; the next write then refuses repair. The public dashboard HTTP route already refuses direct `running` (`plugin_api.py:865–869`), so this is a central-boundary defect, not a claim that its HTTP route currently permits it.
- Run comparison accepts **any** matching snapshot in the transaction (`316`), not the last observation for that run. It does not prove that the final run fact in replay equals committed run state.

**Bounded fix:** validate final task/run/credential-binding relations before COMMIT, require the last per-run snapshot to describe the final run disposition, and fail closed on unsupported credential replacement/orphan/missing-owner states. Establish an enforceable owned-writer boundary for enrolled mutations (including autocommit/explicit-BEGIN callers), or prove and explicitly constrain all supported writers; a comment requiring `write_txn` is insufficient. Do not advertise protection against an administrator rewriting the SQLite file.

**Required RED cases:** coherent credential swap plus emitted heartbeat; emitted invalid running/no-run state; multiple snapshots ending with a misleading run state; coherent task/run autocommit mutation; stale owner/run refusal. Assert task, run, source event, journal and bindings roll back together. Do not substitute another scheduler.

### P1 — Strict reader and immutability claims exceed actual checks

**Pointers:** `kanban_history.py:15–49,76–100,320–379`; tests `:111–118,227–243,306–315`.

Concrete static gaps:

- `json.loads` accepts duplicate keys. Versions use equality rather than exact integer type (`True`/`1.0` compare equal to 1). Run ID/start may be null; IDs can be non-positive. Structural validation does not enforce running/current-run/owner or terminal-state relationships.
- A well-shaped record with another task binding, unrelated run/owner, or reused source-event ID passes: reader never checks immutable relational bindings or event uniqueness. Source events can legitimately disappear through GC, so the check must use retained identity/provenance, not require operational rows to survive.
- `MAX(sequence) == COUNT(*)` alone is not a proof of the exact positive sequence `1..N` when direct INSERT can supply zero/negative sequence values. Test exact start/adjacency and stale cursors without confusing global source-event gaps with journal gaps.
- UPDATE/DELETE triggers alone do not cover SQLite replacement semantics. `INSERT OR REPLACE` can replace a conflicting row using an implicit delete whose delete triggers depend on `recursive_triggers`; this candidate does not establish that guard. Test metadata, scope/owner/run binding and journal replacement, including another connection, not only direct DELETE. This is an additional immutability blocker, not an executed reviewer repro.
- Guard SQL text is checked, but table structure/version/binding consistency is not comprehensively checked. `capability` can return unenrolled with orphan binding rows. `audit_start` does not validate existing journal records/coverage: a corrupted retained stream can still receive a new grant even though reading it fails.

**Bounded fix:** one strict versioned validator for capture/read, strict JSON decoding (duplicate keys/non-finite values refused), exact types and state relations, retained binding/provenance checks and insertion/replacement protection. Validate malformed capability/schema/history before granting new authority; never silently repair/backfill. Test an invalid later record in a page to ensure no partial page is returned, as well as malformed history predating the requested page/cursor under the declared coverage contract. Preserve legitimate unowned synthetic terminal runs; do not impose a blanket owner requirement on every historical run.

### P1 — A real supported lifecycle is already incompatible: scheduling

**Pointers:** `kanban_db.py:5973–6015`; `kanban_history.py:155,181–189,210–222,254–264`; dashboard `plugin_api.py:853–859`.

`scheduled` is an allowed event kind but absent from **both** `TASK_STATES` and `RUN_STATES`. `schedule_task` writes task status `scheduled`, closes a run with status `scheduled`, then captures that event. An enrolled ready/running task therefore rolls back with unknown-state error. This is a concrete hidden lifecycle blocker, not merely missing test coverage. Prospective enrollment also excludes an otherwise unclaimed scheduled task without documenting why.

**Bounded fix:** add real imported tests for schedule/unblock with and without an active or synthetic run; align valid task/run states with supported runtime transitions. Finish the explicit matrix for extension/defer, timeout, stale detection, crash/rate-limit/protocol violation, typed dependency wait/block-loop/unblock, and legacy migration/defensive refusal. Methods are indexed in `.phase3-evidence/lifecycle-inventory.txt` (`release_stale_claims:3758`, `_defer_reclaim_for_live_worker:6378`, `enforce_max_runtime:6470`, `detect_stale_running:6591`, `detect_crashed_workers:6802`, `block_task:4922`, `unblock_task:5208`). Mock only liveness/termination boundaries; no real PID signalling or worker launch. Assert payload/identity and rollback, not just final kind/status. Inventory is an aid, not complete dynamic-SQL/caller proof.

### P1 — Active deletion needs an explicit terminal run contract

**Pointers:** `kanban_db.py:5642–5661`; `kanban_history.py:207–217,310–313`; lifecycle test `:270–278`.

Hard-delete removes the task before capture; no current run can be selected, so `deleted` has null run/owner and all runs are then erased. Audit explicitly exempts their deletion. The current test explicitly skips owner/run assertions for active deletion. A task tombstone is real evidence of logical deletion, but this does not satisfy the brief's full terminal owner/run disposition and must not be interpreted as process termination.

**Smallest acceptable fix:** refuse active deletion for enrolled tasks, preserving ordinary-board behavior. Alternatively close/snapshot the exact active run and owner before deletion, retaining an unambiguous deletion tombstone in the same transaction. Test failure rollback and retention after both hard-delete APIs. No execution-exclusion claim.

### P1 before integration — Incarnation continuity and coverage limits must be explicit

**Pointers:** `kanban_history.py:117–126,142–178,237–241,354–379`; `kanban_db.py:805–811`.

Enrollment generates a UUID in the copied DB; reopening a clone/old snapshot reuses it. `after > highwater` detects one rollback shape, but a reader starting at zero or behind the restored highwater cannot detect a fork/regression. Board-removal refusal/reservation solves the supported removal race, not clone/restore or multiple authorities with the same identity. No external fencing scheme should be invented in this next patch.

**Bounded next decision:** document supported single-database authority semantics; make any supported restore/clone/import activation explicitly refuse enrolled identity reuse until an approved reconciliation/new-incarnation procedure exists. Label arbitrary filesystem copies and snapshot rollback unsupported/UNKNOWN, not continuous authority. Keep clone/restore/new-epoch activation outside this pass if no supported path exists; do not claim the journal alone proves sole authority across DB copies.

Freeze the API contract alongside tests: board enrollment is not enrollment of all tasks; `history_bound` starts each task's prospective coverage; zero records is not global vacancy. Define whether `runtime_id` is a unique runtime **instance** (RESULT claims an instance binding, but there is no separate instance field). The design also requests original observed timestamp: current record drops source `created_at`, leaving no event timestamp for non-run transitions. Retain that timestamp in the versioned record or explicitly obtain approval for the reduced contract; do not invent historical timestamps on replay.

## Minimum next implementation pass / exit gate

1. Write RED tests for the central audit/identity and strict-reader/immutability counterexamples above, then fix those boundaries first. Keep the existing sole claim CAS and same-DB capture.
2. Add schedule/unblock coverage and complete the bounded lifecycle matrix. Choose active-delete refusal unless there is a demonstrated need for the larger close-and-tombstone contract.
3. Document Python API, prospective coverage, instance identity, non-authoritative event kinds, and unsupported clone/restore semantics. No CLI, publication/recovery subsystem, or huge new design is required. The irreversible removal marker is a defensible fail-closed reservation; add failure-path tests/documentation for an ordinary board whose filesystem removal fails and which can never subsequently enroll.
4. Run only reviewed safe suites using `.phase3-evidence/run_isolated.py` with **new unique labels**, preserve RED/history files, and refresh builder evidence after source changes. Existing launcher overwrites duplicate labels (`:44`), so label uniqueness is operationally necessary. Require focused history, boards/init/txn, audited lifecycle and dashboard tests; do not equate two dashboard smoke cases with full integration coverage.
5. Independent re-review remains required. Exit gate: exact owner/run invariants and fail-closed writes/readers, retained terminal disposition, exercised supported lifecycle matrix, explicit prospective/incarnation contract, unchanged ordinary-board behavior and verified evidence.

**Non-negotiable boundary:** durable DB history does not exclude a surviving old execution tree. No safe automatic recovery, runtime installation/enrollment, publication, or activation is approved by this review.
