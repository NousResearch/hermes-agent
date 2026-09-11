# Phase 3 authority-history — final independent spec re-review

## Verdict: REQUEST_CHANGES

The prior P0 ownership/audit blockers and principal P1 history, lifecycle, deletion and copy-identity blockers are addressed in the frozen candidate. One bounded ordinary-board lifecycle regression remains: **a failed filesystem removal permanently prevents a subsequent supported removal retry**, because the new immutable replacement trigger also aborts the reservation's `INSERT OR IGNORE`.

This is a generic Hermes independent spec review, not BUREAU/CLIENT, not a quality-review substitute, and not activation approval. Candidate: `feat/phase3-authority-history`, HEAD `09109fec98016ffd7fef8622223073d296c02fa4`. No integration, installation, enrollment, publication, recovery or old-worker execution-exclusion approval is given.

## Blocking finding — ordinary removal retry is no longer supported

**Priority: P2, spec acceptance blocker (ordinary-board compatibility / requested removal-failure path).**

- `hermes_cli/kanban_db.py:808–811` calls `reserve_removal` on every `remove_board` attempt, before the filesystem operation at `832` or `836`.
- `hermes_cli/kanban_history.py:215–226` commits a permanent reservation using `INSERT OR IGNORE INTO authority_history_removal VALUES (1)`.
- `_guards`, `kanban_history.py:64–74`, includes `authority_history_removal` among the tables with a **BEFORE INSERT** conflict guard. If singleton 1 already exists, this guard executes `RAISE(ABORT, 'authority history is immutable')`.
- `OR IGNORE` does not suppress this trigger's explicit abort. After a first rename/rmtree failure leaves the reservation behind, every later supported removal attempt aborts before reaching the filesystem, even after the original filesystem problem has been resolved. This is not an enrolled board and does not need authority decommissioning.

The irreversible reservation is an acceptable enrollment fence; making its acquisition non-idempotent is not required by that fence. The original brief requires ordinary task/board behavior to be preserved, and the prior review explicitly requested this failure-path treatment. The documented inability to enroll after a failed removal does not document or justify losing the ability to remove that ordinary board.

**Coverage gap:** `tests/hermes_cli/test_kanban_history_lifecycle.py:146–162` injects one rename failure, then verifies enrollment refusal and ordinary task creation. It never retries removal after restoring the filesystem operation. The selected green board/history suites therefore do not contradict this finding.

**Smallest requested correction:** inside the existing serialized, audited reservation transaction, return successfully when the immutable reservation is already present; insert only when absent. Keep enrollment refusal and every production immutability/owned-insert guard intact. Add real candidate regression coverage for failed archive and failed destructive removal followed by successful retry, plus continued enrollment refusal before retry. No schema weakening or new decommissioning architecture is needed.

This finding is from direct control-flow/trigger inspection, **not a newly executed runtime repro**. I did not mutate the frozen candidate or create test state to reproduce it.

## Prior blocker disposition against actual code and tests

| Prior issue | Re-review disposition and anchors |
|---|---|
| Exact credentials, final invariants, last observation | Addressed. `kanban_history.py:335–352` checks credential hash against immutable run binding; `368–407` validates starting/final live task and open-run relations, including orphan runs and terminal credential clearing. `427–464` checks the last task/run observation, including observations when the net live state is unchanged. `kanban_db.py:2318–2342` places audits inside the existing transaction. `test_kanban_history_pass2.py:16–98` covers coherent swap, running without run, misleading final observations, orphan run, and rollback comparisons across operational and retained tables. |
| Unsupported raw writers / insertion / replacement | Addressed for the declared accidental/unsupported-writer boundary. Persistent task/run triggers cover old and new enrolled identities; all authority tables have owned-insert, update/delete and conflict-insert guards (`kanban_history.py:49–102`). Additional connections lack the enabled UDF. Pass2 bypass and strict raw-insert/replace tests exercise real SQLite. This is not an administrator-proof security boundary. The reservation conflict interaction above is the remaining compatibility defect, not a reason to weaken these guards. |
| Strict reader and retained provenance | Addressed for the bounded contract. Exact schema/guard definitions and retained binding checks (`129–179`); source timestamp/provenance captured in the same transaction (`354–365`); shared record validator (`467–534`); duplicate-key/non-finite rejection (`537–550`); exact positive adjacency, version and source uniqueness (`553–567`). Reader validates the whole stream before returning any page (`570–589`), and writes validate it before granting authority. Strict tests corrupt records behind the cursor, foreign bindings/owners, source IDs/kinds/timestamps, shapes, versions and sequence. Source GC is explicitly exercised. No claim of comprehensive coherent administrator-corruption detection or retained-binding/first-record bijection is accepted. |
| Scheduling and missing lifecycle matrix | Addressed in the requested bounded slice. `scheduled` is in both state sets (`300–301`); prospective scheduled binding is permitted (`265–269`). Actual `schedule_task` (`kanban_db.py:5987–6029`) closes/synthesizes runs and captures before commit. Lifecycle tests cover unowned, synthetic and active scheduling/unblocking; extension/defer/timeout/stale/crash/rate-limit/protocol success and journal-fault rollback; typed dependency waits and block-loop/unblock. Process liveness/termination is mocked or injected, not established as safe runtime exclusion. |
| Active deletion / terminal history | Addressed by the smaller refusal contract. `kanban_db.py:5662–5666` refuses enrolled active deletion before deleting anything; archived-only deletion cannot delete a running task. Earlier completion/archive records retain exact run/owner disposition, followed by a logical deletion tombstone. Authority tests cover GC and both hard-delete paths; lifecycle tests compare all affected rows on delete-journal fault and preserve ordinary active-delete behavior. Tombstones do not mean a worker died. |
| Clone/restore and incarnation continuity | Addressed for documented standard locations and supported candidate paths. `refuse_authority_copy` (`kanban_history.py:182–212`) inspects standard DBs read-only without migration. `profiles.py:1059–1071,2081–2083` checks staged clone/import before final placement; `backup.py:582–598,1021–1023` refuses incoming enrolled identities and enrolled restore targets before overlay. Ten profile-copy/restore tests exercise real disposable databases and ordinary compatibility. Arbitrary copies, custom/external paths, symlink aliases, online-copy races and snapshot rollback remain unsupported/UNKNOWN, not fenced authority continuity. |
| API, prospective coverage, instance identity, timestamp | Addressed in `PHASE3-HISTORY-FINISH-API-COVERAGE.md` and the actual facade/capture/reader. Board enrollment does not bind every task. `history_bound` starts task coverage; empty pages are not global vacancy. `runtime_id` is a caller-asserted instance identity, not authenticated identity or a uniqueness service. `observed_at` comes from original source `created_at`. Informational snapshots are not authority grants. |

### Corruption fixture integrity

I read all five history test files. The finish-owned-insert fixture corrections remove only the named owned-insert trigger and restore the captured SQL in `finally` (`test_kanban_history_strict.py:10–22`; `test_kanban_authority_history.py:231–240`). Record-update/defensive-repair fixtures likewise restore the exact captured trigger before their reader/write assertions. Fixtures that intentionally test a missing guard or altered schema are explicitly negative cases. No production trigger was weakened to make the final strict tests pass. Raw-insert refusal tests remain present; these are distinct from administrator-corruption setup.

## Independently verified frozen evidence

Verification here was read-only source/evidence inspection, SHA-256 recomputation, git inspection, and compile-without-imports/pyc. **No tests were rerun by this reviewer.** I inspected the launcher and final logs rather than reporting their results as my execution.

- `finish-freeze.json` and `finish-verification.json` have identical source maps. All **28** source/test/support/config inputs independently match their recorded hashes.
- All **64** preservation-set artifacts match; all **13** finish logs and **3** final documents match their recorded hashes. The preservation set is not asserted to consist exclusively of pre-worker artifacts.
- All **102** listed manifest paths exist. All **27** Python inputs compile without imports or bytecode writes.
- Independently confirmed expected branch/HEAD, empty index and clean `git diff --check`.
- Recomputed exactly the three installed-source hashes against `installed-before.json`: `hermes_cli/kanban_db.py`, `hermes_cli/kanban.py`, `hermes_constants.py` all match. **Not a whole-install or live-state attestation.**

Manifest SHA-256 anchors recomputed during this review:

- `finish-freeze.json`: `05be77a4681a0f80dae8dd49cb730e05528efac035c2800518889bf9ff3c08ea`
- `finish-verification.json`: `15e3106c8b399b5298ef5184c93c991579a33f396dd1f767be5413df8de3c0ba`

| Frozen log | Actual recorded result |
|---|---|
| `finish-frozen-history.log` | 120 passed; exit 0 |
| `finish-frozen-adjacent.log` | 96 passed, 1 POSIX-permissions skip; exit 0 |
| `finish-frozen-profiles.log` | 21 passed, 1 failed, 133 deselected; exit 1 |
| `finish-frozen-imports.log` | 1 passed; exit 0 |

Total, without counting intermediate reruns: **238 passed, 1 failed, 1 skipped, 133 deselected**. No all-green claim is justified. The profile failure is exactly `TestExportImport.test_export_default_handles_broken_symlinks`: WinError 1314 at `test_profiles.py:1439`, creating the fixture symlink, before production `export_profile` at `1448`. It remains a failed/unverified platform case, not a successful export test; it is not the reason for this REQUEST_CHANGES verdict.

The logs identify installed Python 3.11.4 used read-only, checkout-local tempfile and candidate imports. The import probe names backup, profiles, history, Kanban DB and constants inside this checkout; dashboard test loading explicitly targets the candidate plugin. This supports candidate API evidence, not a deployed service or real task-worker test. Existing process contention/self-exit probes are not power-loss, stress, receipt reconciliation or execution-tree exclusion proof.

## Exit gate and scope

Correct only the reservation retry regression, add its focused real-runtime tests, and refresh frozen verification using new unique labels without overwriting prior evidence. Then obtain independent acceptance of that corrected candidate before the subsequent quality-review gate. No broad redesign is requested.

Full enrolled HTTP/browser coverage, all legacy variants, dynamic-SQL exhaustiveness, full-stream performance, arbitrary-copy continuity and old-worker execution exclusion remain outside the demonstrated slice. Durable history alone does not authorize automatic recovery or replacement work.

**Only authored file:** `PHASE3-HISTORY-FINAL-SPEC-REVIEW.md`. No candidate source/test/evidence changes, runtime imports, database/profile writes, installs, task workers, PID signals, sends, staging, commits, pushes, merge or activation were performed by this reviewer.
