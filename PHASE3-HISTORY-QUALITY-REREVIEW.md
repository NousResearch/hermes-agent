# Phase 3 authority-history — independent quality re-review

## Verdict: PASS (the two scoped copy/restore repairs)

No concrete remaining defect was found in either repair within the accepted bounded history/copy contract. Both P2 findings in `PHASE3-HISTORY-QUALITY-REVIEW.md` are closed for this frozen candidate. This is independent generic quality review, not BUREAU/CLIENT, full-suite certification, merge authorization, Phase 3 completion, or activation approval.

Candidate: `C:/Users/sibag/hermes-phase3-authority-history`; branch `feat/phase3-authority-history`; HEAD `09109fec98016ffd7fef8622223073d296c02fa4`.

## Repair findings

### 1. Clone-all destination collision — resolved

Inspected `hermes_cli/profiles.py:1039–1078` and subsequent initialization. The final placement at line 1075 is now `shutil.copytree(staged, profile_dir, symlinks=True)` with default exclusive destination creation. It is not `shutil.move`, a permissive merge, or merely a second existence check. A competing destination causes `FileExistsError` before runtime stripping, metadata initialization, or service registration. The competing directory cannot become a container for the staged clone.

The staged source is still checked by real `refuse_authority_copy` before placement. Source-copy ignore behavior and symlink-copy policy remain intact. Copy placement does not require a cross-device rename.

The dedicated regression at `tests/hermes_cli/test_kanban_history_copy_qualityfix.py:12–39` injects the competing creation after real staged validation, calls real `create_profile`, and checks the complete competing file set and exact config, environment and non-PID sentinel bytes, no registration, and the collision exception. Lines 136–153 exercise real copy I/O with rename forced to raise EXDEV. This closes the reproduced regression rather than masking its output.

Exclusive creation is not atomic publication of an entire tree. A later I/O error can leave an owned partial destination; a second copy costs extra I/O. These are disclosed limitations, not a remaining collision defect or a claim of recovery/hostile-filesystem protection.

### 2. ZIP database destination scope — resolved

Inspected `hermes_cli/backup.py:582–605` against actual extraction at lines 618–687 and `hermes_cli/kanban_history.py:182–234`. ZIP preflight now strips the detected prefix, computes the resolved destination-relative path under the same containment boundary as extraction, and classifies that path before assigning generated staging names.

The shared `is_standard_kanban_database` recognizes root `kanban.db`, `kanban/**/kanban.db`, and those locations under recursively nested `profiles/<name>` roots, using host-native case normalization. Tree discovery retains its existing recursion/candidate enumeration and applies the same predicate. Skill/plugin assets are no longer promoted to root boards merely because their basename is `kanban.db`.

Target authority refusal still precedes any overlay. Selected incoming databases still undergo strict authority/schema checks; malformed standard-location databases are not allowed through. Exact-member WAL/SHM/journal grouping and generated staging paths remain. External-provider handling remains aligned with its separate extraction branch; escaping members are still blocked by extraction.

Dedicated tests cover ordinary SQLite, unrelated authority-named tables and enrolled fixtures outside standard locations; root/named/nested-profile enrolled and malformed boards; wrapped archives; in-root `skills/../kanban.db` normalization; enrollment present only in a real WAL; and traversal preserving an outside disposable sentinel. These exercise real candidate import and SQLite/file behavior, not mocked authority decisions.

## Guard preservation and regression boundary

The independent hash comparison against the repair baseline identifies exactly three changed preserved files: `hermes_cli/backup.py`, `hermes_cli/kanban_history.py`, and `hermes_cli/profiles.py`. The journal/transaction implementation, dashboard production input, existing history tests, reservation-retry tests and prior evidence remain byte-identical in the preservation map. The inspected history addition is the location classifier and its use in copy discovery; reservation retry, enrollment, schema/immutability guards, writer boundary and retained-history behavior are not weakened by these repairs.

Previously accepted full-history cost and fresh-connection raw-SQL UDF caveats remain nonblocking observations, not newly resolved capabilities. Arbitrary-copy/custom-location, symlink-alias, online-copy/rollback continuity and archive-alias sidecar coherence exclusions are not expanded into supported guarantees by this verdict. The EXDEV test is not a physical multi-volume test.

## Independent freeze verification actually executed

Read `qualityfix-verify.py` and the existing parent-completed `qualityfix-verification.json`. Did **not** rerun the manifest-writing script or infer success from the repair worker's timeout. The parent completion is corroborated by the existing manifest and an independently executed read-only `python -B -c` check, exit **0**:

```text
PASS 32 inputs 117 preserved paths 12 evidence hashes 31 compile checks; installed 3 match
```

That check recomputed all current source and preservation hashes, required source-map equality with `qualityfix-freeze.json`, checked baseline preservation-map keys and exactly the three allowed differences, verified all twelve preceding repair evidence hashes and the repair report hash, compiled Python inputs without imports/bytecode writes, and confirmed expected branch/HEAD, empty index and clean `git diff --check`.

Manifest anchors independently recomputed:

```text
qualityfix-verification.json
99df0cb4e43e7ad36cf945268b411684423351231189a39e79436d75caf95a53
qualityfix-freeze.json
0019bf4fbc92d16cd2bca284039b5964c486d221173708adc604a818dd1f5600
```

The installed comparison was read-only and limited to the recorded three files (`hermes_cli/kanban_db.py`, `hermes_cli/kanban.py`, `hermes_constants.py`), not the whole installation or live state. The manifests are bounded explicit input maps, not exhaustive transitive dependency attestation.

## Recorded execution, not a reviewer suite rerun

Read the actual frozen logs, candidate module origins, local temporary roots and exit markers, plus the focused defense log and runtime wrapper:

| Selection | Recorded result | Exit |
|---|---|---|
| Six history files, including repair tests | 164 passed | 0 |
| Bounded adjacent selection | 96 passed, 1 skipped | 0 |
| Profiles `clone_all or import` | 21 passed, 1 failed, 133 deselected | 1 |
| Import-origin smoke | 1 passed | 0 |
| Dedicated repair/defense tests (overlaps history total) | 43 passed | 0 |

The four frozen selections retain the reported **282 passed, 1 failed, 1 skipped, 133 deselected — not all green**. The failure remains `test_profiles.py:1439`, `TestExportImport.test_export_default_handles_broken_symlinks`: WinError 1314 at fixture `symlink_to`, before production export. The skip remains `test_backup.py:846`, POSIX file permissions only. Neither is relabeled successful, blamed on a timeout, or treated as a new repair regression.

No additional runtime tests were needed for this focused re-review. No test logs, manifests, source, tests, installed files, live boards/profiles/configs or runtime state were changed. No workers/services, PID signals, network, installs, git writes or sends were performed.

## Deliverable and issues encountered

Only authored file: `PHASE3-HISTORY-QUALITY-REREVIEW.md`. The shorthand `COPY-REPAIR.md` resolved to the actual `PHASE3-HISTORY-COPY-REPAIR.md`. A search-tool Windows path-resolution error was bypassed by direct reads of the logs; it did not block verification. No implementation or activation was performed.
