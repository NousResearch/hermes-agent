# Phase 3 authority-history — independent code-quality/security review

## Verdict: REQUEST_CHANGES

Two concrete P2 regressions remain in the new copy/restore integration. Both were reproduced against actual frozen candidate imports in disposable local state. The reservation retry fix is not disputed. This is an independent generic Hermes review, not BUREAU/CLIENT, not Phase 3 completion, and **not activation approval**.

Candidate: `C:/Users/sibag/hermes-phase3-authority-history`, branch `feat/phase3-authority-history`, HEAD `09109fec98016ffd7fef8622223073d296c02fa4`.

## Ranked defects

### 1. P2 — clone-all can return another concurrently created profile and strip its runtime files

**Paths:** `hermes_cli/profiles.py:1039–1041,1059–1074`, particularly `shutil.move(str(staged), str(profile_dir))` at line 1071.

The initial destination-exists check precedes the potentially long full-profile staging copy. If another creator makes `profile_dir` before final placement, `shutil.move` treats that existing directory as a container: it moves the staged directory **inside** it, rather than refusing the collision. The function ignores the move's actual destination, strips `_CLONE_ALL_STRIP` from the competing profile at the original `profile_dir`, performs subsequent profile initialization there, and returns that competing profile as the successful clone.

This changes the old `copytree(source_dir, profile_dir, ...)` behavior: its destination creation fails on an existing directory instead of accepting it as a container. This is a destination-creation race, not the explicitly excluded source online-copy/authority-continuity problem. It also affects ordinary, unenrolled profiles.

**Executed deterministic reproduction:** under `.phase3-evidence/state/quality-copy-01`, created an ordinary source containing `config.yaml`. Called the real `profiles.create_profile('quality-target', clone_all=True, no_alias=True)`. Redirected `get_profile_dir` to the disposable target and mocked gateway service registration. Wrapped the real `refuse_authority_copy`: after its successful validation and before final move, created the destination with `config.yaml = 'competitor: true'` and a `gateway.pid` file containing a disposable non-PID sentinel. This models a competing creator without launching any process/service or touching real runtime files.

Actual output (probe exit 0):

```text
CLONE_RETURNED True
TARGET_CONFIG competitor: true
NESTED_CLONE True
COMPETITOR_PID_PRESERVED False
```

`NESTED_CLONE` checks `target/quality-target/config.yaml`. Thus the call reported success for the wrong profile, put the requested clone at the wrong depth, and removed the competing profile's sentinel. This is not merely a theoretical TOCTOU warning.

**Recommendation:** install the staged profile with exclusive destination ownership/no-clobber semantics; never use directory-container move semantics for the final profile path. Ensure a collision aborts before any stripping, metadata initialization, or service-registration action on the destination. A second `exists()` check alone does not close the race. Preserve the staged authority check and cross-filesystem handling. Add a deterministic regression inserting a competing destination between staging/validation and placement, asserting failure and byte-for-byte preservation of all competing files (including runtime markers).

**Coverage gap:** the new profile-copy matrix tests preexisting source authority and absent-target success/refusal, not a destination appearing during staging.

### 2. P2 — ZIP preflight loses destination scope and rejects unrelated skill/plugin databases

**Paths:** `hermes_cli/backup.py:587–596`; contrast `hermes_cli/kanban_history.py:182–212`, especially candidate discovery at 190–198.

The ZIP preflight picks **every** member whose basename is `kanban.db`, regardless of its eventual destination. It relocates each to `<temporary-directory>/<index>/kanban.db` and applies a checker that treats a root `kanban.db` as an authoritative-board candidate. Consequently, an opaque asset/database below `skills/` or another non-board subtree is reclassified as a standard board DB. A non-SQLite fixture, unrelated application schema using an `authority_*` table, or retained authority test fixture can abort the entire restore before even restoring configuration.

The documented bounded discovery contract is root `kanban.db`, `kanban/**/kanban.db`, and nested profiles; external/custom locations are unsupported, not automatically classified as authority. `refuse_authority_copy(incoming_tree)` correctly ignores the example below, while ZIP import rejects it solely because staging erased its path context. This defect does not require arbitrary-copy continuity, symlinks, a concurrent writer, or an enrolled live target.

**Executed reproduction with a valid SQLite asset:** under `.phase3-evidence/state/quality-copy-02`:

1. Created `incoming/skills/example/assets/kanban.db` using real SQLite, with only `CREATE TABLE authority_roles(name TEXT)` (an unrelated application table).
2. Called real `refuse_authority_copy(incoming)`; it accepted the tree.
3. Created a ZIP containing `config.yaml` and that database at `skills/example/assets/kanban.db`.
4. Called real `backup.run_import(SimpleNamespace(zipfile=..., force=True))`, redirecting only `get_default_hermes_root` to a fresh disposable target.

Actual output (probe exit 0, expected exception caught for observation):

```text
STANDARD_LOCATION_CHECK accepted unrelated valid SQLite asset
Backup contains 2 files
ZIP_IMPORT AuthorityHistoryError missing or modified authority history schema: authority_history_meta
RESTORED_CONFIG False
```

An earlier separate probe with an opaque non-SQLite skill fixture also failed with `AuthorityHistoryError cannot establish copied authority identity`. The valid-SQLite case above avoids relying on whether backup generation accepts an opaque `.db` file.

**Recommendation:** classify members using their normalized, prefix-stripped destination-relative path and the same standard-location rules as profile/quick-restore discovery, *before* mapping selected databases to generated safe staging names. Keep archive-path traversal defenses and sidecar grouping; do not solve this by permitting malformed databases at actual standard board locations. Add ordinary ZIP-import cases for an unrelated valid SQLite asset and an authority fixture outside standard locations, plus standard root/named-profile/board refusal cases and wrapped-prefix archives.

**Coverage gap:** all new ZIP/quick-copy tests place their test DB at root `kanban.db`; none tests a same-basename asset outside standard board locations.

## Other quality observations — not additional blockers

### Full-history cost now affects every enrolled-board write

`kanban_history.py:370–426,429–440,555–569` and `kanban_db.py:2337–2342` validate the entire retained stream both before and after every audited write, inside `BEGIN IMMEDIATE`. `_validate_record` issues retained-provenance and binding lookups per record. This includes non-ownership writes such as `set_branch_name`, and boards retain history indefinitely. The strict reader also decodes/materializes the full stream before slicing a page.

A small real-candidate probe (`quality-probe-01`) enrolled one task, generated informational records through `_append_event` inside real audited transactions, and measured one `set_branch_name` at each size using `set_trace_callback` and `perf_counter`:

| Retained records | Traced SQL statements for one write | Observed seconds |
|---:|---:|---:|
| 1 | 39 | 0.0035 |
| 1001 | 4039 | 0.0993 |
| 5001 | 20039 | 0.3250 |

These are single local measurements with tracing enabled, not a production benchmark or timeout proof. They establish linear historical work for an otherwise constant-size write, not an inferred architecture concern. Repeated appends accumulate superlinear total work over board lifetime. Since whole-stream validation and unbenchmarked cost were explicitly disclosed in the accepted bounded contract, this is a nonblocking capacity concern here. Establish a realistic history-size/write-latency budget and regression benchmark before treating this as production-ready; any optimization must preserve strict historical-corruption refusal rather than silently validating only a cursor suffix.

### Ordinary raw-SQL compatibility has a connection-lifecycle caveat

Persistent task/run triggers reference `authority_owned_writer`, but that UDF is only registered when `owned_writer` first runs (`kanban_history.py:54–63,86–102`). In `quality-probe-01`, after creating an ordinary task and closing the connection, a fast-path `kb.connect(path)` followed by `UPDATE tasks SET priority=priority WHERE id=?` failed with `OperationalError: no such function: authority_owned_writer`. SQLite resolves the function even though the trigger's enrollment predicate would be false. A subsequent real `kb.set_branch_name` succeeded because it registers the UDF through `write_txn`.

This is recorded as a compatibility caveat, not a third blocker: the reviewed production mutation entry points use the owned transaction boundary, and I did not establish a supported ordinary API regression from this raw-SQL probe. If ordinary direct SQLite callers are intended to remain compatible, add explicit fresh/additional-connection coverage and clarify the boundary. Do not weaken enrolled-write refusal.

## Review coverage and favorable findings

Read the requested BRIEF, FINISH-API-COVERAGE, RETRY-RESULT, FINAL-SPEC-REVIEW and RETRY-SPEC-REVIEW; read the repository development/testing guidance and the isolation launcher. Inspected all production diffs in `kanban_db.py`, `profiles.py`, `backup.py`, dashboard `plugin_api.py`, the complete new `kanban_history.py`, all five new history test files, and both changed existing test diffs. Inspected surrounding initialization/migration, audited transaction, profile creation and ZIP import code rather than relying on prior verdicts.

- The journal stays in the board's SQLite database and the existing transaction mechanism. No second scheduler, after-commit journal publication, or automatic recovery was introduced.
- Capture/audit exceptions derived from `Exception` are inside the existing rollback path; operational mutations, retained bindings, source provenance and journal writes share the transaction. The owned-writer gate is disabled in `finally`. Immutable/update/delete/replacement guards remain intact.
- Enrollment/removal reservation is serialized, and the retry now checks for an existing marker before inserting. The new failed-removal tests actually retry archive/destructive removal.
- Strict retained records exclude arbitrary prose/event payloads, raw claim credentials and credential hashes. Public identifiers remain caller-asserted non-secret data as documented, not authenticated identity. No new outbound calls or embedded credentials were found in the production diff.
- Reader validation checks the full retained sequence and immutable provenance instead of silently skipping bad older records. Administrator corruption fixtures are distinguished from production access and restore the intentionally removed guards for validation assertions.
- Dashboard ownership-changing direct statuses now use `_append_event`; the inspected new tests exercise enrolled direct status and child demotion, not merely ordinary REST responses.
- The changed existing board test closes its actual SQLite handle before filesystem removal. The boundary-only fake adjustment is explicit; it does not replace the new real-SQLite authority tests.

These favorable findings do not imply exhaustive caller coverage, full legacy-corruption coverage, administrator-proof tamper resistance, power-loss durability, or deployed execution-tree exclusion.

## Freeze, recorded verification and independent execution

Independently recomputed:

- `retry-freeze.json` and `retry-verification.json` source maps are equal; all **29** explicit source/test/support inputs match current bytes.
- All **103** current preservation-map entries match `retry-verification.json`; comparison with `retry-baseline.json` differs only for `hermes_cli/kanban_history.py` and `tests/hermes_cli/test_kanban_history_lifecycle.py`.
- Every entry in the retry-evidence hash map matches. All **28** Python inputs compile without imports or bytecode writes.
- Expected branch/HEAD, empty staged diff and clean `git diff --check` confirmed.

Manifest anchors:

```text
retry-freeze.json
73d07bdedfb2671400dc98b1c51e29aeeb5220e8a8a7356aad298adc2c13cc1b
retry-verification.json
65f6e959120f68e72be30ccef06b56c41c2c9019975a5d1c0e7d6094f3c743b0
```

Read the four frozen execution logs directly. Their recorded total is **239 passed, 1 failed, 1 skipped, 133 deselected**: history 121 passed; adjacent 96 passed/1 skipped; profiles 21 passed/1 failed/133 deselected; imports 1 passed. These are prior recorded runs, **not a suite rerun by this reviewer**. The profile failure remains WinError 1314 during broken-symlink fixture creation, before export; the POSIX-permissions case remains skipped. Neither is relabeled successful or used to explain away the new defects.

This reviewer executed three small `python -B -c` child probes, not pytest suites. Each loaded the reviewed launcher with `runpy`, redirected its `STATE` global **in memory** to a fresh unique `quality-*` subtree before calling `environment()`, and used its configured installed venv interpreter read-only. No launcher bytes were changed. Children inherited the launcher's allowlisted/scrubbed environment, local HOME/USERPROFILE/HERMES_HOME/APPDATA/LOCALAPPDATA/TEMP/TMP/TMPDIR, disabled bytecode/user-site imports, and asserted candidate module/temp origins. No preexisting evidence was overwritten. The probes' real outputs are quoted above; the commands were executed through the review tool transcript rather than saved as new source/test files.

Disposable state roots created:

- `.phase3-evidence/state/quality-probe-01/` — isolated ordinary/enrolled SQLite and performance probe.
- `.phase3-evidence/state/quality-copy-01/` — isolated ZIP asset and deterministic competing-profile placement probe.
- `.phase3-evidence/state/quality-copy-02/` — isolated valid-SQLite asset ZIP probe.

No real worker/service launches, PID signals, sends, network, installs, git writes, installed-source modifications or live board/profile/config writes occurred. The runtime-marker removal reproduced above affected only a newly created disposable sentinel, not a process identifier. This is cooperative containment, not an OS sandbox or installed-runtime attestation. A read-only evidence-check attempt initially resolved evidence basenames against the wrong directory; rerunning with `.phase3-evidence/` resolved it and confirmed the hashes.

## Requested exit gate

Fix the two copy/restore regressions without expanding the authority/recovery contract. Add focused collision and destination-scope tests, rerun the bounded adjacent/copy selections with new unique evidence labels, freeze the corrected bytes, and obtain independent review. No source fix, schema weakening, automatic decommissioning/recovery, activation or merge is approved by this report.

**Only authored deliverable:** `PHASE3-HISTORY-QUALITY-REVIEW.md`. Existing candidate source/tests and frozen evidence are unchanged; only the explicitly permitted disposable local probe state was additionally created.
