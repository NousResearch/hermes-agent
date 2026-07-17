# CPA-002 Control Plane Alignment Report

## Goal
Repair checker-reported evidence defects only for `FAIL_REPAIRABLE` in CPA-002/CPA-001 documents on branch `runtime/hermes-orch-control-plane-alignment`, without modifying source/tests or amending existing commits.

## What was done
1. Repaired only the three CPA markdown files:
   - `docs/evidence/HERMES-ORCH-FINISH-001/CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md`
   - `docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md`
   - `docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md`
2. Corrected candidate identity section to distinguish:
   - live base: `e9b8ae6be137abead6d19ed8a67c523f8c527096`
   - accepted source-stack head: `5ea4a27d9e31bc36d0f4c4cd0a7317e1c1ab3b79`
   - pre-repair evidence commit: `80304e0aaae1d8d45d0bc27897d17d07132affeb`
   - final repair commit: discoverable by `git log -1 --format=%H -- <three files>`
3. Updated path-change accounting from live base to pre-repair evidence commit; included all detected files in scope.
4. Added explicit SHA-256 evidence mappings for CPA documents as pre-repair values and stated that final post-commit hashes are external handoff outputs.
5. Added timestamped operational truth and candidate-vs-live routing distinctions, including process-tree, `PYTHONPATH`, and `auto_decompose` facts.
6. Clarified checker/test harness behavior and replaced implicit pass phrasing with a prescribed rerun command using `C:/Python314/python.exe`.
7. Expanded rollback plan to remove unsafe destructive checkout/reset claims and define a gated, non-destructive restore path requiring state inspection and backups.

## What was verified
- Pre-repair branch context command:
  - `git rev-parse --abbrev-ref HEAD` -> `runtime/hermes-orch-control-plane-alignment`
  - `git rev-parse HEAD` -> `80304e0aaae1d8d45d0bc27897d17d07132affeb`
- Source-stack and pre-repair evidence identity checks run against required commands.
- `py_compile` and `git diff --check` run after edits to enforce syntax and whitespace:
  - `PYTHONPATH='C:\Users\fallo\AppData\Local\hermes\worktrees\hermes-orch-control-plane-alignment' C:/Users/fallo/AppData/Local/Programs/Python/Python311/python.exe -m py_compile hermes_cli/kanban_db.py hermes_cli/kanban.py tools/kanban_tools.py && git diff --check`
  - exit `0`
- No source/test logic edits or pytest rerun were performed in this repair step.
- Checker-facing status remains `FAIL_REPAIRABLE` until the external checker reruns with the prescribed harness command.

## What failed
- Noted externally: first checker invocation under its sanitized profile failed before test collection because `pytest` was unavailable in the profile’s Python311 user-site. This is a verifier harness issue, not product behavior.

## Current exact state
- Candidate chain root: `e9b8ae6be137abead6d19ed8a67c523f8c527096`
- Source stack head: `5ea4a27d9e31bc36d0f4c4cd0a7317e1c1ab3b79`
- Pre-repair evidence commit: `80304e0aaae1d8d45d0bc27897d17d07132affeb`
- Final repair commit is not self-declared in this file; it must be read from:
  - `git log -1 --format=%H -- docs/evidence/HERMES-ORCH-FINISH-001/CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md`
- Working tree was repaired-only and remains clean after commit.
- Runtime actions in this repair lane: none.
- Remote operations in this repair lane: none.
- No rollback/activation/restart/db/config/source edits were executed.

## Remaining blockers
- External checker rerun is still required using the prescribed `C:/Python314/python.exe` harness command.
- CPA-002 cannot be treated as independently accepted in this lane; this is evidence-packaging and checker repair only.

## Next actionable step
1. Re-run checker with command below and confirm `FAIL_REPAIRABLE` cleared:
   ```bash
   cd C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-orch-control-plane-alignment && PYTHONNOUSERSITE=1 PYTHONPATH='C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-orch-control-plane-alignment;C:/Users/fallo/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages' C:/Python314/python.exe -m pytest tests/hermes_cli/test_kanban_task_contract.py tests/hermes_cli/test_kanban_task_admission.py tests/hermes_cli/test_kanban_notify_inheritance.py tests/hermes_cli/test_kanban_task_inspection.py tests/tools/test_kanban_child_policy.py -q -n 0
   ```
   - Why this exact harness is required: checker subprocesses sanitize inherited `PYTHONPATH` and user-site; `C:/Python314/python.exe` can import `pytest 9.1.1` only when the known live venv site-packages path is explicitly added, while candidate remains first and an import probe resolved `hermes_cli` under the candidate. This is checker-harness wiring only, not source behavior evidence or an independent PASS.
2. Re-verify checkpointed path/evidence fields if checker still reports any mismatch.
3. Do not claim CPA-002 independent acceptance until checker clears.

## Evidence
### Commit identities
- live base: `e9b8ae6be137abead6d19ed8a67c523f8c527096`
- accepted source-stack head: `5ea4a27d9e31bc36d0f4c4cd0a7317e1c1ab3b79`
- pre-repair evidence commit: `80304e0aaae1d8d45d0bc27897d17d07132affeb`
- final repair commit: `$(git log -1 --format=%H -- docs/evidence/HERMES-ORCH-FINISH-001/CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md)`
- accepted upstream commit patch-id pairs:
  - `c50e6bf...` -> `38b39d594d335f8255664560f5a404cf53ed681b`
  - `f093334...` -> `6812153f5e0a79f68a3ae47d71eaca1636bdb936`
  - `142ba8d...` -> `7bb1ba3e883675e32b8c690a894e240bc7d4fb2b`
  - `8f300d...` -> `a75322572a80eba461c5c38d3fb199e2f9a739ff`

### Pre-repair changed paths base..pre-repair evidence commit
Observed from `git diff --name-only e9b8ae6be137abead6d19ed8a67c523f8c527096..80304e0aaae1d8d45d0bc27897d17d07132affeb`:
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-A-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-B-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-C-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-D-REPORT.md`
- `hermes_cli/kanban.py`
- `hermes_cli/kanban_db.py`
- `tools/kanban_tools.py`
- `tests/hermes_cli/test_kanban_notify_inheritance.py`
- `tests/hermes_cli/test_kanban_task_admission.py`
- `tests/hermes_cli/test_kanban_task_contract.py`
- `tests/hermes_cli/test_kanban_task_inspection.py`
- `tests/tools/test_kanban_child_policy.py`

### Evidence-commit changed paths (repair commit)
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md`
- `docs/evidence/HERMES-ORCH-FINISH-001/CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md`

### Verification commands and outcomes
- py_compile + git diff --check: pass, exit `0`
- No pytest rerun in this lane.
- No git amend/rebase/checkout reset of candidate branch executed.
- Working state remained clean after staging only three CPA paths.

### SHA-256 mappings (pre-repair document hashes)
- `CPA-001-ORCH-001-RUNTIME-ALIGNMENT-CONTRACT.md`: `d417cdf84eb468cd82742849522605b292e56a072f9ff76ac01db1b548f63318`
- `CPA-002-CONTROL-PLANE-ALIGNMENT-REPORT.md`: `3f010edae44c0ea8c0686b6ce5ca8d0eadd00765e58b346d974d6d8f8529da88`
- `CPA-002-CONTROL-PLANE-ALIGNMENT-ROLLBACK.md`: `2bbe344e324527334df914619f4211a83c349cf47c1aec22bd9e1ca60eb61c47`

### Operational disclosures (for audit)
- timestamped operational truth (live) `2026-07-17T15:37:23Z`:
  - PIDs `11896` and `19416` exist; `19416` is child of `11896`.
  - both cwd `C:/Users/fallo/AppData/Local/hermes`
  - PID 19416 `PYTHONPATH`: `C:/Users/fallo/AppData/Local/hermes/hermes-agent;C:/Users/fallo/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages`
  - PID 23900 absent
  - editable finder maps top-level packages to live source
- `default` and `builder-grok` `kanban.auto_decompose`: `false`
- live `git status --porcelain=v1` tracked/untracked snapshot: untracked only
  - `docs/evidence/HERMES-ORCH-001-REPORT.md`
  - `docs/evidence/HERMES-ORCH-FINISH-001-HOF-000-RECOVERY.md`
  - `docs/evidence/HERMES-ORCH-FINISH-001-RUNTIME-INVENTORY.md`
  - `docs/evidence/HERMES-ORCH-FINISH-001/`
  - no tracked modifications
- all-board read-only title query: no matching new CPA/HKNIR/HOF-020A/HOF-020-R1/HOF-029 cards
