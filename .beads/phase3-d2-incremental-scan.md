---
id: phase3-d2-incremental-scan
title: Phase 3 D2 — scan_project.py: --incremental flag for fingerprint-aware scanning
status: completed
executor: delegate-coder
parallel_safe: false
base_branch: docs/ua-flywheel-phase3-plan
allowed_files:
  - scripts/code-scan/scan_project.py
  - tests/code_scan/test_scan_project.py
  - tests/code_scan/fixtures/incremental/
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - scripts/code-scan/extract_imports.py
  - scripts/code-scan/fingerprints.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - .hermesignore
  - skills/
depends_on:
  - phase1-code-scan-completion-fix
  - phase3-d1-fingerprint-model
verification:
  - python -m pytest tests/code_scan/test_scan_project.py -v
  - python scripts/code-scan/scan_project.py /home/jarrad/work/testbeds/ua-flywheel/cass_memory_system --incremental --output /tmp/incr-scan.json
risk: medium
---

# Phase 3 D2 — scan_project.py: --incremental Flag for Fingerprint-Aware Scanning

## Context & Intent

**Why this bead exists.** Phase 1 `scan_project.py` performs a full deterministic walk of the project tree on every invocation. For large projects re-scanned frequently, this is wasteful when most files are unchanged. Phase 3 D1 provides fingerprints and change classification. D2 wires fingerprints into the scanner so that `--incremental` mode performs the same deterministic file walk as full mode, builds current fingerprints, classifies each file, persists updated fingerprints, and exposes the STRUCTURAL subset for downstream heavy analysis while keeping the scan output schema fresh and identical to full mode.

**Authority docs.** `.plans/phase-3-incremental-analysis.md` (D2 section) defines the high-level scope. `.plans/ua-incorporation-strategy.md` (§3. Fingerprint-Based Incremental Re-Analysis) specifies the `--incremental` flag behavior.

**Intent.** Modify `scripts/code-scan/scan_project.py` to:
1. Accept an `--incremental` argparse flag.
2. When `--incremental` is set and a valid fingerprints file exists, load it, perform the normal fresh file walk, build current fingerprints for all files, compare to classify, and report counts/paths for UNCHANGED/COSMETIC/STRUCTURAL files.
3. Emit fresh scan results from the current file tree; do not reconstruct scan records from fingerprints. The output JSON schema remains identical to a full scan, with only additional warning/metadata entries allowed.
4. Save the updated fingerprints (with classifications removed) to `.hermes/code-state/fingerprints.json` after the scan completes.
5. Accept an `--full` flag that forces full-mode behavior regardless of existing fingerprints; it may refresh fingerprints if the implementation shares the persistence path, but it must not require fingerprints to exist.

**Non-goals.** No tree-sitter/WASM. No new runtime dependencies. No dashboard, React UI, auto-injection, SQLite, CLI command. No changes to existing Phase 1 output schema. This bead DOES modify `scan_project.py` and its tests — this is explicitly authorized.

## Implementation Details

### Target files

| File | Purpose | Max LOC change |
|---|---|---|
| `scripts/code-scan/scan_project.py` | Add `--incremental` and `--full` flags, conditional logic, classification behavior | +60-80 LOC |
| `tests/code_scan/test_scan_project.py` | Add tests for incremental mode, merge correctness, full override | +80-100 LOC |
| `tests/code_scan/fixtures/incremental/` | Fixture project for incremental testing | varies |

### CLI changes to `scan_project.py`

Add two new argparse arguments to `main()`:

```python
parser.add_argument(
    '--incremental',
    action='store_true',
    help='Use fingerprints for incremental scan; classify UNCHANGED/COSMETIC files',
)
parser.add_argument(
    '--full',
    action='store_true',
    help='Force a full scan, ignoring any existing fingerprints',
)
```

### Behavioral changes in `main()`

Current flow (Phase 1/2) is:

```
parse_args → load_hermesignore → walk_project → detect_frameworks → build_summary → output
```

Modified flow when `--incremental` is true:

```
parse_args
if args.full → force full scan (same as current flow)
elif args.incremental:
    load_fingerprint_file(...)
    if no valid fingerprints → treat as full scan, save fingerprints afterward
    else:
        walk_project → collect all file records as before
        build_fingerprint_map(all_files)  # fresh fingerprints
        compare_fingerprints(old=loaded, new=fresh)  # classify each file
        separate files into UNCHANGED/COSMETIC/STRUCTURAL
        mark only STRUCTURAL files as needing downstream heavy analysis: for each, re-read line count, detect framework changes
        merge: take STRUCTURAL file records from fresh scan, UNCHANGED/COSMETIC from current scan records
        build_summary with merged files → output
        save updated fingerprints (without change_level classification)
else:
    # No incremental flag
    walk_project → detect_frameworks → build_summary → output (original flow)
```

### Merge logic (critical)

When `--incremental` produces a incremental result, the output `files` array must contain the SAME RECORDS as a full scan would produce, with these exceptions:

| Field | Source in incremental mode |
|---|---|
| `path` | Fresh `_walk_project()` result from current tree |
| `relative_path` | Fresh `_walk_project()` result from current tree |
| `language` | Fresh `_walk_project()` result from current tree |
| `category` | Fresh `_walk_project()` result from current tree |
| `lines` | Fresh `_walk_project()` result from current tree |
| `size_bytes` | Fresh `_walk_project()` result from current tree |
| `frameworks` | Fresh `detect_frameworks()` result from current tree |
| classification metadata | Derived from old-vs-new fingerprints and reported separately from the Phase 1 file records |

The `scanned_at` timestamp in the output JSON should always be the current time (not the original fingerprint capture time). The `warnings` array should include a note like `"incremental_scan: 42 files skipped (UNCHANGED), 5 files scanned (STRUCTURAL)"` when in incremental mode.

### Fingerprint save after incremental scan

After a successful incremental scan, the fingerprints file must be updated to reflect the fresh fingerprints of all current files. The `change_level` field must NOT be persisted — it exists only for the merge classification step.

### Allowed modifications to `scan_project.py`

This bead explicitly authorizes modifying `scripts/code-scan/scan_project.py`:
1. Add `--incremental` and `--full` argparse arguments.
2. Add `from fingerprints import get_fingerprint_path, load_fingerprint_file, save_fingerprint_file, build_fingerprint_map, compare_fingerprints` import (after the existing `from language_registry import ...` line).
3. Add `_run_incremental_scan()` helper function that encapsulates the incremental logic.
4. Modify `main()` to dispatch to `_run_incremental_scan()` when `--incremental` is set and `--full` is not.
5. No changes to `_walk_project()`, `_build_summary()`, `_load_hermesignore()`, `_is_ignored()`, `_is_hardcoded_dir_excluded()`, or `_count_lines()` — these functions are not modified.

### New helper function

| Function | Purpose |
|---|---|
| `_run_incremental_scan(target_dir: str, args: argparse.Namespace, patterns: List[str]) -> int` | Full incremental scan workflow. Returns exit code (0 = success). Encapsulates: load fingerprints, classify, merge, save, output. |
| `_merge_scan_files(fresh_files: List[dict], persisted_fps: dict, project_root: str) -> List[dict]` | Merge fresh scan results with persisted fingerprint data. Returns merged file list matching Phase 1 output schema. |

### Test fixture requirements

`tests/code_scan/fixtures/incremental/` must contain a small project with:

| File | Purpose |
|---|---|
| `main.py` | Baseline file with known functions/imports |
| `utils.py` | Second file — will be "touched" in test scenarios |
| `config.json` | Non-code file — should always be UNCHANGED in tests |
| `.hermesignore` | Default rules |

Test scenarios using this fixture:

1. **No prior fingerprints:** `--incremental` with no fingerprints file → behaves like full scan, creates fingerprints.
2. **All UNCHANGED:** Run `--incremental` twice with no file changes → second run skips all files, output matches first run.
3. **One STRUCTURAL:** Modify `main.py` to add a function → `--incremental` re-scans only `main.py`, preserves `utils.py` and `config.json`.
4. **Full override:** `--full` ignores fingerprints and performs full scan.

## Complexity Tier

**T3** — Modifies an existing file requiring careful logic insertion without breaking Phase 1 behavior. The classification/persistence logic has subtle correctness requirements (identical output schema, freshly emitted data from fingerprints, fresh data for structural files). Requires careful test coverage of all code paths. Estimated 10-12 subagent iterations. Requires coder subagent + Hermes verification + reviewer signoff before presentation for commit approval.

## Execution Engine

**Executor:** `delegate-coder` — Hermes dispatches a coder subagent via `delegate_task` with full plan context.

**Process:**
1. Coder subagent creates fixture files and test scaffolding first, then writes tests that fail (RED).
2. Implement incremental logic in `scan_project.py` to pass tests (GREEN).
3. Hermes runs `pytest tests/code_scan/test_scan_project.py -v` — all must pass including existing Phase 1 tests (FULL).
4. Hermes runs end-to-end: full scan → save → modify one file → incremental scan → compare outputs.
5. Reviewer subagent validates: spec compliance, schema equivalence (incremental output = full scan output for data portion), Phase 1 regression, scope guardrails.
6. Hermes presents evidence bundle to JC — this bead authorizes implementation only, not commit/push.

**Subagent reliability preflight:**
- Task shape: modify existing script + classification/persistence logic + integration tests
- Expected artifacts: 1 modified source file, 1 modified test file, fixture directory
- `max_iterations`: 15 per subagent dispatch
- File-write: YES. Run-test: YES. Commit: NO (await JC approval).

## Required Inline Context

### Project context

- **Repo:** `/home/jarrad/.hermes/hermes-agent`
- **Current branch:** `docs/ua-flywheel-phase3-plan`
- **Parent bead:** `phase3-d1-fingerprint-model` — fingerprints.py module must exist and pass before this bead executes
- **No new pip dependencies** — stdlib only
- **D1 dependency:** `scripts/code-scan/fingerprints.py` must exist with the functions defined in the D1 bead

### Workspace cleanliness requirement

Phase 3 execution starts from a clean worktree for each bead. The executor must capture `git status --short` before and after the bead. Any file outside `allowed_files` is a scope violation and must be reverted before review.

### Scope guardrails

This bead may ONLY create or modify files listed in `allowed_files`. Specifically:
- `scripts/code-scan/scan_project.py`: ALLOWED — add `--incremental` and `--full` flags + classification/persistence logic.
- `tests/code_scan/test_scan_project.py`: ALLOWED — add tests for incremental mode.
- `scripts/code-scan/extract_imports.py`: FORBIDDEN — no changes.
- `scripts/code-scan/language_registry.py`: FORBIDDEN — no changes.
- `scripts/code-scan/graph_schema.py`: FORBIDDEN — no changes.
- All other Phase 1/2 files: FORBIDDEN.

### Preservation requirement

ALL existing Phase 1 test cases must continue to pass. The incremental additions must not alter the behavior of `scan_project.py` when invoked WITHOUT `--incremental`.

## Dependencies

| Dependency | Type | Status |
|---|---|---|
| Phase 1 code-scan scripts | prerequisite | Completed, committed, verified (80 tests pass) |
| Phase 3 D1 fingerprints module | prerequisite | Must be complete and passing before D2 execution |
| `scan_project.py --incremental` runtime behavior | test input | Created by this bead |
| Python 3.10+ stdlib | runtime | Assumed present |
| pytest | test runtime | Assumed present |

## Test Obligations

### TDD protocol

Strict RED → GREEN → REFACTOR for each new code path. Every new function and branch must have at least one failing-first test. Existing tests must not regress.

### Test file mapping

| Source | Test | Coverage target |
|---|---|---|
| `--incremental` without prior fingerprints | `test_scan_project.py` | Behaves like full scan; creates fingerprints on disk |
| `--incremental` with no changes | `test_scan_project.py` | All files classified UNCHANGED; output identical to full scan (excluding timestamps) |
| `--incremental` with one structural change | `test_scan_project.py` | Only changed file re-scanned; UNCHANGED files freshly emitted; output schema identical |
| `--full` flag | `test_scan_project.py` | Ignores fingerprints; performs full scan |
| Classification correctness | `test_scan_project.py` | Output has fresh file records and correct classifications for both STRUCTURAL and UNCHANGED files |
| Summary warnings in incremental mode | `test_scan_project.py` | "incremental_scan" warning present in output warnings array |
| Regression: existing tests | `test_scan_project.py` | All Phase 1 tests still pass |

### RED/GREEN/FULL evidence required

- **RED:** New tests fail because `--incremental` flag doesn't exist or classification/persistence logic is wrong
- **GREEN:** Each new test passes with minimal implementation
- **FULL:** `pytest tests/code_scan/test_scan_project.py -v` — all tests pass (existing + new), no warnings

### Incremental output equivalence test

A critical test must verify byte-for-byte equivalence (excluding timestamps) between full and incremental scan results:

```python
def test_incremental_output_equivalent_to_full(small_project_fixture):
    """Running --incremental on an unchanged project produces output 
    equivalent to a full run (excluding scanned_at timestamps)."""
    # 1. Full scan
    full_output = run_scan_project(target_dir, args=[])
    
    # 2. First incremental (no fingerprints) → same as full
    incr_output = run_scan_project(target_dir, args=['--incremental'])
    
    # 3. Compare excluding timestamps
    assert full_output['total_files'] == incr_output['total_files']
    assert full_output['total_lines'] == incr_output['total_lines']
    assert set(f['relative_path'] for f in full_output['files']) == \
           set(f['relative_path'] for f in incr_output['files'])
```

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
source venv/bin/activate

# Step 1: Unit tests (including existing Phase 1 regression)
python -m pytest tests/code_scan/test_scan_project.py -v

# Step 2: Incremental smoke test
python scripts/code-scan/scan_project.py /home/jarrad/work/testbeds/ua-flywheel/cass_memory_system --output /tmp/full-scan.json
python scripts/code-scan/scan_project.py /home/jarrad/work/testbeds/ua-flywheel/cass_memory_system --incremental --output /tmp/incr-scan.json
python -c "
import json
full = json.load(open('/tmp/full-scan.json'))
incr = json.load(open('/tmp/incr-scan.json'))
assert full['total_files'] == incr['total_files'], f'total_files mismatch: {full[\"total_files\"]} vs {incr[\"total_files\"]}'
assert full['total_lines'] == incr['total_lines']
assert len(full['files']) == len(incr['files'])
print(f'INCREMENTAL EQUIVALENCE PASS: {full[\"total_files\"]} files, {full[\"total_lines\"]} lines')
"

# Step 3: Structural change detection
touch test_file_to_modify.py  # or edit a file in the test fixture
python scripts/code-scan/scan_project.py /home/jarrad/work/testbeds/ua-flywheel/cass_memory_system --incremental --output /tmp/incr-changed.json
python -c "
import json
incr = json.load(open('/tmp/incr-changed.json'))
assert 'incremental_scan' in str(incr.get('warnings', [])), 'Missing incremental warning in output'
print('STRUCTURAL CHANGE SCAN PASS')
"

# Step 4: --full override
rm -rf /home/jarrad/work/testbeds/ua-flywheel/cass_memory_system/.hermes/code-state/fingerprints.json 2>/dev/null
# Or test with existing fingerprints:
# scan_project.py --full --incremental should behave like --full

# Step 5: Phase 1 regression — existing tests still pass
python -m pytest tests/code_scan/test_scan_project.py -v -k 'not incremental'

# Step 6: Scope guardrail
git diff --name-only | grep -vE '^(scripts/code-scan/scan_project\.py|tests/code_scan/)' && echo 'SCOPE FAIL' || echo 'SCOPE PASS'

# Step 7: Forbidden file check
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py | grep -q . && echo 'FORBIDDEN FAIL' || echo 'FORBIDDEN PASS'

# Step 8: No new runtime dependencies in scan_project.py
python -c "
import ast
from pathlib import Path
stdlib = {'argparse','collections','dataclasses','datetime','enum','fnmatch','hashlib','json','os','pathlib','re','sys','typing','itertools','tempfile'}
tree = ast.parse(Path('scripts/code-scan/scan_project.py').read_text())
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(a.name.split('.')[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        imports.add(node.module.split('.')[0])
nonstdlib = sorted(i for i in imports if i not in stdlib)
if nonstdlib:
    raise SystemExit(f'Non-stdlib imports: {nonstdlib}')
print('DEPENDENCY PASS')
"
```

### Expected pass criteria

1. `pytest tests/code_scan/test_scan_project.py -v` — 100% pass (existing + new tests)
2. Incremental output has same `total_files`, `total_lines`, and file list as full scan (unchanged project)
3. Structural change detection works; incremental warning in output
4. Phase 1 regression tests pass
5. Scope guardrail pass — only `scan_project.py` and its tests modified
6. Forbidden files unchanged
7. No new non-stdlib imports

## Approval Evidence

### Before commit — present this evidence bundle to JC

**1. Test output (verbatim):**
```
$ python -m pytest tests/code_scan/test_scan_project.py -v
```
Include the full output showing PASS/FAIL for every test.

**2. RED/GREEN/FULL evidence per behavior:**
- [ ] `--incremental` without fingerprints: RED → GREEN → FULL
- [ ] `--incremental` with no changes: RED → GREEN → FULL
- [ ] `--incremental` with structural change: RED → GREEN → FULL
- [ ] `--full` override: RED → GREEN → FULL
- [ ] Classification correctness: RED → GREEN → FULL
- [ ] Summary warnings: RED → GREEN → FULL
- [ ] Phase 1 regression: RED → GREEN → FULL

**3. Incremental output equivalence:**
```bash
python scripts/code-scan/scan_project.py <test_repo> --output /tmp/full.json
python scripts/code-scan/scan_project.py <test_repo> --incremental --output /tmp/incr.json
# Compare total_files, total_lines, files list (excluding scanned_at)
```

**4. Diff artifact:**
```bash
git diff scripts/code-scan/scan_project.py
git diff tests/code_scan/test_scan_project.py
```

**5. Scope guardrail:**
```bash
git diff --name-only | grep -vE '^(scripts/code-scan/scan_project\.py|tests/code_scan/)' && echo 'SCOPE FAIL' || echo 'SCOPE PASS'
```

**6. Workspace cleanliness freshly emitted:**
```
`git diff --name-only` shows only D2 allowed files
```

**7. Reviewer verdict:**
- [ ] Spec compliance (flags work, merge is correct, schema identical)
- [ ] Phase 1 behavior freshly emitted (--incremental absent → full scan as before)
- [ ] Scope preservation (only scan_project.py + tests modified)
- [ ] No new runtime dependencies (stdlib only)
- [ ] Context budget (zero — script only)
- [ ] Quality and security (no secrets, no path leaks)
- [ ] Workspace remains clean outside allowed files

**8. Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
```

---

> **Bead execution readiness = this bead passes reviewer polish and JC approves execution.**
> **Bead completion = all verification commands exit 0 + reviewer PASS + JC commit approval.**
> Coder subagent has NO commit/push authority.
