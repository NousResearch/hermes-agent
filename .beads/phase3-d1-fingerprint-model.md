---
id: phase3-d1-fingerprint-model
title: Phase 3 D1 — fingerprints.py: fingerprint extraction, comparison, and persistence module
status: completed
executor: delegate-coder
parallel_safe: false
base_branch: docs/ua-flywheel-phase3-plan
allowed_files:
  - scripts/code-scan/fingerprints.py
  - tests/code_scan/test_fingerprints.py
  - tests/code_scan/fixtures/fingerprints/
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - scripts/code-scan/scan_project.py
  - scripts/code-scan/extract_imports.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - .hermesignore
  - skills/
depends_on:
  - phase1-code-scan-completion-fix
  - phase2-d1-extract-imports
verification:
  - python -m pytest tests/code_scan/test_fingerprints.py -v
  - python -c "import sys; sys.path.insert(0, 'scripts/code-scan'); from fingerprints import extract_fingerprint, compare_fingerprints, load_fingerprint_file, save_fingerprint_file; print('IMPORT PASS')"
risk: low
---

# Phase 3 D1 — fingerprints.py: Fingerprint Extraction, Comparison, and Persistence

## Context & Intent

**Why this bead exists.** Phase 1 (`scan_project.py`) emits a per-file JSON inventory. Phase 2 (`extract_imports.py`) adds import maps. Phase 3 needs a persistence layer that stores per-file fingerprints (content hash, structural metadata) and a comparison engine that classifies each file as UNCHANGED / COSMETIC / STRUCTURAL. This enables incremental re-scanning (D2) by letting the scanner skip files that haven't changed structurally.

**Authority docs.** `.plans/phase-3-incremental-analysis.md` (D1 section + fingerprint format contract) defines the high-level scope. `.plans/ua-incorporation-strategy.md` (§3. Fingerprint-Based Incremental Re-Analysis) specifies that fingerprints are stored in `.hermes/code-state/fingerprints.json` and comparison runs as code, not context.

**Intent.** Create `scripts/code-scan/fingerprints.py` as a stdlib-only module that:
1. Extracts per-file fingerprints from source files: SHA-256 content hash, line count, file size, function names, class names, import module names.
2. Loads and saves fingerprint files to `.hermes/code-state/fingerprints.json`.
3. Compares a freshly-extracted fingerprint set against a persisted one, classifying each file as UNCHANGED, COSMETIC, or STRUCTURAL.
4. Handles new/deleted files correctly.

**Non-goals.** No tree-sitter/WASM. No AST parsing. No new runtime dependencies. No dashboard, React UI, auto-injection, SQLite, CLI command. No changes to existing Phase 1/2 files — fingerprints.py is entirely new.

## Implementation Details

### Target files

| File | Purpose | Max LOC |
|---|---|---|
| `scripts/code-scan/fingerprints.py` | Module: fingerprint extraction, comparison, persistence | ~200 |
| `tests/code_scan/test_fingerprints.py` | Unit + fixture-driven tests for all functions | ~200 |
| `tests/code_scan/fixtures/fingerprints/` | Fixture files: original + modified versions for change classification | varies |

### Required functions (exact naming)

| Function | Signature | Purpose |
|---|---|---|
| `extract_fingerprint` | `(file_path: str, scan_root: str, line_count: int, size_bytes: int, imports: Optional[list[str]]) -> dict` | Extract fingerprint for a single file. Returns: `{ "content_hash": "sha256:...", "line_count": int, "size_bytes": int, "functions": [...], "classes": [...], "imports": [...] }`. `content_hash` is computed via `hashlib.sha256` on raw file bytes. `functions` extracted via regex `^\s*(?:async\s+)?def\s+(\w+)\s*\(`. `classes` extracted via regex `^\s*class\s+(\w+)`. If `imports` provided, uses them; otherwise extracts via regex from `import`/`from` lines. |
| `_extract_functions` | `(source: str) -> list[str]` | Extract function names. Regex: `^\s*(?:async\s+)?def\s+(\w+)\s*\(` with `re.MULTILINE`. Return sorted unique list. |
| `_extract_classes` | `(source: str) -> list[str]` | Extract class names. Regex: `^\s*class\s+(\w+)` with `re.MULTILINE`. Return sorted unique list. |
| `_extract_content_hash` | `(file_path: str) -> str` | Compute SHA-256 hex digest of file content. Return `"sha256:<hex>"`. |
| `load_fingerprint_file` | `(fingerprints_path: str) -> Optional[dict]` | Load `.hermes/code-state/fingerprints.json`. Returns parsed dict or None if file missing/error. Validates `schema_version == "1.0.0"` and required top-level keys; returns None on validation failure. |
| `save_fingerprint_file` | `(fingerprints_path: str, project_root: str, files: dict[str, dict]) -> str` | Write fingerprint file. Creates parent directory if needed. Returns the path written. Schema: `{"schema_version": "1.0.0", "project_root": <abs_path>, "captured_at": <ISO 8601>, "files": <files>}`. |
| `compare_fingerprints` | `(old_fp: dict, new_fp: dict) -> dict[str, str]` | Compare old and new fingerprints. Returns `{relative_path: "UNCHANGED" | "COSMETIC" | "STRUCTURAL"}`. Rules: UNCHANGED = exact `content_hash` match; COSMETIC = hash differs but `functions`, `classes`, `imports` lists are identical (sorted comparison); STRUCTURAL = structural lists differ OR file is new/deleted. |
| `build_fingerprint_map` | `(scan_data: dict, project_root: str, import_data: Optional[dict]) -> dict[str, dict]` | Build complete fingerprint map from scan data. Iterates `scan_data["files"]`, extracts fingerprints for each file, optionally enriches with imports from `import_data`. Returns `{relative_path: fingerprint_dict}`. |
| `get_fingerprint_path` | `(project_root: str) -> str` | Returns `.hermes/code-state/fingerprints.json` path within the project. Uses `os.path.join(project_root, ".hermes", "code-state", "fingerprints.json")`. |

### Fingerprint JSON schema (exact contract)

```json
{
  "schema_version": "1.0.0",
  "project_root": "/absolute/path/to/project",
  "captured_at": "2026-05-30T12:00:00Z",
  "files": {
    "src/main.py": {
      "content_hash": "sha256:abc123def456...",
      "line_count": 150,
      "size_bytes": 4200,
      "functions": ["main", "parse_config", "_helper"],
      "classes": ["ConfigParser"],
      "imports": ["os", "sys", "json"]
    }
  }
}
```

Required top-level keys: `schema_version` (string, always `"1.0.0"`), `project_root` (string, absolute path), `captured_at` (string, ISO 8601), `files` (object with string keys → fingerprint records).

Required per-file keys: `content_hash` (string), `line_count` (integer), `size_bytes` (integer), `functions` (list[str], sorted), `classes` (list[str], sorted), `imports` (list[str], sorted).

### Change-level classification (exact rules)

| Level | Criteria |
|---|---|
| `UNCHANGED` | `content_hash` matches exactly between old and new fingerprint for the same relative path |
| `COSMETIC` | `content_hash` differs, BUT sorted `functions` lists are equal, sorted `classes` lists are equal, AND sorted `imports` lists are equal |
| `STRUCTURAL` | Any of: file is new (present in new but not old), file is deleted (present in old but not new), sorted `functions` lists differ, sorted `classes` lists differ, sorted `imports` lists differ |

### Regex patterns for structural extraction

| Target | Regex | Flags |
|---|---|---|
| Python functions | `^\s*(?:async\s+)?def\s+(\w+)\s*\(` | `re.MULTILINE` |
| Python classes | `^\s*class\s+(\w+)` | `re.MULTILINE` |
| Nested Python functions/classes | Same as above — nested defs/classes inside classes are also captured (acceptable — more conservative = safer) | |
| JS/TS functions | `^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(` | `re.MULTILINE` |
| JS/TS classes | `^\s*(?:export\s+)?class\s+(\w+)` | `re.MULTILINE` |
| Import extraction | Reuse patterns from `extract_imports.py` (not an import — uses equivalent regex in this module to keep it self-contained) | |

### Import extraction within fingerprints.py

The fingerprint module must extract imports for classification purposes. It must NOT import `extract_imports.py` as a module dependency (cyclic import risk). Instead, implement a lightweight `_extract_imports_from_source(source: str, language: str)` function that:

- For Python: same regex as `extract_imports.py` Python patterns
- For JS/TS: same regex as `extract_imports.py` JS/TS patterns
- For other languages: empty list with a warning log to stderr
- Returns a sorted, deduplicated list of module names

This is intentional duplication — the fingerprint extractor is a subset of the full import extractor and must work independently.

## Complexity Tier

**T2** — Single module with extraction, comparison, and persistence functions. Regex-heavy for structural analysis, but no algorithmic complexity beyond set comparisons. One source file, one test file, fixture directory with known file pairs. Estimated 8-10 subagent iterations. Requires coder subagent + Hermes verification + reviewer signoff before presentation for commit approval.

## Execution Engine

**Executor:** `delegate-coder` — Hermes dispatches a coder subagent via `delegate_task` with full plan context.

**Process:**
1. Coder subagent creates fixture files first, then writes tests that fail (RED).
2. Implement `fingerprints.py` to pass tests (GREEN).
3. Hermes runs `pytest tests/code_scan/test_fingerprints.py -v` — all must pass (FULL).
4. Hermes runs end-to-end: extract fingerprints from a test-bed repo, save, modify one file, re-extract, compare.
5. Reviewer subagent validates: spec compliance, scope guardrails, no forbidden-file touches, schema contract adherence, change classification correctness.
6. Hermes presents evidence bundle to JC — this bead authorizes implementation only, not commit/push.

**Subagent reliability preflight:**
- Task shape: deterministic module + regex extraction + set comparison + JSON persistence
- Expected artifacts: 1 source file, 1 test file, fixture directory with ≥3 file pairs (unchanged/cosmetic/structural)
- `max_iterations`: 15 per subagent dispatch
- File-write: YES. Run-test: YES. Commit: NO (await JC approval).

## Required Inline Context

### Project context

- **Repo:** `/home/jarrad/.hermes/hermes-agent`
- **Current branch:** `docs/ua-flywheel-phase3-plan`
- **Parent beads:** `phase1-code-scan-completion-fix` (scan output schema), `phase2-d1-extract-imports` (import extraction patterns)
- **No new pip dependencies** — stdlib only (`os`, `pathlib`, `json`, `hashlib`, `re`, `datetime`, `sys`, `typing`, `typing.Optional`)
- **Fingerprints directory:** `.hermes/code-state/` — must be created automatically if missing when saving

### Workspace cleanliness requirement

Phase 3 execution starts from a clean worktree for each bead. The executor must capture `git status --short` before and after the bead. Any file outside `allowed_files` is a scope violation and must be reverted before review.

### Scope guardrails

This bead may ONLY create or modify files listed in `allowed_files`. Any attempt to modify Phase 1/2 scripts, skills, config, or forbidden files must be rejected during review. This is a new module — no existing files are modified.

## Dependencies

| Dependency | Type | Status |
|---|---|---|
| Phase 1 code-scan scripts | prerequisite | Completed, committed, verified (80 tests pass) |
| `scan_project.py` JSON output | runtime input for `build_fingerprint_map` | Consumed as dict |
| `extract_imports.py` regex patterns | reference | Re-implemented as lightweight subset within fingerprints.py (no import dependency) |
| Python 3.10+ stdlib | runtime | Assumed present |
| pytest | test runtime | Assumed present |

## Test Obligations

### TDD protocol

Strict RED → GREEN → REFACTOR for each public function. Every public function must have at least one failing-first test.

### Test file mapping

| Source | Test | Coverage target |
|---|---|---|
| `_extract_content_hash` | `test_fingerprints.py` | Same file → same hash; different content → different hash; empty file → valid hash |
| `_extract_functions` | `test_fingerprints.py` | Python: `def foo():`, `async def bar():`; skip non-def lines; nested defs captured; sorted unique return |
| `_extract_classes` | `test_fingerprints.py` | Python: `class Foo:`, nested classes; sorted unique return |
| `extract_fingerprint` | `test_fingerprints.py` | Full extraction on a real fixture file; all required keys present; types correct |
| `load_fingerprint_file` | `test_fingerprints.py` | Valid file loads; missing file returns None; invalid schema (missing `schema_version`) returns None; corrupt JSON returns None |
| `save_fingerprint_file` | `test_fingerprints.py` | Creates directory if missing; writes valid JSON; schema matches contract; `captured_at` is ISO 8601 |
| `compare_fingerprints` | `test_fingerprints.py` | Identical fingerprints → UNCHANGED for all; hash-only difference → COSMETIC; function added → STRUCTURAL; new file → STRUCTURAL; deleted file → STRUCTURAL; classes changed → STRUCTURAL |
| `build_fingerprint_map` | `test_fingerprints.py` | From scan JSON data → full fingerprint map; optional import data enrichment; all keys present |
| `get_fingerprint_path` | `test_fingerprints.py` | Returns correct path for any project root |
| `_extract_imports_from_source` | `test_fingerprints.py` | Python imports extracted; JS/TS imports extracted; unknown language returns empty list |

### Fixture requirements

`tests/code_scan/fixtures/fingerprints/` must contain:

| Fixture | Purpose |
|---|---|
| `original/main.py` | Baseline Python file with known functions, classes, imports |
| `unchanged/main.py` | Byte-identical copy of original — tests UNCHANGED classification |
| `cosmetic/main.py` | Adds whitespace, comments, docstrings; identical functions/classes/imports — tests COSMETIC classification |
| `structural_func/main.py` | Adds a new `def new_function():` — tests STRUCTURAL detection via functions |
| `structural_class/main.py` | Adds a new `class NewClass:` — tests STRUCTURAL detection via classes |
| `structural_import/main.py` | Adds an `import new_module` — tests STRUCTURAL detection via imports |
| `mixed/main.py` | Mix of functions, classes, imports — tests composite fingerprint correctness |

### RED/GREEN/FULL evidence required

- **RED:** Tests fail with `AttributeError` / `ModuleNotFoundError` / incorrect classification
- **GREEN:** Each test passes with minimal implementation
- **FULL:** `pytest tests/code_scan/test_fingerprints.py -v` — all pass, no warnings

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
source venv/bin/activate

# Step 1: Unit tests
python -m pytest tests/code_scan/test_fingerprints.py -v

# Step 2: Schema validation
python -c "
import json, sys
sys.path.insert(0, 'scripts/code-scan')
from fingerprints import save_fingerprint_file, load_fingerprint_file
import tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    project = os.path.join(tmpdir, 'testproj')
    fp_path = os.path.join(project, '.hermes', 'code-state', 'fingerprints.json')
    
    # Test save
    files = {'main.py': {'content_hash': 'sha256:abc', 'line_count': 10, 'size_bytes': 100, 'functions': [], 'classes': [], 'imports': []}}
    result = save_fingerprint_file(fp_path, project, files)
    assert os.path.isfile(result), f'Save did not create file: {result}'
    
    # Test load
    data = load_fingerprint_file(fp_path)
    assert data is not None
    assert data['schema_version'] == '1.0.0'
    assert data['project_root'] == project
    assert 'captured_at' in data
    assert 'main.py' in data['files']
    for key in ['content_hash', 'line_count', 'size_bytes', 'functions', 'classes', 'imports']:
        assert key in data['files']['main.py'], f'Missing key: {key}'
    print('SCHEMA PASS')
"

# Step 3: End-to-end change classification
python -c "
import json, sys
sys.path.insert(0, 'scripts/code-scan')
from fingerprints import extract_fingerprint, compare_fingerprints
import tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    # Create baseline file
    f1 = os.path.join(tmpdir, 'main.py')
    with open(f1, 'w') as f:
        f.write('import os\n\ndef main():\n    pass\n')
    
    # Extract fingerprint
    fp1 = extract_fingerprint(f1, tmpdir, line_count=4, size_bytes=os.path.getsize(f1), imports=None)
    
    # Create changed file (cosmetic only)
    f2 = os.path.join(tmpdir, 'main.py')
    with open(f2, 'w') as f:
        f.write('# Added comment\nimport os\n\ndef main():\n    pass\n')
    
    fp2 = extract_fingerprint(f2, tmpdir, line_count=5, size_bytes=os.path.getsize(f2), imports=None)
    
    # Compare
    result = compare_fingerprints(
        {'files': {'main.py': fp1}},
        {'files': {'main.py': fp2}}
    )
    assert result.get('main.py') == 'COSMETIC', f'Expected COSMETIC, got {result}'
    print('CLASSIFICATION PASS')
"

# Step 4: Scope guardrail — Phase 1/2 files unchanged
git diff -- scripts/code-scan/scan_project.py scripts/code-scan/extract_imports.py scripts/code-scan/language_registry.py scripts/code-scan/graph_schema.py .hermesignore | grep -q . && echo 'GUARDRAIL FAIL: Phase 1/2 files modified' && exit 1 || echo 'GUARDRAIL PASS'

# Step 5: Forbidden file check
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py | grep -q . && echo 'FORBIDDEN FAIL' || echo 'FORBIDDEN PASS'

# Step 6: No new runtime dependencies
python -c "
import ast
from pathlib import Path
stdlib = {'argparse','collections','dataclasses','datetime','enum','fnmatch','hashlib','json','os','pathlib','re','sys','typing','itertools','tempfile'}
tree = ast.parse(Path('scripts/code-scan/fingerprints.py').read_text())
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

1. `pytest tests/code_scan/test_fingerprints.py -v` — 100% pass
2. Schema validation: fingerprints.json has all required keys, correct types, valid ISO timestamp
3. Change classification: UNCHANGED on identical hashes, COSMETIC on hash-only diffs, STRUCTURAL on structural diffs
4. Phase 1/2 files unchanged in git diff
5. Forbidden files unchanged
6. No new non-stdlib imports

## Approval Evidence

### Before commit — present this evidence bundle to JC

**1. Test output (verbatim):**
```
$ python -m pytest tests/code_scan/test_fingerprints.py -v
```
Include the full output showing PASS/FAIL for every test.

**2. RED/GREEN/FULL evidence per function:**
- [ ] `_extract_content_hash`: RED → GREEN → FULL
- [ ] `_extract_functions`: RED → GREEN → FULL
- [ ] `_extract_classes`: RED → GREEN → FULL
- [ ] `extract_fingerprint`: RED → GREEN → FULL
- [ ] `load_fingerprint_file`: RED → GREEN → FULL
- [ ] `save_fingerprint_file`: RED → GREEN → FULL
- [ ] `compare_fingerprints`: RED → GREEN → FULL
- [ ] `build_fingerprint_map`: RED → GREEN → FULL
- [ ] `get_fingerprint_path`: RED → GREEN → FULL
- [ ] `_extract_imports_from_source`: RED → GREEN → FULL

**3. Change classification evidence:**
```bash
# Create three test scenarios and verify classification
# UNCHANGED: identical file → exact hash match
# COSMETIC: whitespace-only change → structural lists identical
# STRUCTURAL: new function → functions list differs
```

**4. Schema compliance:**
```python
# Validate fingerprints.json against schema contract
data = json.load(open(fingerprints_path))
assert data['schema_version'] == '1.0.0'
assert isinstance(data['project_root'], str)
assert isinstance(data['captured_at'], str)
assert isinstance(data['files'], dict)
for path, fp in data['files'].items():
    for key in ['content_hash', 'line_count', 'size_bytes', 'functions', 'classes', 'imports']:
        assert key in fp, f'Missing key {key} in {path}'
```

**5. Scope guardrail:**
```bash
git diff --name-only | grep -vE '^(scripts/code-scan/fingerprints\.py|tests/code_scan/)' && echo 'SCOPE FAIL' || echo 'SCOPE PASS'
```

**6. Workspace cleanliness preserved:**
```
cmp /tmp/d1-forbidden-pre.diff /tmp/d1-forbidden-post.diff → exit 0
```

**7. Reviewer verdict:**
- [ ] Spec compliance (all functions present, schema contract matched, classification rules correct)
- [ ] Scope preservation (no Phase 1/2/other files touched, no forbidden patterns)
- [ ] No new runtime dependencies (stdlib only)
- [ ] Context budget (zero — script only, no SKILL.md)
- [ ] Quality and security (no secrets, no path leaks, sorted/deduplicated lists)
- [ ] Workspace remains clean outside allowed files

**8. Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
```

---

> **Bead execution readiness = this bead passes reviewer polish and JC approves execution.**
> **Bead completion = all verification commands exit 0 + reviewer PASS + JC commit approval.**
> Coder subagent has NO commit/push authority.
