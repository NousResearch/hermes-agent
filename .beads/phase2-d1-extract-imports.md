---
id: phase2-d1-extract-imports
title: Phase 2 D1 — extract_imports.py: import/dependency map extractor
status: complete-and-committed
executor: delegate-coder
parallel_safe: false
base_branch: docs/ua-flywheel-phase1-phase2-plan
allowed_files:
  - scripts/code-scan/extract_imports.py
  - tests/code_scan/test_extract_imports.py
  - tests/code_scan/fixtures/imports/
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - scripts/code-scan/scan_project.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - .hermesignore
  - skills/
depends_on:
  - phase1-code-scan-completion-fix
verification:
  - python -m pytest tests/code_scan/test_extract_imports.py -v
  - python scripts/code-scan/extract_imports.py /tmp/scan-cass.json
risk: low
---

# Phase 2 D1 — extract_imports.py: Import/Dependency Map Extractor

## Context & Intent

**Why this bead exists.** Phase 1 (`phase1-code-scan-completion-fix`) produced `scan_project.py`, which emits a per-file JSON inventory (path, language, category, lines, size). Phase 2 needs an import/dependency map layered on top of that scan so downstream consumers (code-scan skill, validation-gate skill, future graph assembly) can understand module relationships without parsing every source file.

**Authority docs.** `.plans/phase-2-flywheel-ua-integration.md` (§D1: `scripts/code-scan/extract_imports.py`) defines the high-level scope. `.plans/ua-incorporation-strategy.md` (§1. Script-First / LLM-Second) mandates that import extraction is deterministic, regex-based, and uses zero runtime dependencies.

**Intent.** Create `scripts/code-scan/extract_imports.py` that reads `scan_project.py` JSON output, extracts import/dependency statements per file using language-specific regex patterns, and emits a structured import map JSON. No LLM involvement. Must support at least 5 languages: Python, JavaScript/TypeScript, Rust, Go, shell/bash.

**Non-goals.** No tree-sitter/WASM (Phase 4). No AST parsing. No new runtime dependencies. No dashboard, no React UI, no auto-injection, no SQLite store, no CLI command, no always-on scanning. No changes to existing Phase 1 files.

## Implementation Details

### Target files

| File | Purpose | Max LOC |
|---|---|---|
| `scripts/code-scan/extract_imports.py` | CLI script: reads scan JSON, extracts imports, outputs import map JSON | ~200 |
| `tests/code_scan/test_extract_imports.py` | Unit + fixture-driven tests for all language extractors | ~150 |
| `tests/code_scan/fixtures/imports/` | Language-specific fixture files with known imports | varies |

### Required functions (exact naming)

| Function | Purpose |
|---|---|
| `load_scan_output(path: str) -> dict` | Load and validate scan JSON from file path; return parsed dict. Raise `ValueError` on invalid schema. |
| `iter_scanned_files(scan_data: dict) -> Iterator[tuple[str, str]]` | Yield `(relative_path, language)` pairs from the `files` array of scan output. |
| `extract_python_imports(source: str) -> list[str]` | Extract `import X`, `from X import Y` patterns. Return list of top-level module names (e.g., `import os.path` → `os`). |
| `extract_js_ts_imports(source: str) -> list[str]` | Extract `import X from 'Y'`, `require('Y')`, dynamic `import('Y')`. Return resolved module names. |
| `extract_rust_imports(source: str) -> list[str]` | Extract `use X::Y`, `extern crate X`. Return top-level crate names. |
| `extract_go_imports(source: str) -> list[str]` | Extract `import "pkg"` and `import ( "pkg" )` blocks. Return package paths. |
| `extract_shell_imports(source: str) -> list[str]` | Extract `source <file>`, `. <file>`, `bash <file>`, `zsh <file>`. Return sourced file paths. |
| `extract_imports_for_file(path: str, language: str, scan_root: str) -> tuple[list[str], list[str]]` | Dispatch to language-specific extractor. Returns `(imports, warnings)`. |
| `build_import_map(scan_data: dict, scan_root: str) -> dict` | Orchestrate: iterate files, extract imports, assemble final output dict following the output schema. |
| `main() -> int` | CLI entry point. Parse args: `extract_imports.py <scan_output.json>`. Write JSON to stdout. Return exit code. |

### Output schema (exact contract)

The JSON emitted by `extract_imports.py` must be parseable by downstream consumers (code-scan skill, validation-gate skill, future graph assembly). Required structure:

```json
{
  "schema_version": "1.0.0",
  "source_scan": {
    "project_root": "/path/to/project",
    "total_files": 423
  },
  "generated_at": "2026-05-30T12:00:00Z",
  "files": {
    "src/main.py": {
      "imports": ["os", "sys", "pathlib"],
      "warnings": []
    },
    "src/index.ts": {
      "imports": ["react", "./components/Header", "./utils/helpers"],
      "warnings": ["dynamic import detected"]
    }
  },
  "totals": {
    "files_with_imports": 156,
    "files_without_imports": 267,
    "unique_modules": 89,
    "total_warnings": 3
  }
}
```

Required top-level keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | yes | Always `"1.0.0"` for Phase 2 |
| `source_scan` | object | yes | `project_root` (string) and `total_files` (integer) from scan |
| `generated_at` | string | yes | ISO 8601 timestamp |
| `files` | object | yes | Map of relative_path → `{ imports: string[], warnings: string[] }` |
| `totals` | object | yes | Aggregate counts (see schema above) |

### Regex patterns (per language)

- **Python:** `^\s*(?:import\s+(\w+)|from\s+(\w+))` — capture group 1 or 2, use first segment before `.`.
- **JS/TS:** `(?:import\s+.*?\s+from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]|import\s*\(\s*['"]([^'"]+)['"])` — capture all groups.
- **Rust:** `^\s*use\s+((?:\w+)(?:::\w+)*)` — capture first segment before `::`; also `^\s*extern\s+crate\s+(\w+)`.
- **Go:** `(?:import\s+"([^"]+)"|^\s+"([^"]+)")` — also handle parenthesized blocks.
- **Shell:** `(?:^(?:source|\.)\s+([^\s#]+)|^(?:bash|zsh|sh)\s+([^\s#]+))` — extract sourced/script paths.

All regexes operate on file content read as UTF-8 with replacement for decoding errors. Unsupported languages emit a warning and an empty imports list.

## Complexity Tier

**T2** — Single-file implementation with test fixtures. Regex-heavy but no algorithmic complexity. One script + one test file + fixture files. Estimated 6–8 subagent iterations. Requires coder subagent + Hermes verification + reviewer signoff before presentation for commit approval.

## Execution Engine

**Executor:** `delegate-coder` — Hermes dispatches a coder subagent via `delegate_task` with full plan context.

**Process:**
1. Coder subagent creates fixture files first, then writes tests that fail (RED).
2. Implement `extract_imports.py` to pass tests (GREEN).
3. Hermes runs `pytest tests/code_scan/test_extract_imports.py -v` — all must pass (FULL).
4. Hermes runs end-to-end: `python scripts/code-scan/extract_imports.py <scan_output.json>` against real Phase 1 scan output from a test-bed repo.
5. Reviewer subagent validates: spec compliance, scope guardrails, no forbidden-file touches, schema contract adherence.
6. Hermes presents evidence bundle to JC — this bead authorizes implementation only, not commit/push.

**Subagent reliability preflight:**
- Task shape: deterministic script + regex extraction + fixture-driven testing
- Expected artifacts: 1 source file, 1 test file, fixture directory with ≥5 language samples
- `max_iterations`: 15 per subagent dispatch
- File-write: YES. Run-test: YES. Commit: NO (await JC approval).

## Required Inline Context

### Project context

- **Repo:** `/home/jarrad/.hermes/hermes-agent`
- **Current branch:** `docs/ua-flywheel-phase1-phase2-plan`
- **Parent bead:** `phase1-code-scan-completion-fix` — scan output schema is defined there and must be consumed faithfully.
- **No new pip dependencies** — stdlib only (`os`, `pathlib`, `json`, `argparse`, `re`, `datetime`, `sys`, `typing`).

### Existing dirty files — DO NOT TOUCH

```
tools/skills_sync.py                 # dirty — unrelated
tests/tools/test_skills_sync.py      # dirty — unrelated
```

### Scope guardrails

This bead may ONLY create or modify files listed in `allowed_files`. Any attempt to modify Phase 1 scripts, skills, config, or forbidden files must be rejected during review.

## Dependencies

| Dependency | Type | Status |
|---|---|---|
| Phase 1 code-scan scripts | prerequisite | Completed, committed, verified (80 tests pass) |
| `scan_project.py` JSON output | runtime input | Consumed as file path argument |
| Python 3.10+ stdlib | runtime | Assumed present |
| pytest | test runtime | Assumed present |

## Test Obligations

### TDD protocol

Strict RED → GREEN → REFACTOR for each language extractor and the orchestration layer. Every public function must have at least one failing-first test.

### Test file mapping

| Source | Test | Coverage target |
|---|---|---|
| `load_scan_output` | `test_extract_imports.py` | Valid JSON load, missing file, invalid schema (missing `files` key), empty scan |
| `extract_python_imports` | `test_extract_imports.py` | `import os`, `from pathlib import Path`, `import os.path` → `["os", "pathlib", "os"]`, multi-line, comments ignored |
| `extract_js_ts_imports` | `test_extract_imports.py` | Named imports, default imports, `require()`, dynamic `import()`, relative paths |
| `extract_rust_imports` | `test_extract_imports.py` | `use std::io`, `use serde::Deserialize`, `extern crate tokio` |
| `extract_go_imports` | `test_extract_imports.py` | Single import, grouped import, relative imports |
| `extract_shell_imports` | `test_extract_imports.py` | `source env.sh`, `. ~/.bashrc`, `bash script.sh` |
| `build_import_map` | `test_extract_imports.py` | Full pipeline from scan JSON → import map JSON, schema compliance, totals correct |
| `main()` | `test_extract_imports.py` | CLI arg parsing, stdout output, non-zero exit on invalid input |

### RED/GREEN/FULL evidence required

For each language extractor and orchestration function:

- **RED:** Tests fail with `AttributeError` / `ModuleNotFoundError` / empty returns
- **GREEN:** Each test passes with minimal implementation
- **FULL:** `pytest tests/code_scan/test_extract_imports.py -v` — all pass, no warnings

### Fixture verification

Each fixture file in `tests/code_scan/fixtures/imports/` must produce a deterministic, assertion-matched imports list:
- `python_sample.py`: expect `["os", "sys", "json", "pathlib"]`
- `ts_sample.ts`: expect `["react", "react-dom", "./App", "./utils"]`
- `rust_sample.rs`: expect `["std", "serde", "tokio"]`
- `go_sample.go`: expect `["fmt", "net/http", "github.com/gin-gonic/gin"]`
- `shell_sample.sh`: expect `["env.sh", "~/.bashrc"]`

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
source venv/bin/activate

# Step 1: Unit tests
python -m pytest tests/code_scan/test_extract_imports.py -v

# Step 2: End-to-end against real scan output
python scripts/code-scan/scan_project.py /home/jarrad/.hermes/hermes-agent/cass_memory_system --output /tmp/scan-e2e.json
python scripts/code-scan/extract_imports.py /tmp/scan-e2e.json > /tmp/imports-e2e.json
python -c "
import json
d = json.load(open('/tmp/imports-e2e.json'))
assert d['schema_version'] == '1.0.0'
assert 'files' in d
assert 'totals' in d
assert isinstance(d['files'], dict)
print(f'E2E PASS: {len(d[\"files\"])} files indexed, {d[\"totals\"][\"files_with_imports\"]} with imports')
"

# Step 3: Scope guardrail — Phase 1 files unchanged
git diff -- scripts/code-scan/scan_project.py scripts/code-scan/language_registry.py scripts/code-scan/graph_schema.py .hermesignore | grep -q . && echo 'GUARDRAIL FAIL: Phase 1 files modified' && exit 1 || echo 'GUARDRAIL PASS'

# Step 4: Forbidden file check
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py 2>/dev/null | cmp - /tmp/d1-forbidden-pre.diff && echo 'FORBIDDEN PASS' || echo 'FORBIDDEN FAIL'

# Step 5: No new runtime dependencies
python -c "
import ast
from pathlib import Path
stdlib = {'argparse','collections','dataclasses','datetime','enum','fnmatch','hashlib','json','os','pathlib','re','sys','typing','itertools'}
tree = ast.parse(Path('scripts/code-scan/extract_imports.py').read_text())
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

1. `pytest tests/code_scan/test_extract_imports.py -v` — 100% pass
2. E2E: import map JSON has correct schema, `files_with_imports > 0`
3. Phase 1 files unchanged in git diff
4. Forbidden files unchanged
5. No new non-stdlib imports

## Approval Evidence

### Before commit — present this evidence bundle to JC

**1. Test output (verbatim):**
```
$ python -m pytest tests/code_scan/test_extract_imports.py -v
```
Include the full output showing PASS/FAIL for every test.

**2. RED/GREEN/FULL evidence per extractor:**
- [ ] `extract_python_imports`: RED → GREEN → FULL
- [ ] `extract_js_ts_imports`: RED → GREEN → FULL
- [ ] `extract_rust_imports`: RED → GREEN → FULL
- [ ] `extract_go_imports`: RED → GREEN → FULL
- [ ] `extract_shell_imports`: RED → GREEN → FULL
- [ ] `build_import_map`: RED → GREEN → FULL
- [ ] `load_scan_output`: RED → GREEN → FULL
- [ ] `main()`: RED → GREEN → FULL

**3. Schema compliance:**
```bash
python -c "
import json
d = json.load(open('/tmp/imports-e2e.json'))
for key in ['schema_version','source_scan','generated_at','files','totals']:
    assert key in d, f'Missing key: {key}'
assert d['schema_version'] == '1.0.0'
for path, data in d['files'].items():
    assert 'imports' in data
    assert 'warnings' in data
    assert isinstance(data['imports'], list)
for key in ['files_with_imports','files_without_imports','unique_modules','total_warnings']:
    assert key in d['totals']
print('SCHEMA PASS')
"
```

**4. Diff artifact:**
```bash
git diff --no-index -- /dev/null scripts/code-scan/extract_imports.py
git diff --no-index -- /dev/null tests/code_scan/test_extract_imports.py
git ls-files --others --exclude-standard tests/code_scan/fixtures/imports/
```

**5. Scope guardrail:**
```bash
git diff --name-only | grep -vE '^(scripts/code-scan/extract_imports\.py|tests/code_scan/)' && echo 'SCOPE FAIL' || echo 'SCOPE PASS'
```

**6. Existing dirty files preserved:**
```
cmp /tmp/d1-forbidden-pre.diff /tmp/d1-forbidden-post.diff → exit 0
```

**7. Reviewer verdict:**
- [ ] Spec compliance (all functions present, schema contract matched, ≥5 languages supported)
- [ ] Scope preservation (Phase 1 files untouched, no forbidden patterns)
- [ ] No new runtime dependencies (stdlib only)
- [ ] Context budget (no SKILL.md files — script only)
- [ ] Quality and security (no secrets, no path leaks)
- [ ] Existing dirty files not modified

**8. Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
```

---

> **Bead execution readiness = this bead passes reviewer polish and JC approves execution.**
> **Bead completion = all verification commands exit 0 + reviewer PASS + JC commit approval.**
> Coder subagent has NO commit/push authority.
