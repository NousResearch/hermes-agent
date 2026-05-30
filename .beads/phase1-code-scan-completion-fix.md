---
id: phase1-code-scan-completion-fix
title: Phase 1 Foundation — deterministic code-scan scripts + .hermesignore
status: ready-for-review
executor: delegate-coder
parallel_safe: false
base_branch: docs/ua-flywheel-phase1-phase2-plan
allowed_files:
  - scripts/code-scan/scan_project.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - .hermesignore
  - tests/code_scan/__init__.py
  - tests/code_scan/test_scan_project.py
  - tests/code_scan/test_language_registry.py
  - tests/code_scan/test_graph_schema.py
  - tests/code_scan/fixtures/
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
touches:
  - scripts/code-scan/scan_project.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - .hermesignore
  - tests/code_scan/__init__.py
  - tests/code_scan/test_scan_project.py
  - tests/code_scan/test_language_registry.py
  - tests/code_scan/test_graph_schema.py
  - tests/code_scan/fixtures/
depends_on: []
verification:
  - python -m pytest tests/code_scan/ -v
  - python scripts/code-scan/scan_project.py /home/jarrad/.hermes/hermes-agent/cass_memory_system
  - python scripts/code-scan/scan_project.py /home/jarrad/.hermes/hermes-agent/mission-control
  - python scripts/code-scan/scan_project.py /home/jarrad/.hermes/hermes-agent
risk: low
---

# Phase 1 Code-Scan Foundation — Completion/Fix Bead

## Context & Intent

**Why this bead exists.** Phase 1 of the UA→Flywheel integration is the foundation tier: deterministic file enumeration, language/category classification, lightweight graph-schema validation, and project-level ignore rules. Phase 2 (orchestration layer with JIT skills) is blocked because `scripts/code-scan/scan_project.py` does not yet exist. Running `python scripts/code-scan/scan_project.py ...` raises `No such file or directory`.

**Authority docs.** `.plans/ua-incorporation-strategy.md` (§Phase 1) and `understand-anything-to-flywheel-review.md` define the scope. `.plans/phase-2-flywheel-ua-integration.md` (§Prerequisites) explicitly requires all four Phase 1 deliverables before Phase 2 can begin. `.plans/project-state-ua-flywheel.md` lists every deliverable as ⬜ Not started.

**Intent.** Create the four Phase 1 artifacts so that Phase 2's prerequisite gate passes. This bead does **not** build the orchestration layer, CLI command, dashboard, or any always-on scanning. It produces a Python-stdlib-only scanner module with tests, validated against three local test-bed repos.

**Non-goals.** No `flywheel scan` CLI command. No React/Vite/web endpoint. No automatic prompt or context injection. No always-on background scanning. No tree-sitter or WASM binaries. No SQLite summary store. No new runtime dependencies without JC approval. No changes to `tools/skills_sync.py` or `tests/tools/test_skills_sync.py` (existing unrelated dirty files — leave them untouched).

## Implementation Details

### Target files

Create these directories first if absent: `scripts/code-scan/`, `tests/code_scan/`, and `tests/code_scan/fixtures/`. Do **not** create or modify `.hermes/PROJECT_STATE.md` from the coder subagent; Hermes owns project-state updates outside this implementation bead.

| File | Purpose | Max LOC |
|---|---|---|
| `scripts/code-scan/language_registry.py` | Static extension→language, category, and framework-pattern lookup tables | ~120 |
| `scripts/code-scan/scan_project.py` | CLI script: walks project tree, applies `.hermesignore`, classifies files, counts lines, outputs JSON | ~180 |
| `scripts/code-scan/graph_schema.py` | Node/edge type enums, alias map, in-process validation function | ~100 |
| `.hermesignore` | Default exclusion rules (one pattern per line, `#` comments, blank lines ignored) | ~30 |
| `tests/code_scan/test_language_registry.py` | Unit tests for registry lookups, priority rules, unknown ext handling | ~80 |
| `tests/code_scan/test_scan_project.py` | Unit + integration tests for walker, filter, output schema | ~150 |
| `tests/code_scan/test_graph_schema.py` | Unit tests for schema validation, alias resolution, orphan detection | ~80 |
| `tests/code_scan/__init__.py` | Empty package marker if needed for pytest discovery/import stability | 0 |
| `tests/code_scan/fixtures/` | Small fixture trees for deterministic test assertions | varies |

### language_registry.py

A single Python module exposing three public constants and helper functions:

```python
# ── Extension → Language lookup ──────────────────────────────
LANGUAGE_BY_EXT: dict[str, str] = {
    # Core languages
    '.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.rs': 'rust', '.go': 'go', '.rb': 'ruby',
    '.java': 'java', '.kt': 'kotlin', '.scala': 'scala',
    '.cs': 'csharp', '.fs': 'fsharp',
    '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.hpp': 'cpp',
    '.swift': 'swift', '.m': 'objective-c', '.mm': 'objective-cpp',
    '.dart': 'dart', '.lua': 'lua', '.r': 'r', '.R': 'r',
    '.ex': 'elixir', '.exs': 'elixir', '.erl': 'erlang', '.hrl': 'erlang',
    '.hs': 'haskell', '.lhs': 'haskell',
    '.php': 'php', '.vue': 'vue', '.svelte': 'svelte',
    '.sql': 'sql', '.sh': 'shell', '.bash': 'shell', '.zsh': 'shell',
    '.ps1': 'powershell', '.bat': 'batch', '.cmd': 'batch',
    '.tf': 'hcl', '.hcl': 'hcl', '.yaml': 'yaml', '.yml': 'yaml',
    '.json': 'json', '.toml': 'toml', '.xml': 'xml',
    '.md': 'markdown', '.rst': 'rst', '.txt': 'text',
    '.css': 'css', '.scss': 'scss', '.sass': 'sass', '.less': 'less',
    '.html': 'html', '.htm': 'html', '.svg': 'svg',
    '.proto': 'protobuf', '.graphql': 'graphql', '.gql': 'graphql',
    '.dockerfile': 'dockerfile', '.nix': 'nix',
    '.cu': 'cuda',
    # Add more as needed during Phase 1 — these cover 95% of common repos
}

# ── Extension → Category ─────────────────────────────────────
# Categories: code, test, config, doc, data, infra, template, other
CATEGORY_BY_EXT: dict[str, str] = {
    '.py': 'code', '.js': 'code', '.ts': 'code', '.tsx': 'code',
    '.rs': 'code', '.go': 'code', '.rb': 'code', '.java': 'code',
    '.kt': 'code', '.c': 'code', '.cpp': 'code', '.swift': 'code',
    '.dart': 'code', '.lua': 'code', '.r': 'code', '.ex': 'code',
    '.hs': 'code', '.php': 'code', '.vue': 'code', '.svelte': 'code',
    '.sql': 'code', '.cu': 'code',
    '.css': 'code', '.scss': 'code', '.sass': 'code', '.less': 'code',
    '.html': 'template', '.htm': 'template', '.svg': 'template',
    '.yaml': 'config', '.yml': 'config', '.json': 'config',
    '.toml': 'config', '.xml': 'config', '.ini': 'config',
    '.cfg': 'config', '.env': 'config',
    '.md': 'doc', '.rst': 'doc', '.txt': 'doc',
    '.proto': 'code', '.graphql': 'code', '.gql': 'code',
    '.tf': 'infra', '.hcl': 'infra',
    '.dockerfile': 'infra', '.nix': 'infra',
    '.sh': 'infra', '.bash': 'infra', '.zsh': 'infra',
    '.ps1': 'infra', '.bat': 'infra', '.cmd': 'infra',
    # Note: test-category is determined by path convention, not extension
}

# ── Special filename → Category (extension-agnostic) ─────────
INFRA_FILENAMES: set[str] = {
    'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
    'Makefile', 'CMakeLists.txt', 'Vagrantfile',
    'Jenkinsfile', '.gitlab-ci.yml', '.github',
    'nginx.conf', '.env', '.env.example',
    'tsconfig.json', 'webpack.config.js', 'vite.config.ts',
    'pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt',
    'Cargo.toml', 'go.mod', 'go.sum', 'package.json', 'bun.lock',
    'pnpm-lock.yaml', 'yarn.lock', 'package-lock.json',
    'poetry.lock', 'Pipfile', 'Pipfile.lock',
}

# ── Framework detection patterns ─────────────────────────────
# Each entry: (manifest_path, key_or_pattern) → framework_name
FRAMEWORK_MANIFEST_PATTERNS: list[tuple[str, str, str]] = [
    ('package.json', 'react', 'react'),
    ('package.json', 'next', 'nextjs'),
    ('package.json', 'vue', 'vue'),
    ('package.json', 'svelte', 'svelte'),
    ('package.json', '@angular', 'angular'),
    ('package.json', 'express', 'express'),
    ('package.json', 'fastify', 'fastify'),
    ('package.json', 'nest', 'nestjs'),
    ('pyproject.toml', 'django', 'django'),
    ('pyproject.toml', 'fastapi', 'fastapi'),
    ('pyproject.toml', 'flask', 'flask'),
    ('Cargo.toml', 'actix', 'actix-web'),
    ('Cargo.toml', 'tokio', 'tokio'),
    ('go.mod', 'gin-gonic/gin', 'gin'),
    ('go.mod', 'labstack/echo', 'echo'),
]


def get_language(filepath: str) -> str:
    """Return language string for a given file path.

    Checks INFRA_FILENAMES first for special handling, then falls back
    to LANGUAGE_BY_EXT lookup, then returns 'unknown'.
    """
    ...

def get_category(filepath: str) -> str:
    """Return category string for a given file path.

    Priority: INFRA_FILENAMES → CATEGORY_BY_EXT → path contains 'test' → 'other'.
    """
    ...

def detect_frameworks(project_root: str) -> list[str]:
    """Scan project root for framework-indicating manifest files."""
    ...
```

### scan_project.py

Command-line script. Invoked as:
```bash
python scripts/code-scan/scan_project.py <target_dir> [--output file.json] [--verbose]
```

Behavior:
1. Read `.hermesignore` from the project root (or use default if absent). Parse as gitignore-style patterns (one per line, `#` comments, `!` negation not required for Phase 1).
2. Walk the directory tree, excluding `.git/`, `node_modules/`, `__pycache__/`, `.venv/`, and any patterns from `.hermesignore`.
3. For each file: resolve language via `language_registry.get_language()`, category via `language_registry.get_category()`, count lines (physical lines, excluding blanks and comment-only lines for complexity estimation).
4. Build JSON output:
```json
{
  "project_root": "/path/to/project",
  "scanned_at": "2026-05-30T12:00:00Z",
  "total_files": 423,
  "total_lines": 15234,
  "languages": { "python": 280, "typescript": 143 },
  "categories": { "code": 350, "config": 45, "doc": 28 },
  "frameworks": ["react", "nextjs"],
  "files": [
    {
      "path": "src/main.py",
      "relative_path": "src/main.py",
      "language": "python",
      "category": "code",
      "lines": 45,
      "size_bytes": 1234
    }
  ],
  "warnings": []
}
```
5. If `--output` is given, write JSON to that path; otherwise print to stdout.
6. Exit 0 on success, non-zero on errors (with message to stderr).

Do **not** add a `--dry-run` mode for Phase 1. The approval gate depends on real JSON output from the exact scanner command JC requested.

### graph_schema.py

Lightweight schema module — no external dependencies. Provides:

```python
from enum import Enum

class NodeType(str, Enum):
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    # Reserved for Phase 3+: INTERFACE = "interface", CONFIG = "config", etc.

class EdgeType(str, Enum):
    IMPORTS = "imports"
    CONTAINS = "contains"
    CALLS = "calls"
    TESTED_BY = "tested_by"
    CONFIGURES = "configures"
    DOCUMENTS = "documents"

# Alias map for normalizing LLM or external input
NODE_TYPE_ALIASES: dict[str, NodeType] = {
    "func": NodeType.FUNCTION, "fn": NodeType.FUNCTION,
    "method": NodeType.FUNCTION,
    "file": NodeType.FILE, "src": NodeType.FILE,
    "class": NodeType.CLASS, "type": NodeType.CLASS,
    "module": NodeType.MODULE, "pkg": NodeType.MODULE, "package": NodeType.MODULE,
}

EDGE_TYPE_ALIASES: dict[str, EdgeType] = {
    "import": EdgeType.IMPORTS, "imports_from": EdgeType.IMPORTS,
    "contains": EdgeType.CONTAINS, "has": EdgeType.CONTAINS,
    "calls": EdgeType.CALLS, "invoke": EdgeType.CALLS, "invokes": EdgeType.CALLS,
    "tested_by": EdgeType.TESTED_BY, "tested_by_file": EdgeType.TESTED_BY,
    "configures": EdgeType.CONFIGURES, "configured_by": EdgeType.CONFIGURES,
    "documents": EdgeType.DOCUMENTS, "doc": EdgeType.DOCUMENTS,
}


def validate_node(node: dict) -> list[str]:
    """Validate a node dict. Returns list of issue strings (empty = valid)."""
    ...

def validate_edge(edge: dict, known_node_ids: set[str]) -> list[str]:
    """Validate an edge dict against known node IDs. Returns issue list."""
    ...

def validate_graph(graph: dict) -> dict[str, list[str]]:
    """Validate an entire graph dict. Returns {"issues": [...], "warnings": [...]}."""
    ...
```

### .hermesignore

Default exclusion rules. One pattern per line, `#` for comments, blank lines ignored. Patterns match against relative paths.

```
# ── Version control ──────────────────────────────────────────
.git/
.svn/
.hg/

# ── Build artifacts ──────────────────────────────────────────
node_modules/
__pycache__/
*.pyc
*.pyo
dist/
build/
out/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# ── Virtual environments ─────────────────────────────────────
.venv/
venv/
env/
.env/

# ── IDE / editor ─────────────────────────────────────────────
.vscode/
.idea/
*.swp
*.swo

# ── OS ───────────────────────────────────────────────────────
.DS_Store
Thumbs.db

# ── Secrets (never scan) ─────────────────────────────────────
.env
.env.local
.env.*.local
*.pem
*.key
*.p12

# ── Large/generated dirs ─────────────────────────────────────
vendor/
.wasm/

# ── UA / Flywheel artifacts (generated, not source) ──────────
.understand-anything/
.hermes/code-state/
```

### Test fixture structure

```
tests/code_scan/fixtures/
├── small_project/
│   ├── src/
│   │   ├── main.py          # 20 lines, imports os, sys
│   │   └── utils.py         # 15 lines, imports json
│   ├── tests/
│   │   └── test_main.py     # 10 lines
│   ├── pyproject.toml       # 5 lines
│   └── README.md            # 3 lines
├── mixed_project/
│   ├── src/
│   │   ├── index.ts         # 25 lines, imports from react
│   │   └── helpers.js       # 20 lines
│   ├── package.json         # framework detection: react
│   └── .hermesignore        # custom: exclude vendor/
└── ignored_project/
    ├── src/main.py           # should be scanned
    └── node_modules/pkg.js   # should be excluded
```

## Complexity Tier

**T2** — Multi-file implementation with test fixtures, but low algorithmic complexity. All logic is deterministic lookups, tree walking, and JSON serialization. Estimated 8–12 subagent iterations across 7–8 target files. Requires coder subagent + Hermes verification + reviewer signoff before commit.

Breakdown:
- ~5 iterations for `language_registry.py` + `test_language_registry.py`
- ~5 iterations for `scan_project.py` + `test_scan_project.py`
- ~4 iterations for `graph_schema.py` + `test_graph_schema.py`
- ~2 iterations for `.hermesignore` + fixture trees
- ~2 iterations for integration cross-validation

## Execution Engine

**Executor:** `delegate-coder` — Hermes dispatches a coder subagent via `delegate_task` with full plan context.

**Process:**
1. Coder subagent implements the files in strict TDD order (language_registry → graph_schema → scan_project → .hermesignore).
2. Hermes runs `pytest tests/code_scan/ -v` after each module completion.
3. Reviewer subagent validates: spec compliance, scope guardrails, no forbidden-file touches, context-budget cleanliness.
4. On reviewer PASS, Hermes presents the evidence bundle to JC for commit approval.

**Routing detail:** No external CLI executor (claude-code/codex). Hermes delegates internally.

**Subagent reliability preflight:**
- Task shape: deterministic script + test authoring
- Expected artifacts: 4 source files, 3 test files, optional empty test package marker, fixture tree
- `max_iterations`: 20 per subagent dispatch
- File-write: YES. Run-test: YES. Commit: NO (await JC approval).

## Required Inline Context

### Schema contract (scan_project.py JSON output)

The JSON emitted by `scan_project.py` must be parseable by downstream consumers (Phase 2's `extract_imports.py`, JIT skills, validation gate). Required top-level keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `project_root` | string | yes | Absolute path to the scanned project |
| `scanned_at` | string | yes | ISO 8601 timestamp |
| `total_files` | integer | yes | Count of files in `files` array |
| `total_lines` | integer | yes | Sum of `lines` across all files |
| `languages` | object | yes | `{"language_name": file_count, ...}` |
| `categories` | object | yes | `{"category_name": file_count, ...}` |
| `frameworks` | array | yes | List of detected framework names |
| `files` | array | yes | Per-file records (see below) |
| `warnings` | array | yes | Non-blocking issue strings |

Per-file record keys:
| Key | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Absolute file path |
| `relative_path` | string | yes | Path relative to `project_root` |
| `language` | string | yes | From `LANGUAGE_BY_EXT` or `"unknown"` |
| `category` | string | yes | From category logic |
| `lines` | integer | yes | Physical line count |
| `size_bytes` | integer | yes | File size on disk |

### Schema contract (graph_schema.py validation)

`validate_graph(graph: dict)` returns `{"issues": [...], "warnings": [...]}`:
- `issues`: list of strings — critical problems that invalidate the graph
- `warnings`: list of strings — non-blocking concerns

Node validation checks:
- `node_type` present and resolves (via direct enum or alias)
- `node_id` present and string
- `filePath` present and relative

Edge validation checks:
- `edge_type` present and resolves (via direct enum or alias)
- `source` and `target` reference existing node IDs
- No self-referencing edges (unless explicitly allowed in future)

### Existing dirty files — DO NOT TOUCH

```
tools/skills_sync.py                 # 596 LOC — skills manifest syncer (dirty)
tests/tools/test_skills_sync.py      # dirty test counterpart
```
These files have uncommitted changes on the current branch that are **unrelated** to Phase 1. The coder subagent must not read, modify, or commit them. If a commit is approved, only the Phase 1 files listed in `allowed_files` should be staged.

### Project context for this bead

- **Repo:** `/home/jarrad/.hermes/hermes-agent` (hermes-agent)
- **Current branch:** `docs/ua-flywheel-phase1-phase2-plan`
- **Python:** 3.10+ (`venv` currently has pytest in this checkout; use `source venv/bin/activate` for this bead)
- **Test runner:** `python -m pytest` (or `scripts/run_tests.sh` for CI-style isolation)
- **No new pip dependencies** — stdlib only (`os`, `pathlib`, `json`, `argparse`, `hashlib`, `re`, `datetime`, `dataclasses`)

## Dependencies

### Internal dependencies

| Dependency | Type | Status |
|---|---|---|
| `scripts/code-scan/language_registry.py` | runtime import (by `scan_project.py`) | Created by this bead |
| `scripts/code-scan/graph_schema.py` | runtime import (future Phase 2 consumers) | Created by this bead |
| `.hermesignore` | read at runtime (by `scan_project.py`) | Created by this bead |

### External dependencies

| Dependency | Type | Status |
|---|---|---|
| Python 3.10+ stdlib | runtime | Assumed present |
| pytest | test runtime | Assumed present in `.venv` |
| None | runtime | **No new runtime dependencies allowed** |

### Phase gate dependency

Phase 2 **must not begin** until this bead is verified and committed. The Phase 2 prerequisite gate (`.plans/phase-2-flywheel-ua-integration.md` §Prerequisites) requires:
- `scan_project.py` exists and emits stable JSON
- `language_registry.py` exists and is imported by `scan_project.py`
- `graph_schema.py` exists and can validate node/edge contracts
- `.hermesignore` exists with default exclusions
- Phase 1 tests pass on selected test-bed repos

## Test Obligations

### TDD protocol

Strict RED → GREEN → REFACTOR for every new function or module. Every public function in `language_registry.py`, `scan_project.py`, and `graph_schema.py` must have at least one failing-first test.

### Test file mapping

| Source file | Test file | Coverage target |
|---|---|---|
| `language_registry.py` | `test_language_registry.py` | `get_language` (known ext, unknown ext, special filename, case sensitivity), `get_category` (path-based test detection, infra filenames, unknown), `detect_frameworks` (manifest parsing, missing manifest, multiple frameworks) |
| `scan_project.py` | `test_scan_project.py` | `.hermesignore` parsing (comments, blanks, patterns), directory walk (excludes `.git/`, `node_modules/`), JSON output schema compliance, stdout JSON mode, `--output` flag, error on non-existent path |
| `graph_schema.py` | `test_graph_schema.py` | `validate_node` (valid, missing type, invalid type, alias resolution), `validate_edge` (valid, missing nodes, self-ref), `validate_graph` (empty, valid-full, orphan nodes, mixed issues/warnings) |

### RED/GREEN/FULL evidence required

For each module, evidence must be presented:

**language_registry.py:**
- RED: Each test fails first with `AttributeError` / `ModuleNotFoundError`
- GREEN: Each test passes with minimal implementation
- FULL: `pytest tests/code_scan/test_language_registry.py -v` — all pass, no warnings

**scan_project.py:**
- RED: Tests fail with expected errors (script not yet implemented)
- GREEN: Tests pass against fixture trees
- FULL: `pytest tests/code_scan/test_scan_project.py -v` — all pass

**graph_schema.py:**
- RED: Tests fail (enums/functions not defined)
- GREEN: Tests pass
- FULL: `pytest tests/code_scan/test_graph_schema.py -v` — all pass

**Integration:**
- FULL: `pytest tests/code_scan/ -v` — all tests pass, no warnings
- FULL: Script execution against 3 test-bed repos produces valid JSON

### Fixture verification

Each fixture project in `tests/code_scan/fixtures/` must produce deterministic, assertion-matched output:
- `small_project`: expect 4 files, specific language/category counts, total line count match
- `mixed_project`: custom `.hermesignore` respected, framework detection finds `react`
- `ignored_project`: `node_modules/` excluded, `src/main.py` included

## Verification Command

Run the full verification bundle in this order. All commands must exit 0:

```bash
# ── Step 1: Unit + integration tests ─────────────────────────
cd /home/jarrad/.hermes/hermes-agent
source venv/bin/activate
python -m pytest tests/code_scan/ -v

# ── Step 2: Small repo scan (cass_memory_system) ─────────────
python scripts/code-scan/scan_project.py \
  /home/jarrad/.hermes/hermes-agent/cass_memory_system \
  --output /tmp/scan-cass.json
python -c "import json; d=json.load(open('/tmp/scan-cass.json')); assert d['total_files'] > 0; assert 'typescript' in d['languages']; print(f'SMALL PASS: {d[\"total_files\"]} files')"

# ── Step 3: Medium repo scan (mission-control) ───────────────
python scripts/code-scan/scan_project.py \
  /home/jarrad/.hermes/hermes-agent/mission-control \
  --output /tmp/scan-mc.json
python -c "import json; d=json.load(open('/tmp/scan-mc.json')); assert d['total_files'] > 0; assert 'typescript' in d['languages']; print(f'MEDIUM PASS: {d[\"total_files\"]} files')"

# ── Step 4: Large repo scan (hermes-agent) ───────────────────
python scripts/code-scan/scan_project.py \
  /home/jarrad/.hermes/hermes-agent \
  --output /tmp/scan-hermes.json
python -c "import json; d=json.load(open('/tmp/scan-hermes.json')); assert d['total_files'] > 0; assert len(d['warnings']) >= 0; print(f'LARGE PASS: {d[\"total_files\"]} files, {len(d[\"warnings\"])} warnings')"

# ── Step 5: Validate graph schema module ──────────────────────
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('scripts/code-scan/graph_schema.py')
spec = importlib.util.spec_from_file_location('graph_schema', path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
assert hasattr(module, 'validate_graph')
assert hasattr(module, 'NodeType')
assert hasattr(module, 'EdgeType')
print('IMPORT PASS')
PY

# ── Step 6: Scope guardrail (no forbidden patterns) ───────────
git diff --name-only -- scripts/ .hermesignore tests/code_scan/ | grep -q 'skills_sync' && echo 'GUARDRAIL FAIL' && exit 1 || echo 'GUARDRAIL PASS'

# ── Step 7: Existing unrelated dirty files still excluded ─────
# Run before implementation and preserve these snapshots for comparison:
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py > /tmp/phase1-forbidden-pre.diff
git diff --cached -- tools/skills_sync.py tests/tools/test_skills_sync.py > /tmp/phase1-forbidden-pre.cached.diff
# Run again after implementation:
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py > /tmp/phase1-forbidden-post.diff
git diff --cached -- tools/skills_sync.py tests/tools/test_skills_sync.py > /tmp/phase1-forbidden-post.cached.diff
cmp /tmp/phase1-forbidden-pre.diff /tmp/phase1-forbidden-post.diff
cmp /tmp/phase1-forbidden-pre.cached.diff /tmp/phase1-forbidden-post.cached.diff
echo 'FORBIDDEN FILE SNAPSHOT PASS'

# ── Step 8: No new runtime dependencies ───────────────────────
python - <<'PY'
from pathlib import Path
import ast
stdlib_ok = {
    'argparse', 'collections', 'dataclasses', 'datetime', 'enum', 'fnmatch',
    'hashlib', 'json', 'os', 'pathlib', 're', 'sys', 'typing'
}
for rel in ['scripts/code-scan/scan_project.py', 'scripts/code-scan/language_registry.py', 'scripts/code-scan/graph_schema.py']:
    tree = ast.parse(Path(rel).read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
    nonstdlib = sorted(i for i in imports if i not in stdlib_ok and i != 'language_registry')
    if nonstdlib:
        raise SystemExit(f'{rel}: non-stdlib imports: {nonstdlib}')
print('DEPENDENCY PASS')
PY
```

### Expected pass criteria

1. `pytest tests/code_scan/ -v` — 100% of code_scan tests pass
2. Small repo scan produces valid JSON with `total_files > 0`, TypeScript detected
3. Medium repo scan produces valid JSON with `total_files > 0`, TypeScript detected
4. Large repo scan completes without scanning `node_modules/` or `.git/`, produces valid JSON
5. `graph_schema.py` imports cleanly
6. No forbidden files in diff
7. No new non-stdlib imports in source files

## Approval Evidence

### Before commit — present this evidence bundle to JC

**1. Test output (verbatim):**
```
$ python -m pytest tests/code_scan/ -v
```
Include the full output showing PASS/FAIL for every test. All must be PASS.

**2. RED/GREEN/FULL evidence per module:**
For each of the three modules, confirm and record:
- [ ] Tests written first and observed failing (RED)
- [ ] Minimal code written to pass each test (GREEN)
- [ ] Full test suite passes with no warnings (FULL)
- [ ] Refactoring performed with tests still green

**3. Real-project verification output:**
```
SMALL PASS: <N> files
MEDIUM PASS: <N> files
LARGE PASS: <N> files, <N> warnings
IMPORT PASS
GUARDRAIL PASS
```

**4. Diff artifact:**
```bash
git diff -- scripts/ .hermesignore tests/code_scan/
```
Show the full diff. For new untracked files, use:
```bash
git diff --no-index -- /dev/null scripts/code-scan/scan_project.py
git diff --no-index -- /dev/null scripts/code-scan/language_registry.py
git diff --no-index -- /dev/null scripts/code-scan/graph_schema.py
git diff --no-index -- /dev/null .hermesignore
git diff --no-index -- /dev/null tests/code_scan/test_scan_project.py
git diff --no-index -- /dev/null tests/code_scan/test_language_registry.py
git diff --no-index -- /dev/null tests/code_scan/test_graph_schema.py
```

**5. Scope guardrail check:**
```bash
git diff --name-only | grep -vE '^(scripts/code-scan/|\.hermesignore|tests/code_scan/)'
```
Expected: empty output (only Phase 1 files touched).

**6. Existing dirty files preserved / not staged:**
```bash
cmp /tmp/phase1-forbidden-pre.diff /tmp/phase1-forbidden-post.diff
cmp /tmp/phase1-forbidden-pre.cached.diff /tmp/phase1-forbidden-post.cached.diff
```
Expected: both comparisons exit 0. The files may already be dirty before execution; this bead only requires that their unstaged and staged diffs are unchanged by Phase 1 work.

**7. Reviewer verdict:**
Reviewer subagent must return explicit PASS on:
- [ ] Spec compliance (all Phase 1 deliverables present and matching contracts)
- [ ] Scope preservation (no Phase 2/3/4 features, no forbidden patterns)
- [ ] No new runtime dependencies (stdlib only)
- [ ] Context budget (no SKILL.md files — scripts only, zero context impact)
- [ ] Quality and security (no secrets, no path leaks, proper error handling)
- [ ] Existing dirty files not modified

**8. Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
This bead implements Phase 1 foundation only.
Phase 2 remains blocked until JC approves this bead's commit.
```

Hermes records the approval decision and any conditions in the project-state ledger outside the coder subagent's allowed file set.

---

> **Bead execution readiness = this bead passes reviewer polish and JC approves execution.**
> **Bead completion = all verification commands exit 0 + reviewer PASS + JC commit approval.**
> Until then: do not start Phase 2 work.
