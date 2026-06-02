---
id: phase4-d5-semantic-extraction
title: Phase 4 D5 — Non-LLM Semantic Extraction
status: approved-pending
executor: delegate-coder
parallel_safe: false
allowed_files:
  - scripts/code-scan/semantic_extract.py
  - tests/code_scan/test_semantic_extract.py
  - tests/code_scan/fixtures/**
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - skills/code-analysis/code-scan/SKILL.md
depends_on:
  - phase4-d4-hub-ranking
verification:
  - python -m pytest tests/code_scan/test_semantic_extract.py -v
  - python -m pytest tests/code_scan/ -v --tb=short
risk: medium
---

# Bead: phase4-d5-semantic-extraction - Non-LLM Semantic Extraction

## Context & Intent

Extract bounded, deterministic semantic signals from source files so Hermes can see public symbols, docstrings, classes, decorators, route-like functions, and model-like structures before LLM interpretation.

## Implementation Details

- Create: `scripts/code-scan/semantic_extract.py`
- Create: `tests/code_scan/test_semantic_extract.py`
- Optional create: fixtures under `tests/code_scan/fixtures/semantic_extract/`

Required CLI:

```text
python scripts/code-scan/semantic_extract.py <scan.json> [--scan-root <project-root>] [--max-signals-per-file 50] > <semantic-signals.json>
```

`--scan-root` defaults to the project root recorded in `scan.json` when present, otherwise current working directory.

Required behavior:

- Python: use stdlib `ast` for module/class/function docstrings, function/class names, decorators, base classes, and simple annotated assignments.
- JS/TS: bounded regex for exports, function declarations, class declarations, and common route/listen patterns.
- Go/Rust/Shell: minimal regex extraction for top-level functions and main functions.
- Never execute or import target files.
- Cap per-file signals and include truncation flags.
- Fail soft on parse errors, recording warnings rather than aborting the entire scan.

Required output:

```json
{
  "schema_version": "1.0.0",
  "files": {
    "src/app.py": {
      "language": "python",
      "symbols": [{"kind": "function", "name": "main"}],
      "docstrings": [{"owner": "main", "summary": "Run CLI."}],
      "decorators": ["app.command"],
      "warnings": [],
      "truncated": false
    }
  },
  "totals": {"files_processed": 1, "symbols": 1, "warnings": 0}
}
```

## Complexity Tier

- T2 — AST/regex extraction with performance and overclaim risk.
- Execution routing: coder subagent + Hermes verification + reviewer required.

## Execution Engine

- hermes-coder

## Required Inline Context

This bead must not become an LLM summarizer. It extracts signals only. Use labels like `symbol`, `docstring`, `decorator`, `route_hint`; do not produce natural-language architecture conclusions.

## Dependencies

- D4 complete. D5 is conceptually independent but executes after D4 for serialized Phase 4 checkpoints.

## Test Obligations

- RED: tests for Python AST docstrings/decorators/classes and JS/TS regex extraction fail first.
- GREEN: focused semantic extraction tests pass.
- FULL: all `tests/code_scan/` pass.
- Include malformed-file tests and truncation/cap tests.

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
python -m pytest tests/code_scan/test_semantic_extract.py -v
python -m pytest tests/code_scan/ -v --tb=short
git diff --name-only -- tools/skills_sync.py tests/tools/test_skills_sync.py
python - <<'PY'
from pathlib import Path
for path in Path('scripts/code-scan').glob('*.py'):
    text = path.read_text(errors='ignore').lower()
    for needle in ['tree_sitter', 'web-tree-sitter', 'wasm', 'sqlite3', 'chromadb', 'openai', 'anthropic', 'litellm']:
        if needle in text:
            raise SystemExit(f'SCOPE FAIL: {needle} in {path}')
print('SCOPE GUARD PASS')
PY
```

## Approval Evidence

- Diff artifact: `/tmp/ua-phase4-d5-diff.patch` with line/byte counts.
- Scope/stale-language check: verify no LLM summary/output prose and no execution/import of target project modules.
- Subagent reliability requirement: expected artifacts are script/test/fixtures; no commit/push.
- Reviewer requirement: required for parser safety, signal caps, and deterministic-only boundary.
- Commit/push gate: explicit JC approval required after evidence bundle.
