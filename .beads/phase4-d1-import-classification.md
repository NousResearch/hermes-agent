---
id: phase4-d1-import-classification
title: Phase 4 D1 — Import Classification
status: active
executor: delegate-coder
parallel_safe: false
allowed_files:
  - scripts/code-scan/classify_imports.py
  - tests/code_scan/test_classify_imports.py
  - tests/code_scan/fixtures/**
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - skills/code-analysis/code-scan/SKILL.md
depends_on:
  - phase2-d1-extract-imports
verification:
  - python -m pytest tests/code_scan/test_classify_imports.py -v
  - python -m pytest tests/code_scan/ -v --tb=short
risk: medium
---

# Bead: phase4-d1-import-classification - Import Classification

## Context & Intent

Help Hermes tell whether raw imports are project-local, Python/Node/Go/Rust standard library, third-party package, relative import, or unknown, so architecture summaries can separate project coupling from dependency usage.

## Implementation Details

- Create: `scripts/code-scan/classify_imports.py`
- Create: `tests/code_scan/test_classify_imports.py`
- Optional create: small fixtures under `tests/code_scan/fixtures/import_classification/`
- Do not modify `extract_imports.py` unless reviewer approves a tiny backwards-compatible bug fix discovered during TDD.

Required script behavior:

```text
python scripts/code-scan/classify_imports.py <scan.json> <imports.json> > <classified-imports.json>
```

Required output shape:

```json
{
  "schema_version": "1.0.0",
  "source_scan": "...",
  "source_imports": "...",
  "files": {
    "src/app.py": {
      "imports": [
        {"module": "os", "classification": "stdlib"},
        {"module": "fastapi", "classification": "third_party"},
        {"module": "mcp_agent_mail.db", "classification": "local"}
      ],
      "totals": {"local": 1, "stdlib": 1, "third_party": 1, "relative": 0, "unknown": 0}
    }
  },
  "totals": {"local": 1, "stdlib": 1, "third_party": 1, "relative": 0, "unknown": 0}
}
```

Implementation rules:

- Stdlib only: use `sys.stdlib_module_names` where available plus static fallback sets for JS/TS/Go/Rust basics.
- Derive local module roots from scanned file paths and package-like directories.
- Detect relative imports before stdlib/third-party classification.
- Treat uncertain cases as `unknown`, not fabricated third-party facts.
- Keep classification additive; do not break `extract_imports.py` schema.

## Complexity Tier

- T2 — New script plus classification heuristics and fixture coverage.
- Execution routing: coder subagent + Hermes verification + reviewer required.

## Execution Engine

- hermes-coder

## Required Inline Context

Existing import map shape from Phase 2 uses top-level keys including `schema_version`, `source_scan`, `generated_at`, `files`, and `totals`. The `files` object is keyed by path and each entry includes an `imports` list. Preserve raw module names from that artifact; this bead only enriches classification.

Guardrails to copy into coder prompt verbatim:

```text
JIT-only. No dashboard/UI. No auto-injection. No SQLite/vector store. No tree-sitter/WASM. No new runtime dependencies. Do not edit tools/skills_sync.py or tests/tools/test_skills_sync.py. No commit/push authority.
```

## Dependencies

- `phase2-d1-extract-imports` complete.
- Existing Phase 3 graph assembly must remain passing.

## Test Obligations

- RED: add failing tests for stdlib, third-party, local, relative, and unknown classifications before implementation.
- GREEN: focused `test_classify_imports.py` passes.
- FULL: `python -m pytest tests/code_scan/ -v --tb=short` passes.
- Include CLI tests for valid input, missing input, malformed JSON, and output totals.

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
python -m pytest tests/code_scan/test_classify_imports.py -v
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

- Diff artifact: `/tmp/ua-phase4-d1-diff.patch` with line/byte counts, including new files via `git add -N` or `git diff --no-index`.
- Scope/stale-language check: verify no references to dashboard/UI, auto-injection, SQLite, tree-sitter/WASM, or new runtime deps in implementation files except guardrail prose in bead/docs.
- Subagent reliability requirement: expected artifacts are exactly the script and tests listed above; coder may write files and run tests, but may not commit/push.
- Reviewer requirement: required for spec compliance, output schema quality, and false-positive classification risk.
- Commit/push gate: explicit JC approval required after evidence bundle.
