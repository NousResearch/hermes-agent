---
id: phase4-d2-entrypoint-detection
title: Phase 4 D2 — Entrypoint Detection
status: approved-pending
executor: delegate-coder
parallel_safe: false
allowed_files:
  - scripts/code-scan/detect_entrypoints.py
  - tests/code_scan/test_detect_entrypoints.py
  - tests/code_scan/fixtures/**
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - skills/code-analysis/code-scan/SKILL.md
depends_on:
  - phase4-d1-import-classification
verification:
  - python -m pytest tests/code_scan/test_detect_entrypoints.py -v
  - python -m pytest tests/code_scan/ -v --tb=short
risk: medium
---

# Bead: phase4-d2-entrypoint-detection - Entrypoint Detection

## Context & Intent

Help Hermes find where execution likely starts in a repo, producing deterministic entrypoint hints before subagents begin deep reading.

## Implementation Details

- Create: `scripts/code-scan/detect_entrypoints.py`
- Create: `tests/code_scan/test_detect_entrypoints.py`
- Optional create: fixtures under `tests/code_scan/fixtures/entrypoints/`
- Do not wire this into `scan_project.py` yet unless JC separately approves skill/pipeline integration.

Required CLI:

```text
python scripts/code-scan/detect_entrypoints.py <scan.json> > <entrypoints.json>
```

Minimum detection patterns:

- Python: `if __name__ == "__main__"`, `def main`, `__main__.py`, `argparse`, `typer.Typer`, `click.command`, FastAPI/Uvicorn app startup hints.
- JS/TS: `index.js`, `main.ts`, `app.listen`, `package.json` `main`/`bin`/`scripts.start` hints where available from scanned files.
- Go: `package main` + `func main`.
- Rust: `fn main` in `src/main.rs` or `src/bin/*.rs`.
- Shell: executable-looking scripts under root, `bin/`, or `scripts/` based on file path and shebang.

Required output:

```json
{
  "schema_version": "1.0.0",
  "entrypoints": [
    {
      "file": "src/cli.py",
      "language": "python",
      "type": "python_cli",
      "signals": ["def main", "if __name__ == '__main__'"],
      "confidence": 0.95
    }
  ],
  "totals": {"entrypoints_found": 1, "by_type": {"python_cli": 1}}
}
```

## Complexity Tier

- T2 — Multi-language heuristics with false-positive risk.
- Execution routing: coder subagent + Hermes verification + reviewer required.

## Execution Engine

- hermes-coder

## Required Inline Context

Entrypoints are hints, not authoritative control-flow proofs. False positives must be labelled with confidence and evidence. Do not execute target repo code. Do not import scanned modules.

Guardrails:

```text
JIT-only. Read-only against target repos. No auto-injection. No LLM summaries. No new dependencies. No tree-sitter/WASM. No edits to tools/skills_sync.py or tests/tools/test_skills_sync.py.
```

## Dependencies

- D1 complete, so report-builder later can combine import classification and entrypoint facts.

## Test Obligations

- RED: tests for each supported language/pattern fail before implementation.
- GREEN: focused entrypoint tests pass.
- FULL: all `tests/code_scan/` pass.
- Include negative tests: helper functions named `main_helper`, docs mentioning `main`, and non-executable files should not be high-confidence entrypoints.

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
python -m pytest tests/code_scan/test_detect_entrypoints.py -v
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

- Diff artifact: `/tmp/ua-phase4-d2-diff.patch` with line/byte counts.
- Scope/stale-language check: no claims that entrypoints are guaranteed true; output must use confidence/evidence language.
- Subagent reliability requirement: coder expected to create exactly one script and one test file plus fixtures; no commit/push.
- Reviewer requirement: required for false-positive control, confidence labelling, and no-execution guarantee.
- Commit/push gate: explicit JC approval required after evidence bundle.
