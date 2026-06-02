---
id: phase4-d3-orphan-triage
title: Phase 4 D3 — Orphan Warning Triage
status: approved-pending
executor: delegate-coder
parallel_safe: false
allowed_files:
  - scripts/code-scan/triage_orphans.py
  - tests/code_scan/test_triage_orphans.py
  - tests/code_scan/fixtures/**
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - skills/code-analysis/code-scan/SKILL.md
depends_on:
  - phase4-d2-entrypoint-detection
verification:
  - python -m pytest tests/code_scan/test_triage_orphans.py -v
  - python -m pytest tests/code_scan/ -v --tb=short
risk: medium
---

# Bead: phase4-d3-orphan-triage - Orphan Warning Triage

## Context & Intent

Reduce validation-gate noise by classifying graph orphan nodes into expected, entrypoint-candidate, suspicious, and unknown groups so Hermes can focus on meaningful architecture gaps.

## Implementation Details

- Create: `scripts/code-scan/triage_orphans.py`
- Create: `tests/code_scan/test_triage_orphans.py`
- Optional create: graph/scan fixtures under `tests/code_scan/fixtures/orphan_triage/`
- Do not change `graph_schema.py` verdict rules in this bead.

Required CLI:

```text
python scripts/code-scan/triage_orphans.py <graph.json> <scan.json> [--entrypoints entrypoints.json] > <orphan-triage.json>
```

Required categories:

- `expected`: docs, config, tests, fixtures, workflows, images/assets, templates.
- `entrypoint_candidate`: source files orphaned in import graph but detected as likely standalone entrypoints.
- `suspicious`: source files that are orphaned and not plausible entrypoints.
- `unknown`: missing metadata or unsupported language.

Required output:

```json
{
  "schema_version": "1.0.0",
  "orphans": {
    "expected": [{"node_id": "file:README.md", "reason": "doc"}],
    "entrypoint_candidate": [],
    "suspicious": [{"node_id": "file:src/legacy.py", "reason": "unreferenced source"}],
    "unknown": []
  },
  "totals": {"total_orphans": 2, "expected": 1, "entrypoint_candidate": 0, "suspicious": 1, "unknown": 0}
}
```

## Complexity Tier

- T1/T2 — Small script, but affects how validation warnings are interpreted.
- Execution routing: coder subagent + Hermes verification + reviewer required.

## Execution Engine

- hermes-coder

## Required Inline Context

Orphan triage must not change Validation Gate semantics. `validation-gate` still reports schema issues/warnings deterministically. This bead only creates a companion artifact that summarizes which orphan warnings deserve human/agent attention.

## Dependencies

- D2 complete, so entrypoint candidates can be downgraded from suspicious when evidence exists.
- Phase 3 graph assembly complete.

## Test Obligations

- RED: tests for docs/config/tests/assets/templates expected orphans and source-code suspicious orphans fail first.
- GREEN: focused triage tests pass.
- FULL: all `tests/code_scan/` pass.
- Include integration test using a real assembled graph artifact fixture with multiple orphan classes.

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
python -m pytest tests/code_scan/test_triage_orphans.py -v
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

- Diff artifact: `/tmp/ua-phase4-d3-diff.patch` with line/byte counts.
- Scope/stale-language check: verify no wording says orphan triage is a schema-validity verdict; it is an interpretation aid only.
- Subagent reliability requirement: expected artifacts are one script, one test file, fixtures; no commit/push.
- Reviewer requirement: required for validation semantics preservation and suspicious/expected category reasonableness.
- Commit/push gate: explicit JC approval required after evidence bundle.
