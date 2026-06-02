---
id: phase4-d6-delta-reporting
title: Phase 4 D6 — Delta Reporting
status: approved-pending
executor: delegate-coder
parallel_safe: false
allowed_files:
  - scripts/code-scan/delta_report.py
  - tests/code_scan/test_delta_report.py
  - tests/code_scan/fixtures/**
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - skills/code-analysis/code-scan/SKILL.md
depends_on:
  - phase4-d5-semantic-extraction
verification:
  - python -m pytest tests/code_scan/test_delta_report.py -v
  - python -m pytest tests/code_scan/ -v --tb=short
risk: medium
---

# Bead: phase4-d6-delta-reporting - Delta Reporting

## Context & Intent

Let Hermes compare two UA scan/fingerprint snapshots and quickly understand what changed between repo states, branches, or review checkpoints.

## Implementation Details

- Create: `scripts/code-scan/delta_report.py`
- Create: `tests/code_scan/test_delta_report.py`
- Optional create: fixtures under `tests/code_scan/fixtures/delta_report/`

Required CLI:

```text
python scripts/code-scan/delta_report.py <old-scan.json> <new-scan.json> [--old-fingerprints old.json] [--new-fingerprints new.json] > <delta.json>
```

Required behavior:

- Compare file sets: added, removed, common.
- Compare language/category/framework counts.
- If fingerprint snapshots are supplied, summarize unchanged/cosmetic/structural changes using Phase 3 fingerprint semantics.
- Optionally compare import totals if import/classification artifacts are supplied in a later extension, but do not block core D6 on that.
- Do not create or persist snapshots; consume only user-supplied artifacts.

Required output:

```json
{
  "schema_version": "1.0.0",
  "files": {"added": [], "removed": [], "common_count": 10},
  "languages": {"python": {"old": 9, "new": 10, "delta": 1}},
  "categories": {},
  "frameworks": {"added": [], "removed": []},
  "fingerprints": {"UNCHANGED": 8, "COSMETIC": 1, "STRUCTURAL": 1},
  "warnings": []
}
```

## Complexity Tier

- T2 — Snapshot comparison must be precise and schema-aware.
- Execution routing: coder subagent + Hermes verification + reviewer required.

## Execution Engine

- hermes-coder

## Required Inline Context

D6 must not mutate `.hermes/code-state/fingerprints.json`, run scans by itself, or maintain history. It only compares artifacts passed explicitly on the command line.

## Dependencies

- D5 complete for serialized Phase 4 execution.
- Phase 3 fingerprints available for optional fingerprint delta tests.

## Test Obligations

- RED: tests for identical scans, added files, removed files, language deltas, framework deltas, missing snapshot handling fail first.
- GREEN: focused delta tests pass.
- FULL: all `tests/code_scan/` pass.
- Include schema-version mismatch behavior and clear error messages.

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
python -m pytest tests/code_scan/test_delta_report.py -v
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

- Diff artifact: `/tmp/ua-phase4-d6-diff.patch` with line/byte counts.
- Scope/stale-language check: verify no snapshot persistence, no background scan, and no mutation of `.hermes/code-state`.
- Subagent reliability requirement: expected artifacts are script/test/fixtures; no commit/push.
- Reviewer requirement: required for snapshot semantics, schema mismatch behavior, and no-mutation guarantee.
- Commit/push gate: explicit JC approval required after evidence bundle.
