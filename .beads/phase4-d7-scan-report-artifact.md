---
id: phase4-d7-scan-report-artifact
title: Phase 4 D7 — Scan-to-Report Artifact
status: approved-pending
executor: delegate-coder
parallel_safe: false
allowed_files:
  - scripts/code-scan/report_builder.py
  - tests/code_scan/test_report_builder.py
  - tests/code_scan/test_phase4_pipeline.py
  - tests/code_scan/fixtures/**
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - skills/code-analysis/code-scan/SKILL.md
depends_on:
  - phase4-d1-import-classification
  - phase4-d2-entrypoint-detection
  - phase4-d3-orphan-triage
  - phase4-d4-hub-ranking
  - phase4-d5-semantic-extraction
  - phase4-d6-delta-reporting
verification:
  - python -m pytest tests/code_scan/test_report_builder.py tests/code_scan/test_phase4_pipeline.py -v
  - python -m pytest tests/code_scan/ -v --tb=short
risk: high
---

# Bead: phase4-d7-scan-report-artifact - Scan-to-Report Artifact

## Context & Intent

Create the compact artifact that gives Hermes and specialist subagents a fast, deep starting point for understanding a codebase without loading raw graphs or full source trees into context.

## Implementation Details

- Create: `scripts/code-scan/report_builder.py`
- Create: `tests/code_scan/test_report_builder.py`
- Create: `tests/code_scan/test_phase4_pipeline.py`
- Optional create: report fixtures under `tests/code_scan/fixtures/report_builder/`
- Do not update `skills/code-analysis/code-scan/SKILL.md` in this bead unless JC explicitly expands scope.

Required CLI:

```text
python scripts/code-scan/report_builder.py \
  --scan scan.json \
  --imports imports.json \
  --classified-imports classified-imports.json \
  --entrypoints entrypoints.json \
  --graph graph.json \
  --orphan-triage orphan-triage.json \
  --hubs hubs.json \
  --semantic semantic-signals.json \
  --delta delta.json \
  --output UA_REPORT.md
```

Required report sections:

1. Project Overview
2. Deterministic Inventory
3. Entrypoints / Where to Start
4. Architectural Hubs
5. Import Profile
6. Orphan Triage
7. Semantic Signals
8. Delta Summary, if provided
9. Suggested Reading Path
10. Validation/Caveats

Required behavior:

- Accept partial artifacts; render only sections with available data.
- Enforce size cap with truncation warning.
- Prefer deterministic facts and caveats over prose claims.
- Output Markdown by default; JSON summary optional if simple.
- D7 may be split into D7a JSON report and D7b Markdown renderer if implementation exceeds a single safe bead.

## Complexity Tier

- T3 unless split — It integrates all upstream outputs and becomes the primary agent-facing artifact.
- Execution routing: coder subagent + Hermes verification + reviewer required + JC checkpoint before implementation/commit.

## Execution Engine

- hermes-coder, checkpointed.

## Required Inline Context

This report is a JIT handoff artifact, not a stored knowledge base. It must not auto-run or persist globally. It must not invent conclusions that are not present in deterministic artifacts.

## Dependencies

- D1-D6 complete and reviewer-approved.

## Test Obligations

- RED: tests for scan-only report, full-artifact report, missing optional artifacts, table formatting, size cap, and truncation warning fail first.
- GREEN: focused report tests pass.
- PIPELINE: `test_phase4_pipeline.py` chains scan/import/graph/enricher/report artifacts on fixtures and validates cross-artifact counts.
- FULL: all `tests/code_scan/` pass.

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
python -m pytest tests/code_scan/test_report_builder.py tests/code_scan/test_phase4_pipeline.py -v
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

- Diff artifact: `/tmp/ua-phase4-d7-diff.patch` with line/byte counts.
- Generated sample report from hermes-agent or fixture repo.
- Scope/stale-language check: report must say results are deterministic hints and include caveats/truncation notes.
- Subagent reliability requirement: expected artifacts are `report_builder.py`, focused tests, and pipeline tests; no commit/push.
- Reviewer requirement: mandatory PASS for output contract, size cap, caveats, and integration correctness.
- JC gate: D7 should not be committed/pushed until JC reviews a sample report and approves the artifact contract.
