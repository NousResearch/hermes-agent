---
id: phase4-d4-hub-ranking
title: Phase 4 D4 — Architectural Hub Ranking
status: approved-pending
executor: delegate-coder
parallel_safe: false
allowed_files:
  - scripts/code-scan/hub_ranking.py
  - tests/code_scan/test_hub_ranking.py
  - tests/code_scan/fixtures/**
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - skills/code-analysis/code-scan/SKILL.md
depends_on:
  - phase4-d3-orphan-triage
verification:
  - python -m pytest tests/code_scan/test_hub_ranking.py -v
  - python -m pytest tests/code_scan/ -v --tb=short
risk: medium
---

# Bead: phase4-d4-hub-ranking - Architectural Hub Ranking

## Context & Intent

Identify the files most likely to be architectural anchors so Hermes and subagents can prioritize reading high-signal modules first.

## Implementation Details

- Create: `scripts/code-scan/hub_ranking.py`
- Create: `tests/code_scan/test_hub_ranking.py`
- Optional create: graph fixtures under `tests/code_scan/fixtures/hub_ranking/`

Required CLI:

```text
python scripts/code-scan/hub_ranking.py <graph.json> [--classified-imports classified-imports.json] [--top 20] > <hubs.json>
```

Required behavior:

- Compute in-degree and out-degree for project file nodes.
- Score hubs with a simple deterministic formula.
- Prefer project-local edges when classified import information is available.
- Exclude docs/config/assets from top hubs unless explicitly requested.
- Include confidence/coverage notes when classification data is absent.

Required output:

```json
{
  "schema_version": "1.0.0",
  "hub_rankings": [
    {"node_id": "file:src/app.py", "file_path": "src/app.py", "hub_score": 42, "in_degree": 10, "out_degree": 12, "confidence": "medium"}
  ],
  "entrypoint_like": [],
  "totals": {"files_ranked": 1, "top_n": 20}
}
```

## Complexity Tier

- T2 — Centrality scoring can mislead if overclaimed.
- Execution routing: coder subagent + Hermes verification + reviewer required.

## Execution Engine

- hermes-coder

## Required Inline Context

Hub scores are ranking hints, not proof of importance. The report must state this explicitly. Avoid complex graph algorithms unless they are stdlib-only, readable, and tested. No visualization.

## Dependencies

- D3 complete for cleaner orphan handling.
- Phase 3 graph assembly complete.

## Test Obligations

- RED: tests for in-degree, out-degree, sorted ranking, threshold filtering, and non-code exclusion fail first.
- GREEN: focused hub ranking tests pass.
- FULL: all `tests/code_scan/` pass.
- Include deterministic tie-breaking tests so output order is stable.

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
python -m pytest tests/code_scan/test_hub_ranking.py -v
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

- Diff artifact: `/tmp/ua-phase4-d4-diff.patch` with line/byte counts.
- Scope/stale-language check: verify output/prose says hub scores are hints and includes confidence/coverage.
- Subagent reliability requirement: expected artifacts are script/test/fixtures only; no commit/push.
- Reviewer requirement: required for scoring formula, deterministic ordering, and overclaim prevention.
- Commit/push gate: explicit JC approval required after evidence bundle.
