---
id: ua-tier1-004-entrypoint-hotspot-refinement
title: Entrypoint and Hotspot Refinement
status: completed
executor: codex-coder
parallel_safe: false
risk: medium
---

# Bead: ua-tier1-004-entrypoint-hotspot-refinement - Entrypoint and Hotspot Refinement

## Context & Intent

Reduce noisy coding-agent handoffs by improving top entrypoint/hotspot selection without hiding raw inventory.

## Implementation Details

- Modify likely: `scripts/code-scan/detect_entrypoints.py`.
- Modify likely: `scripts/code-scan/recommended_files.py`.
- Modify tests likely: `tests/code_scan/test_detect_entrypoints.py`, `tests/code_scan/test_build_context_bundle.py`, or existing recommended-files tests if present.
- Add or refine noise filters for docs/assets/beads/root metadata in ranked entrypoint/hotspot summaries.
- Prefer framework/application entrypoints and Supabase function entrypoints.
- Do not delete raw file inventory or change graph construction semantics.

## Complexity Tier

- T2 — additive ranking/filtering that can affect subagent context quality.
- Expected implementation size: 150-350 LOC including tests.
- Execution routing: coder + Hermes verification + reviewer.

## Execution Engine

- codex-coder / gpt-5.5 recommended because this is evidence-boundary-sensitive UA work.
- `fast-coder` may provide a first pass only if explicitly approved for a low-risk slice; it must not be treated as autonomous finisher.
- Hermes owns verification and integration.

## Required Inline Context

Approved planning-only scope quote to preserve verbatim:

```text
[JC] Approve planning package for UA Tier 1 static-signals layer only:
create/update the Tier 1 plan package and beads under .hermes/plans, .beads, and .hermes/handoffs;
do not execute implementation beads yet;
do not modify run_ua.py or production code in this approval;
do not commit or push without a separate explicit approval.
```

Core UA evidence-boundary contract:

```text
Tier 1 static signals are heuristic content markers only. They do not prove security, RLS correctness, auth correctness, runtime behavior, deployment readiness, CI success, or policy semantics. Every emitted Tier 1 claim must be labelled heuristic_signal and not_validated unless it is an existing deterministic inventory fact from Tier 0.
```

Noise candidates to exclude from top entrypoint/hotspot recommendations, while preserving raw inventory elsewhere:

```text
.beads/**
docs/**
*.png
*.jpg
*.jpeg
*.webp
*.svg
.gitignore
.vercelignore
AGENTS.md
CLAUDE.md
PLAN.md
package-lock.json
```

Preferred candidates include:

```text
src/main.*
src/App.*
vite.config.*
supabase/functions/*/index.*
package.json scripts and framework roots
```

## Dependencies

- Phase 6 recommended-files and handoff safeguards complete on current branch.
- Can execute after T1-001; does not require T1-002/T1-003 unless integration is deliberately combined later.

## Test Obligations

- RED: add fixture/ranking test showing noisy docs/assets/beads/root metadata currently rank too highly or are not filtered.
- GREEN: focused entrypoint/recommended-files tests pass and preserve raw inventory availability.
- Regression: existing Phase 6 context/recommended-files tests still pass.
- FULL: `python -m pytest tests/code_scan -q`.

## Verification Command

```bash
cd /home/jarrad/work/hermes-agent-ua-local
python -m pytest tests/code_scan/test_detect_entrypoints.py tests/code_scan/test_build_context_bundle.py -q
python -m pytest tests/code_scan -q
python -m py_compile scripts/code-scan/detect_entrypoints.py scripts/code-scan/recommended_files.py
git diff --check
```

## Approval Evidence

- Diff artifact requirement: generate `/tmp/ua-tier1-004-entrypoint-hotspot-refinement-diff.patch` with line/byte counts; include untracked new files via `git add -N` or `git diff --no-index`.
- Scope/stale-language check: search changed source/report/context files for forbidden certifier language and confirm matches are only disclaimers/tests for overclaim prevention.
- Subagent reliability requirement: coder may write implementation/tests and run local verification only; coder has no commit/push/merge/deploy authority. If timeout/no-summary occurs, Hermes must inspect actual files/diff before retrying or accepting.
- Reviewer requirement: reviewer PASS required for spec compliance, evidence-boundary preservation, and overclaim risk before acceptance.
- Commit/push gate: explicit JC approval required after evidence bundle; this bead grants no commit or push authority.
