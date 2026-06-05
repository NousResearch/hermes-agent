---
id: ua-tier1-001-static-signals-schema
title: Static Signals Schema and Harness
status: completed
executor: codex-coder
parallel_safe: false
risk: medium
---

# Bead: ua-tier1-001-static-signals-schema - Static Signals Schema and Harness

## Context & Intent

Create the Tier 1 `static-signals.json` schema and pure helper harness before adding detectors or UA bundle integration.

## Implementation Details

- Create: `scripts/code-scan/static_signals.py`.
- Create: `tests/code_scan/test_static_signals.py`.
- Do not modify `scripts/code-scan/run_ua.py`, `report_data.py`, `render_report.py`, or production code in this bead.
- Define a minimal artifact builder returning `schema_version`, `claim_type`, `semantic_status`, `signals`, `summary`, and `boundaries`.
- Define signal record helpers with fields for `surface`, `path`, `line`, `marker_type`, `marker`, `claim_type`, `semantic_status`, and `boundary`.
- Keep implementation stdlib-only.

## Complexity Tier

- T1/T2 boundary — small new module and tests, but evidence-boundary-sensitive.
- Expected implementation size: under 200 LOC plus tests.
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

Expected empty artifact shape:

```json
{
  "schema_version": "1.0.0",
  "claim_type": "heuristic_signal",
  "semantic_status": "not_validated",
  "signals": [],
  "summary": {"total_signals": 0, "by_surface": {}, "by_marker_type": {}},
  "boundaries": ["Tier 1 static signals are content markers only; they do not prove security, RLS correctness, auth correctness, runtime behavior, deployment readiness, CI success, or policy semantics."]
}
```

## Dependencies

- Phase 6 final bundle consistency fix must remain present on this branch.
- No Tier 1 implementation bead dependency.

## Test Obligations

- RED: add tests expecting `build_static_signals_artifact([])` or equivalent schema helper before implementation.
- GREEN: focused `tests/code_scan/test_static_signals.py` passes.
- Boundary tests must assert `claim_type == "heuristic_signal"`, `semantic_status == "not_validated"`, and disclaimer text includes `does not prove security` and `RLS correctness`.
- FULL: `python -m pytest tests/code_scan -q` after focused pass.

## Verification Command

```bash
cd /home/jarrad/work/hermes-agent-ua-local
python -m pytest tests/code_scan/test_static_signals.py -q
python -m pytest tests/code_scan -q
python -m py_compile scripts/code-scan/static_signals.py
git diff --check
```

## Approval Evidence

- Diff artifact requirement: generate `/tmp/ua-tier1-001-static-signals-schema-diff.patch` with line/byte counts; include untracked new files via `git add -N` or `git diff --no-index`.
- Scope/stale-language check: search changed source/report/context files for forbidden certifier language and confirm matches are only disclaimers/tests for overclaim prevention.
- Subagent reliability requirement: coder may write implementation/tests and run local verification only; coder has no commit/push/merge/deploy authority. If timeout/no-summary occurs, Hermes must inspect actual files/diff before retrying or accepting.
- Reviewer requirement: reviewer PASS required for spec compliance, evidence-boundary preservation, and overclaim risk before acceptance.
- Commit/push gate: explicit JC approval required after evidence bundle; this bead grants no commit or push authority.
