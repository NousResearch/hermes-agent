---
id: ua-tier1-005-run-ua-report-integration
title: Run UA, Report, and Context Integration
status: planned-not-approved
executor: codex-coder
parallel_safe: false
risk: high
---

# Bead: ua-tier1-005-run-ua-report-integration - Run UA, Report, and Context Integration

## Context & Intent

Integrate Tier 1 static signals into the UA bundle, manifest, report, summary, and subagent context with explicit non-proof boundaries.

## Implementation Details

- Modify likely: `scripts/code-scan/run_ua.py`.
- Modify likely: `scripts/code-scan/report_data.py`.
- Modify likely: `scripts/code-scan/render_report.py`.
- Modify likely: `scripts/code-scan/build_context_bundle.py` if subagent-context ingestion requires it.
- Modify tests likely: `tests/code_scan/test_run_ua.py`, `tests/code_scan/test_report_data.py`, `tests/code_scan/test_render_report.py`, `tests/code_scan/test_build_context_bundle.py`, and `tests/code_scan/test_e2e_ua_workflow.py`.
- Add `static-signals.json` to the generated bundle and manifest integrity.
- Add bounded report/context summaries; top signals only, not full noisy dumps.
- This is the only Tier 1 bead planned to touch `run_ua.py`.
- Require reviewer PASS before acceptance.

## Complexity Tier

- T2/high-risk — cross-file integration on final bundle/report/context contracts.
- Expected implementation size: 300-600 LOC including tests/fixture updates.
- Execution routing: codex-coder + Hermes verification + reviewer required; do not use fast-coder as finisher.

## Execution Engine

- codex-coder / gpt-5.5 recommended because this is evidence-boundary-sensitive UA work.
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

Required report section language:

```markdown
## Static Signals

These are heuristic content markers only. They do not prove security, RLS correctness, auth correctness, runtime behavior, deployment readiness, CI success, or policy semantics.
```

Manifest/context requirements:

```text
static-signals.json must be listed in manifest artifact paths and covered by artifact integrity.
subagent-context.json may summarize top signals, but each summary must preserve claim_type=heuristic_signal and semantic_status=not_validated.
No Tier 1 output may use executed_external_gate unless a separately approved Tier 2 runner actually executed a command and captured evidence.
```

## Dependencies

- `.beads/ua-tier1-001-static-signals-schema.md` complete.
- `.beads/ua-tier1-002-supabase-migration-markers.md` complete.
- `.beads/ua-tier1-003-edge-package-config-markers.md` complete.
- `.beads/ua-tier1-004-entrypoint-hotspot-refinement.md` complete or explicitly deferred by Hermes with reviewer agreement.

## Test Obligations

- RED: integration tests fail before `static-signals.json` is emitted and registered.
- GREEN: focused integration tests pass for artifact existence, manifest hash, report section, summary counts, and subagent context top signals.
- Boundary tests: assert report/context do not claim security, RLS/auth correctness, runtime correctness, deployment readiness, or CI success.
- Golden fixture smoke: run UA against synthetic `tests/code_scan/fixtures/static_signals_supabase/` and inspect generated `static-signals.json`.
- FULL: `python -m pytest tests/code_scan -q`.

## Verification Command

```bash
cd /home/jarrad/work/hermes-agent-ua-local
python -m pytest tests/code_scan/test_static_signals.py tests/code_scan/test_run_ua.py tests/code_scan/test_report_data.py tests/code_scan/test_render_report.py tests/code_scan/test_build_context_bundle.py -q
python -m pytest tests/code_scan/test_e2e_ua_workflow.py -q
python -m pytest tests/code_scan -q
python -m py_compile scripts/code-scan/static_signals.py scripts/code-scan/run_ua.py scripts/code-scan/report_data.py scripts/code-scan/render_report.py scripts/code-scan/build_context_bundle.py
git diff --check
```

## Approval Evidence

- Diff artifact requirement: generate `/tmp/ua-tier1-005-run-ua-report-integration-diff.patch` with line/byte counts; include untracked new files via `git add -N` or `git diff --no-index`.
- Scope/stale-language check: search changed source/report/context files for forbidden certifier language and confirm matches are only disclaimers/tests for overclaim prevention.
- Subagent reliability requirement: coder may write implementation/tests and run local verification only; coder has no commit/push/merge/deploy authority. If timeout/no-summary occurs, Hermes must inspect actual files/diff before retrying or accepting.
- Reviewer requirement: reviewer PASS required for spec compliance, evidence-boundary preservation, final bundle consistency, manifest/subagent-context ordering, and overclaim risk before acceptance.
- Commit/push gate: explicit JC approval required after evidence bundle; this bead grants no commit or push authority.
