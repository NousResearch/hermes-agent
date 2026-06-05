# UA Tier 1 Static Signals Layer — Planning Package

> Status: execution approved by JC on 2026-06-05; serial bead execution in progress.
> Branch: `feat/ua-tier1-static-signals`
> Baseline: `97dcc97e1 fix(code-scan): enforce final bundle consistency`

## Approved Scope Quote

```text
[JC] Approve planning package for UA Tier 1 static-signals layer only:
create/update the Tier 1 plan package and beads under .hermes/plans, .beads, and .hermes/handoffs;
do not execute implementation beads yet;
do not modify run_ua.py or production code in this approval;
do not commit or push without a separate explicit approval.
```

## Goal

Add a realistic Tier 1 coding-systems layer to Understand-Anything: deterministic content-marker inventory plus heuristic routing signals that help coding/review agents decide what to inspect first while preserving UA's high-trust scout role.

Tier 1 must enrich handoff quality without becoming a security reviewer, runtime validator, RLS validator, auth certifier, deployment readiness checker, or CI substitute.

## Tier 0 Invariant

UA remains a high-trust scout. It can report deterministic facts, inventory, hashes, manifests, graph relationships, content-marker presence, and suggested gates. It must not convert marker presence into proof of correctness.

All new Tier 1 claims must use:

- `heuristic_signal`
- `not_validated`
- explicit boundary text saying marker presence is not semantic validation

Forbidden Tier 1 claims unless inside disclaimers/tests:

- `secure`
- `certified`
- `RLS correct`
- `auth correct`
- `policy validated`
- `runtime verified`
- `deployment ready`
- `CI passed`

## Architecture

Tier 1 adds one additive artifact, expected to be named `static-signals.json`, plus report/context summaries after integration.

Recommended implementation shape:

```text
scripts/code-scan/static_signals.py
  - schema helpers
  - marker inventory over known domain surfaces
  - bounded line/context snippets
  - summaries by surface and marker type

tests/code_scan/test_static_signals.py
  - schema tests
  - marker fixture tests
  - boundary/overclaim tests

tests/code_scan/fixtures/static_signals_supabase/
  - synthetic Supabase migration, Edge Function, package/config, and CI files
```

Tier 1 should consume existing UA artifacts and conventions:

- `scripts/code-scan/domain_surfaces.py` remains the surface inventory predicate.
- `scripts/code-scan/report_data.py` remains the confidence-label/report model boundary.
- `scripts/code-scan/run_ua.py` integration happens only in the final Tier 1 bead.
- `scripts/code-scan/runtime_readiness.py` remains suggested gates only; no external execution in Tier 1.

## Layer Boundaries

| Layer | Role | Claims Allowed |
|---|---|---|
| Tier 0 | deterministic UA evidence | `deterministic_fact`, inventory, hashes, paths, manifests |
| Tier 1 | static marker inventory | `heuristic_signal`, `not_validated`, content-marker presence |
| Tier 2 | execution oracles | `executed_external_gate` only after approved commands run |
| Tier 3 | bounded LLM synthesis | `inferred_summary` with artifact citations and labels |
| Tier 4 | human decision | approval, certification, merge/deploy/security signoff |

## Highest-Value Tier 1 Inclusions

1. Static signal schema and boundary contract.
2. Supabase migration marker inventory for RLS/policy/auth SQL markers.
3. Edge Function marker inventory for auth/CORS/secret/request/external-call markers.
4. Package/config marker inventory for scripts, CI gates, public env/config surfaces.
5. Entrypoint/hotspot refinement to reduce noisy coding-agent context.
6. Final `run_ua`/report/context integration with manifest integrity and overclaim tests.

## Out of Scope for This Planning Package

This package does not approve implementation. It does not approve edits to:

- `scripts/code-scan/run_ua.py`
- `scripts/code-scan/report_data.py`
- `scripts/code-scan/render_report.py`
- any production source file
- any test file
- any dependency/config/runtime file

It also does not approve commit, push, merge, deploy, production mutation, external command execution against target repos, Supabase operations, browser automation, dependency installation, new runtime dependencies, dashboard/UI, auto-injection, SQLite/vector store, tree-sitter/WASM, or LLM/provider calls inside scanner scripts.

## Bead Index

Execute later only after separate explicit approval. JC separately approved serial execution on 2026-06-05:

```text
[JC] prepare a branch and begin serial execution of all beads using codex-coder, if bead passes all verification tests commit and push to branch before executing the next bead.
```

Current execution order:

1. `.beads/ua-tier1-001-static-signals-schema.md` — implemented and verified on this branch.
2. `.beads/ua-tier1-002-supabase-migration-markers.md`
3. `.beads/ua-tier1-003-edge-package-config-markers.md`
4. `.beads/ua-tier1-004-entrypoint-hotspot-refinement.md`
5. `.beads/ua-tier1-005-run-ua-report-integration.md`

## Recommended Execution Routing

Because this is security-adjacent evidence-boundary work, default execution should use `codex-coder` for implementation beads and `reviewer` before acceptance. `fast-coder` may be used only for a low-risk first pass on T1-001 or T1-004, never as autonomous finisher.

Hermes owns integration, verification, scope checks, and final synthesis.

## Verification Required Before Any Future Commit

At minimum after implementation beads:

```bash
cd /home/jarrad/work/hermes-agent-ua-local
python -m pytest tests/code_scan/test_static_signals.py -q
python -m pytest tests/code_scan/test_run_ua.py tests/code_scan/test_report_data.py tests/code_scan/test_render_report.py -q
python -m pytest tests/code_scan -q
python -m py_compile scripts/code-scan/static_signals.py scripts/code-scan/run_ua.py scripts/code-scan/report_data.py scripts/code-scan/render_report.py
git diff --check
```

## Reviewer Checklist

- Beads have exact required headings.
- Dependencies are serial and non-circular.
- No bead claims Tier 1 proves security, RLS correctness, auth correctness, runtime correctness, deployment readiness, or CI success.
- T1-005 is the only bead allowed to plan `run_ua.py` integration.
- Commit/push remains a separate approval gate.
