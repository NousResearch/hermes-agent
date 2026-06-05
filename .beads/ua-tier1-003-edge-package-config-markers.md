---
id: ua-tier1-003-edge-package-config-markers
title: Edge Function and Package/Config Markers
status: planned-not-approved
executor: codex-coder
parallel_safe: false
risk: medium
---

# Bead: ua-tier1-003-edge-package-config-markers - Edge Function and Package/Config Markers

## Context & Intent

Inventory Edge Function auth/CORS/secret/request markers and package/config gate markers without executing commands.

## Implementation Details

- Modify: `scripts/code-scan/static_signals.py`.
- Modify: `tests/code_scan/test_static_signals.py`.
- Optional create fixture files under `tests/code_scan/fixtures/static_signals_supabase/supabase/functions/example/`, `.github/workflows/`, and root package/config fixture paths.
- Detect Edge Function markers in `supabase/functions/*/index.ts` and `supabase/functions/*/index.js`.
- Detect package/config markers in `package.json`, `.github/workflows/*.yml`, `.github/workflows/*.yaml`, `vite.config.*`, `vercel.json`, and `netlify.toml`.
- Do not run npm, audit, CI, Supabase, Deno, browser, or external commands.

## Complexity Tier

- T2 — multiple marker families with evidence-boundary and false-positive risk.
- Expected implementation size: 200-400 LOC including tests/fixtures.
- Execution routing: coder + Hermes verification + reviewer.

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

Edge Function marker types:

```text
authorization_header
bearer_token
jwt
get_user
service_role_env
deno_env
cors_header
cors_wildcard
request_json
external_fetch
```

Package/config marker types:

```text
script_test
script_build
script_lint
script_typecheck
script_audit
ci_npm_ci
ci_test
ci_build
ci_typecheck
vite_public_env
vercel_config
netlify_config
```

Required boundary language: package/config markers identify available or declared gates only. They do not prove the gates were executed or passed.

## Dependencies

- `.beads/ua-tier1-001-static-signals-schema.md` complete.
- `.beads/ua-tier1-002-supabase-migration-markers.md` complete unless this bead is explicitly split/rebased by Hermes.

## Test Obligations

- RED: Edge/package/config fixture tests fail before marker support exists.
- GREEN: focused tests assert marker families, line numbers where applicable, and summary counts.
- Negative test: declared `npm test` produces a marker but no `executed_external_gate` claim.
- FULL: `python -m pytest tests/code_scan -q`.

## Verification Command

```bash
cd /home/jarrad/work/hermes-agent-ua-local
python -m pytest tests/code_scan/test_static_signals.py -q
python -m pytest tests/code_scan -q
python -m py_compile scripts/code-scan/static_signals.py
git diff --check
```

## Approval Evidence

- Diff artifact requirement: generate `/tmp/ua-tier1-003-edge-package-config-markers-diff.patch` with line/byte counts; include untracked new files via `git add -N` or `git diff --no-index`.
- Scope/stale-language check: search changed source/report/context files for forbidden certifier language and confirm matches are only disclaimers/tests for overclaim prevention.
- Subagent reliability requirement: coder may write implementation/tests and run local verification only; coder has no commit/push/merge/deploy authority. If timeout/no-summary occurs, Hermes must inspect actual files/diff before retrying or accepting.
- Reviewer requirement: reviewer PASS required for spec compliance, evidence-boundary preservation, and overclaim risk before acceptance.
- Commit/push gate: explicit JC approval required after evidence bundle; this bead grants no commit or push authority.
