---
id: ua-tier1-002-supabase-migration-markers
title: Supabase Migration Marker Inventory
status: completed
executor: codex-coder
parallel_safe: false
risk: medium
---

# Bead: ua-tier1-002-supabase-migration-markers - Supabase Migration Marker Inventory

## Context & Intent

Detect high-value Supabase SQL migration markers as heuristic inventory, not policy evaluation.

## Implementation Details

- Modify: `scripts/code-scan/static_signals.py`.
- Modify: `tests/code_scan/test_static_signals.py`.
- Optional create fixture files under `tests/code_scan/fixtures/static_signals_supabase/supabase/migrations/`.
- Detect markers only inside paths matching `supabase/migrations/*.sql` or equivalent migration surfaces from domain-surfaces inventory.
- Emit line-numbered marker records with bounded context; cap noisy repeated markers if needed.
- Do not parse SQL AST or determine whether a policy is safe.

## Complexity Tier

- T2 — content-marker matching over security-adjacent SQL.
- Expected implementation size: 150-300 LOC including tests/fixtures.
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

Supabase migration marker types to inventory:

```text
enable_rls
create_policy
drop_policy
using_clause
with_check_clause
auth_uid
auth_role
anon_role
authenticated_role
permissive_true
security_definer
service_role
grant_statement
revoke_statement
create_function
```

Example signal:

```json
{
  "surface": "supabase_migration",
  "path": "supabase/migrations/001_rls.sql",
  "line": 12,
  "marker_type": "create_policy",
  "marker": "CREATE POLICY",
  "claim_type": "heuristic_signal",
  "semantic_status": "not_validated",
  "boundary": "marker presence only; policy semantics not validated"
}
```

## Dependencies

- `.beads/ua-tier1-001-static-signals-schema.md` complete and accepted.

## Test Obligations

- RED: synthetic migration fixture contains representative markers; tests fail before matcher exists.
- GREEN: focused static-signal tests pass and assert marker type counts, file paths, line numbers, and boundary text.
- Negative test: benign SQL without target markers emits no security/RLS correctness claim.
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

- Diff artifact requirement: generate `/tmp/ua-tier1-002-supabase-migration-markers-diff.patch` with line/byte counts; include untracked new files via `git add -N` or `git diff --no-index`.
- Scope/stale-language check: search changed source/report/context files for forbidden certifier language and confirm matches are only disclaimers/tests for overclaim prevention.
- Subagent reliability requirement: coder may write implementation/tests and run local verification only; coder has no commit/push/merge/deploy authority. If timeout/no-summary occurs, Hermes must inspect actual files/diff before retrying or accepting.
- Reviewer requirement: reviewer PASS required for spec compliance, evidence-boundary preservation, and overclaim risk before acceptance.
- Commit/push gate: explicit JC approval required after evidence bundle; this bead grants no commit or push authority.
