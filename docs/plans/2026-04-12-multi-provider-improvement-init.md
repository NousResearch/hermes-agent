# Multi-Provider Improvement Init Plan

> For Hermes: execute this only inside the routing-lab worktree. Every accepted change must include implementation tests before PR.

Goal: Improve Hermes multi-provider behavior so provider/model selection is predictable, testable, and easier to operate without accidental model switching across fallback, auxiliary, routing, and delegation paths.

Architecture: Treat multi-provider as four separate control planes instead of one mixed concept: primary runtime, fallback runtime, auxiliary runtime, and delegation runtime. Normalize configuration semantics first, add implementation tests around current behavior, then make small code changes behind explicit rules so PRs remain safe and reviewable.

Tech Stack: Python, pytest, Hermes config loader, AIAgent, smart routing code, auxiliary client, CLI/gateway/cron routing entrypoints.

---

## What we already know

Current relevant surfaces in the codebase:
- `hermes_cli/config.py`
  - `DEFAULT_CONFIG["fallback_providers"] = []`
  - `DEFAULT_CONFIG["smart_model_routing"]`
  - `DEFAULT_CONFIG["auxiliary"]`
- `agent/auxiliary_client.py`
  - supports `summary_provider`, auxiliary task providers, and `main`
- `cron/scheduler.py`
  - reads `smart_model_routing`
  - reads `fallback_providers` or `fallback_model`
- existing tests:
  - `tests/run_agent/test_fallback_model.py`
  - `tests/agent/test_credential_pool_routing.py`
  - `tests/integration/test_kimi_bad_key_claude_leak.py`

Main risk areas:
1. fallback provider/model switching
2. smart turn routing switching models unexpectedly
3. auxiliary `provider: auto` choosing a different provider than the main runtime
4. delegation `provider_routes` / `role_routes` using a different provider family than expected
5. config semantics unclear when features are disabled, empty, or removed

## Desired outcome

A future PR sequence should produce:
- clearer configuration semantics for multi-provider behavior
- less accidental provider/model switching
- deterministic tests for current and desired behavior
- confidence that any routing/fallback change is protected by implementation tests

---

## Phase 0 — Baseline and safety

### Task 1: Freeze baseline evidence
Objective: Record the current baseline before any behavioral change.

Files:
- Use: `tests/fixtures/config.pre-remove-routing.yaml`
- Create later if needed: `docs/plans/baseline-notes.md`

Steps:
1. Keep the current fixture as the reference config snapshot.
2. Do not restore it over `~/.hermes/config.yaml`.
3. Use the fixture only in tests, parser checks, or documentation.

Verification:
- Fixture exists in this worktree.
- No write occurs to `~/.hermes/config.yaml`.

### Task 2: Define multi-provider domains explicitly
Objective: Use one vocabulary across plan, code, and PRs.

Domains:
- Primary runtime: main request execution
- Fallback runtime: post-failure provider/model activation
- Auxiliary runtime: vision, compression, search, approvals, etc.
- Delegation runtime: subagent/provider role routing

Acceptance criteria:
- Every PR states which domain(s) it changes.
- No PR should say only “fix multi-provider” without naming the affected domain.

---

## Phase 1 — Test coverage before behavior changes

### Task 3: Inventory current implementation tests
Objective: Confirm which behaviors are already covered.

Files to inspect:
- `tests/run_agent/test_fallback_model.py`
- `tests/agent/test_credential_pool_routing.py`
- `tests/integration/test_kimi_bad_key_claude_leak.py`

Expected output:
- fallback activation coverage map
- smart routing/credential-pool coverage map
- provider-leak regression coverage map

Verification command:
- `pytest tests/run_agent/test_fallback_model.py tests/agent/test_credential_pool_routing.py tests/integration/test_kimi_bad_key_claude_leak.py -q`

### Task 4: Add missing baseline tests for config semantics
Objective: Lock current semantics for absent vs empty routing sections.

Proposed new test file:
- `tests/hermes_cli/test_multi_provider_config_semantics.py`

Tests to add:
1. config loads when `fallback_providers` is absent
2. config loads when `fallback_providers` is empty list
3. config loads when `smart_model_routing` is absent
4. config loads when `smart_model_routing.enabled` is false
5. auxiliary `summary_provider=auto` remains distinguishable from `main`

Purpose:
- avoid regressions when users delete disabled config blocks
- make config migration behavior explicit

Verification command:
- `pytest tests/hermes_cli/test_multi_provider_config_semantics.py -q`

### Task 5: Add missing tests for delegation/provider-route expectations
Objective: Protect against silent provider drift in subagents.

Proposed new test file:
- `tests/agent/test_delegation_provider_routing.py`

Tests to add:
1. delegation defaults to configured provider route when role-specific route exists
2. role route overrides default route deterministically
3. removing or emptying route config falls back predictably
4. planner/reviewer routes do not silently cross providers unless explicitly configured

Verification command:
- `pytest tests/agent/test_delegation_provider_routing.py -q`

---

## Phase 2 — First improvement target (recommended)

Recommended first PR target:
- Improve config semantics and visibility without changing runtime policy much.

Why this first:
- lowest risk
- high clarity gain
- gives stable base for later behavioral changes

Scope of PR 1:
1. add tests for absent/empty routing config
2. add tests for delegation provider-route semantics
3. if needed, add tiny code/doc fix so deleted disabled sections are treated the same as explicit disabled/empty config
4. document which runtime domain each setting belongs to

This PR should not yet redesign provider selection. It should make current behavior explicit and safely testable.

---

## Phase 3 — Candidate follow-up PRs

### PR 2: Auxiliary provider hardening
Focus:
- make `auto` vs `main` behavior explicit and consistent
- reduce unexpected switching in compression/search/approval tasks

Tests required:
- auxiliary provider resolution for `auto`
- auxiliary provider resolution for `main`
- no silent cross-provider jump when main is intended

### PR 3: Fallback provider chain hardening
Focus:
- clarify whether `fallback_model` and `fallback_providers` can coexist
- normalize single vs multiple fallback behavior
- ensure fallback activates only under intended failures

Tests required:
- primary failure triggers configured fallback only
- no fallback when config absent/empty
- normalized model name/provider assertions

### PR 4: Smart routing guardrails
Focus:
- preserve useful cheap routing while preventing accidental model changes during active work
- improve sticky behavior and disabled/removed-config behavior

Tests required:
- disabled routing returns primary runtime unchanged
- short/simple prompt routing respects config
- continuation prompts stay on stronger model when session state requires it

### PR 5: Delegation provider policy cleanup
Focus:
- unify `provider_routes` and `role_routes` semantics
- reduce invisible role-based provider switching

Tests required:
- route resolution precedence
- explicit provider pinning per role
- fallback to default route when role route missing

---

## Implementation rules for this lab

1. Every PR must include implementation tests.
2. No production config edits during coding in this lab.
3. Prefer small PRs, one runtime domain at a time.
4. Do not combine fallback + auxiliary + delegation changes in the same first PR.
5. For every behavior change, write the failing test first.

---

## Recommended immediate next step

Start with PR 1:
- baseline test pass on current fallback/routing regression files
- add `tests/hermes_cli/test_multi_provider_config_semantics.py`
- add `tests/agent/test_delegation_provider_routing.py`
- run targeted pytest
- only then decide the first code patch

## Suggested test commands

Baseline:
- `pytest tests/run_agent/test_fallback_model.py tests/agent/test_credential_pool_routing.py tests/integration/test_kimi_bad_key_claude_leak.py -q`

New config semantics tests:
- `pytest tests/hermes_cli/test_multi_provider_config_semantics.py -q`

New delegation route tests:
- `pytest tests/agent/test_delegation_provider_routing.py -q`

Combined targeted gate:
- `pytest tests/run_agent/test_fallback_model.py tests/agent/test_credential_pool_routing.py tests/integration/test_kimi_bad_key_claude_leak.py tests/hermes_cli/test_multi_provider_config_semantics.py tests/agent/test_delegation_provider_routing.py -q`
