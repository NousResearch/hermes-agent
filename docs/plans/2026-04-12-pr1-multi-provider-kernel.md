# PR1 — Multi-Provider Kernel Implementation Plan

> For Hermes: Use strict TDD. Do all work in `~/.hermes/worktrees/routing-lab`. No production config edits. Every code step must begin with a failing test.

Goal: Introduce the first internal Multi-Provider Kernel for Hermes by adding a normalized provider-policy model and a decision object that can describe route choices without yet forcing large runtime behavior changes.

Architecture: PR1 creates a pure policy layer that sits above existing routing behavior. It translates current config shapes (`fallback_providers`, `fallback_model`, `smart_model_routing`, `compression.summary_provider`, `auxiliary.*`, `delegation`) into one internal policy object, and produces structured route-decision outputs that future PRs can wire into runtime domains.

Tech Stack: Python, pytest, dataclasses, Hermes config structures, smart routing module, auxiliary/runtime config semantics, fixture-based tests.

---

## Scope of PR1

In scope:
- Add a new internal policy module for multi-provider normalization and decision explanation
- Add advanced implementation tests for normalization and decision behavior
- Keep runtime behavior mostly compatible
- Add documentation for future PR sequence

Out of scope:
- Large rewiring of `run_agent.py`
- Full delegation runtime changes
- Full auxiliary runtime adoption
- Learned routing
- Live provider scoring

PR1 is a foundation PR.

---

## Files to create

### New production files
- Create: `agent/provider_policy.py`

### New test files
- Create: `tests/routing/test_provider_policy_normalization.py`
- Create: `tests/routing/test_provider_policy_decisions.py`

### New docs/plan files
- Create: `docs/plans/2026-04-12-pr1-multi-provider-kernel.md`

---

## Files to read/reference during implementation

- Read: `hermes_cli/config.py`
- Read: `agent/smart_model_routing.py`
- Read: `agent/auxiliary_client.py`
- Read: `tests/run_agent/test_fallback_model.py`
- Read: `tests/agent/test_credential_pool_routing.py`
- Read: `tests/integration/test_kimi_bad_key_claude_leak.py`

---

## Design for PR1

### New internal abstractions in `agent/provider_policy.py`

Use dataclasses.

#### `DomainPolicy`
Fields:
- `name: str`
- `strategy: str`
- `provider: str | None`
- `model: str | None`
- `enabled: bool`
- `allow_cross_provider: bool`
- `metadata: dict[str, Any]`

#### `ProviderPolicy`
Fields:
- `mode: str`
- `primary_provider: str | None`
- `primary_model: str | None`
- `fallback_providers: list[dict[str, Any]]`
- `domains: dict[str, DomainPolicy]`
- `constraints: dict[str, Any]`
- `source_summary: dict[str, Any]`

#### `RoutingDecision`
Fields:
- `domain: str`
- `selected_provider: str | None`
- `selected_model: str | None`
- `decision_source: str`
- `reason: str`
- `cross_provider: bool`
- `blocked_alternatives: list[str]`
- `metadata: dict[str, Any]`

### Core functions

#### `normalize_provider_policy(config: dict) -> ProviderPolicy`
Responsibilities:
- Accept current Hermes config dict
- Normalize current legacy fields into a structured internal object
- Never do provider calls
- Never read credentials
- Never mutate config

Must interpret at minimum:
- `model.provider`
- `model.default`
- `fallback_model`
- `fallback_providers`
- `smart_model_routing`
- `compression.summary_provider`
- `compression.summary_model`
- `auxiliary.*`
- `delegation.provider`
- `delegation.model`
- `delegation.provider_routes`
- `delegation.role_routes`

#### `decide_provider_route(domain: str, policy: ProviderPolicy, *, requested_provider=None, requested_model=None, context=None) -> RoutingDecision`
Responsibilities:
- Produce a structured decision for a domain
- Respect explicit request overrides when allowed
- Mark if the chosen provider crosses away from `primary_provider`
- Record why a route was chosen
- Return deterministic output for tests

Initial PR1 behavior can be simple:
- `primary` → pick normalized primary
- `fallback` → describe configured fallback policy only
- `auxiliary` → honor `main`, `auto`, or explicit provider semantics at a policy level
- `delegation` → describe default delegation provider/model at policy level

No need yet to fully implement role-route resolution logic in runtime. Only policy-level decision behavior is needed.

---

## Advanced implementation test strategy

PR1 must include more than trivial tests. It needs strong, implementation-oriented coverage.

### Test file 1: `tests/routing/test_provider_policy_normalization.py`

Purpose: prove that legacy Hermes config shapes normalize into a stable internal policy.

Test cases to implement:

1. `test_normalizes_minimal_primary_config`
- config contains only `model.provider` and `model.default`
- assert primary provider/model normalize correctly
- assert domains exist for `primary`, `fallback`, `auxiliary`, `delegation`

2. `test_normalizes_absent_fallback_sections_as_disabled`
- config has no `fallback_model` and no `fallback_providers`
- assert fallback domain becomes disabled/empty but valid

3. `test_normalizes_single_legacy_fallback_model`
- config uses `fallback_model`
- assert normalized `fallback_providers` contains one entry
- assert source summary notes legacy single-fallback origin

4. `test_normalizes_multiple_fallback_providers`
- config uses `fallback_providers`
- assert ordering is preserved
- assert normalized fallback list is stable

5. `test_normalizes_missing_smart_model_routing_as_disabled`
- config omits smart routing entirely
- assert smart-routing-related metadata defaults correctly

6. `test_normalizes_explicit_disabled_smart_model_routing`
- config includes `smart_model_routing.enabled = False`
- assert normalization distinguishes explicit disabled from absent only in metadata if desired

7. `test_normalizes_compression_summary_provider_main`
- config uses `compression.summary_provider = main`
- assert auxiliary/compression domain records strategy `main`

8. `test_normalizes_compression_summary_provider_auto`
- config uses `compression.summary_provider = auto`
- assert auxiliary/compression domain records strategy `auto`

9. `test_normalizes_auxiliary_task_specific_provider`
- config sets `auxiliary.vision.provider = openrouter`
- assert domain metadata includes task-specific explicit provider

10. `test_normalizes_delegation_default_and_routes`
- config contains `delegation.provider`, `delegation.model`, `provider_routes`, `role_routes`
- assert delegation domain metadata preserves these maps exactly

11. `test_normalization_does_not_mutate_input_config`
- deep-copy input before call
- assert input remains unchanged after normalization

12. `test_normalization_handles_deleted_disabled_sections`
- build config with no `fallback_providers` and no `smart_model_routing`
- assert normalize still returns valid policy
- this protects the exact style the user now prefers

### Test file 2: `tests/routing/test_provider_policy_decisions.py`

Purpose: prove that policy decisions are deterministic and explainable.

Test cases to implement:

1. `test_primary_decision_returns_primary_provider_and_model`
- normalized from a simple config
- assert selected provider/model match primary
- assert decision_source indicates primary config

2. `test_auxiliary_main_strategy_follows_primary`
- policy from config where compression summary is `main`
- call `decide_provider_route(domain="auxiliary", context={"task": "compression"})`
- assert selected provider/model follow primary unless explicit override exists

3. `test_auxiliary_auto_strategy_marks_reason_as_auto`
- config with `provider=auto`
- assert decision reason/source reflect auto strategy, not silent hard pinning

4. `test_auxiliary_explicit_provider_marks_cross_provider_true`
- primary provider = `copilot`
- auxiliary vision provider = `openrouter`
- assert `cross_provider is True`
- assert reason mentions explicit task override

5. `test_fallback_decision_reports_first_available_fallback_candidate`
- config with multiple fallback providers
- assert selected provider/model are first configured candidate in PR1 policy logic
- assert decision_source notes fallback chain

6. `test_fallback_decision_reports_disabled_when_not_configured`
- no fallback configured
- assert decision reason/source say disabled or unconfigured

7. `test_delegation_decision_uses_explicit_delegation_provider`
- config with delegation provider/model
- assert decision returns them

8. `test_delegation_decision_falls_back_to_primary_when_empty`
- config with empty delegation provider/model
- assert decision uses primary route
- assert decision_source indicates inheritance

9. `test_requested_provider_override_is_reflected_in_decision_metadata`
- call `decide_provider_route(... requested_provider="anthropic")`
- if PR1 allows requested override, assert metadata and reason reflect it
- if blocked by policy, assert override appears in blocked alternatives

10. `test_decision_object_is_stable_and_serializable`
- convert decision to dict using `dataclasses.asdict`
- assert expected keys exist
- this is preparation for later logging/telemetry

11. `test_cross_provider_false_when_domain_matches_primary`
- exact same provider in domain and primary
- assert `cross_provider is False`

12. `test_blocked_alternatives_defaults_to_empty_list`
- sanity/stability test for future expansion

---

## Order of work (strict TDD)

### Task 1: Create PR notes file

**Objective:** Document exact PR1 intent before code.

**Files:**
- Create: `docs/plans/2026-04-12-pr1-multi-provider-kernel.md`

**Step 1: Write the doc**
Include:
- purpose of PR1
- in-scope/out-of-scope
- exact test commands
- expected future PR sequence

**Step 2: Verify file exists**
Run:
- `test -f docs/plans/2026-04-12-pr1-multi-provider-kernel.md && echo OK`
Expected: `OK`

### Task 2: Write first failing normalization tests

**Objective:** Define the policy module contract before implementation.

**Files:**
- Create: `tests/routing/test_provider_policy_normalization.py`
- Create later: `agent/provider_policy.py`

**Step 1: Write only these initial failing tests first**
- `test_normalizes_minimal_primary_config`
- `test_normalizes_absent_fallback_sections_as_disabled`
- `test_normalization_handles_deleted_disabled_sections`

**Step 2: Run tests to verify RED**
Run:
- `pytest tests/routing/test_provider_policy_normalization.py -q`
Expected: FAIL with import/module errors or missing symbol failures.

### Task 3: Add minimal production skeleton

**Objective:** Create the smallest module needed for tests to import and start failing meaningfully.

**Files:**
- Create: `agent/provider_policy.py`

**Step 1: Add minimal dataclasses and stubs**
Add:
- `DomainPolicy`
- `ProviderPolicy`
- `RoutingDecision`
- `normalize_provider_policy`
- `decide_provider_route`

Return placeholder values only sufficient to progress.

**Step 2: Run tests again**
Run:
- `pytest tests/routing/test_provider_policy_normalization.py -q`
Expected: FAIL on assertions, not imports.

### Task 4: Make initial normalization tests pass

**Objective:** Implement minimal normalization for primary, fallback-disabled, and deleted-section handling.

**Files:**
- Modify: `agent/provider_policy.py`

**Step 1: Implement primary extraction**
Read from config:
- `model.provider`
- `model.default`

**Step 2: Implement fallback-disabled defaults**
If no fallback fields exist, produce stable fallback domain.

**Step 3: Implement missing/deleted section handling**
If `smart_model_routing` absent and `fallback_providers` absent, normalize cleanly.

**Step 4: Run tests**
Run:
- `pytest tests/routing/test_provider_policy_normalization.py -q`
Expected: first small set PASS.

### Task 5: Expand normalization test coverage

**Objective:** Add the rest of the normalization tests.

**Files:**
- Modify: `tests/routing/test_provider_policy_normalization.py`
- Modify: `agent/provider_policy.py`

**Step 1: Add remaining normalization tests**
Add cases for:
- legacy single fallback
- multiple fallback providers
- explicit disabled smart routing
- compression main/auto
- task-specific auxiliary provider
- delegation maps
- input immutability

**Step 2: Run RED**
Run:
- `pytest tests/routing/test_provider_policy_normalization.py -q`
Expected: FAIL on new assertions.

**Step 3: Implement minimal code to pass**
Update normalization function only as far as needed.

**Step 4: Run GREEN**
Run:
- `pytest tests/routing/test_provider_policy_normalization.py -q`
Expected: PASS.

### Task 6: Write first failing decision tests

**Objective:** Define decision semantics before implementation.

**Files:**
- Create: `tests/routing/test_provider_policy_decisions.py`
- Modify later: `agent/provider_policy.py`

**Step 1: Write initial failing tests**
Start with:
- `test_primary_decision_returns_primary_provider_and_model`
- `test_auxiliary_main_strategy_follows_primary`
- `test_delegation_decision_falls_back_to_primary_when_empty`

**Step 2: Run RED**
Run:
- `pytest tests/routing/test_provider_policy_decisions.py -q`
Expected: FAIL on missing behavior.

### Task 7: Implement minimal decision engine

**Objective:** Make the initial decision tests pass.

**Files:**
- Modify: `agent/provider_policy.py`

**Step 1: Implement simple domain decision rules**
- `primary` uses normalized primary
- `auxiliary` with `main` follows primary
- `delegation` empty inherits primary

**Step 2: Run GREEN**
Run:
- `pytest tests/routing/test_provider_policy_decisions.py -q`
Expected: initial tests PASS.

### Task 8: Expand decision coverage

**Objective:** Add all advanced decision tests and complete the policy behavior for PR1.

**Files:**
- Modify: `tests/routing/test_provider_policy_decisions.py`
- Modify: `agent/provider_policy.py`

**Step 1: Add remaining tests**
Include:
- auxiliary auto
- auxiliary explicit provider cross-provider
- fallback first-candidate behavior
- fallback disabled behavior
- explicit delegation provider
- requested override metadata
- stable serializable decision
- blocked alternatives default

**Step 2: Run RED**
Run:
- `pytest tests/routing/test_provider_policy_decisions.py -q`
Expected: FAIL on newly added cases.

**Step 3: Implement the minimal logic to satisfy PR1 semantics**
Do not overbuild future features.

**Step 4: Run GREEN**
Run:
- `pytest tests/routing/test_provider_policy_decisions.py -q`
Expected: PASS.

### Task 9: Run combined targeted suite

**Objective:** Ensure PR1 foundation does not break related routing tests.

**Files:**
- No code changes unless failures found

**Step 1: Run combined targeted tests**
Run:
- `pytest tests/routing/test_provider_policy_normalization.py tests/routing/test_provider_policy_decisions.py tests/run_agent/test_fallback_model.py tests/agent/test_credential_pool_routing.py tests/integration/test_kimi_bad_key_claude_leak.py -q`

Expected:
- all targeted tests PASS

### Task 10: Refactor lightly and document

**Objective:** Clean up code after GREEN.

**Files:**
- Modify: `agent/provider_policy.py`
- Modify: `docs/plans/2026-04-12-pr1-multi-provider-kernel.md`

**Step 1: Refactor naming and helpers only if tests stay green**
Possible helpers:
- `_normalize_fallbacks`
- `_normalize_auxiliary_domain`
- `_normalize_delegation_domain`
- `_domain_policy`

**Step 2: Re-run targeted suite**
Run same command from Task 9.

### Task 11: Commit PR1 foundation

**Objective:** Make the branch reviewable.

**Files:**
- Stage only files relevant to PR1

**Step 1: Commit**
Run:
- `git add agent/provider_policy.py tests/routing/test_provider_policy_normalization.py tests/routing/test_provider_policy_decisions.py docs/plans/2026-04-12-pr1-multi-provider-kernel.md`
- `git commit -m "feat: add multi-provider policy kernel"`

---

## Exact code structure guidance for `agent/provider_policy.py`

Suggested outline:

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainPolicy:
    name: str
    strategy: str
    provider: str | None = None
    model: str | None = None
    enabled: bool = True
    allow_cross_provider: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderPolicy:
    mode: str
    primary_provider: str | None
    primary_model: str | None
    fallback_providers: list[dict[str, Any]] = field(default_factory=list)
    domains: dict[str, DomainPolicy] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    source_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    domain: str
    selected_provider: str | None
    selected_model: str | None
    decision_source: str
    reason: str
    cross_provider: bool = False
    blocked_alternatives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Keep PR1 pure and deterministic.

---

## Test commands

### RED/GREEN during development
- `pytest tests/routing/test_provider_policy_normalization.py -q`
- `pytest tests/routing/test_provider_policy_decisions.py -q`

### Targeted integration gate for PR1
- `pytest tests/routing/test_provider_policy_normalization.py tests/routing/test_provider_policy_decisions.py tests/run_agent/test_fallback_model.py tests/agent/test_credential_pool_routing.py tests/integration/test_kimi_bad_key_claude_leak.py -q`

### Optional broader confidence run
- `pytest tests/routing tests/agent/test_credential_pool_routing.py tests/run_agent/test_fallback_model.py -q`

---

## Future PR sequence to include in the review summary

After PR1, the expected follow-up PRs are:

### PR2 — Routing Explainability Layer
- add structured reason/explanation emission
- expose why this provider/model was selected
- prepare future logs/telemetry

### PR3 — Strict Single-Provider Profile
- make stability-first mode explicit
- auxiliary follows main
- delegation defaults to main unless explicitly overridden
- no accidental cross-provider drift

### PR4 — Controlled Multi-Provider Profile
- explicit domain-separated provider policies
- explicit role-based delegation
- bounded fallback behavior

### PR5 — Shadow Routing Simulator
- dry-run policy comparison
- evaluate route choices without live provider execution
- useful for future dashboard/reporting

### PR6 — Provider Health/Scoring Substrate
- latency/failure/cost signals
- still constrained by explicit policy, never silent autonomous drift

---

## About Hermes learning flow

Yes — this work should fit into the Hermes learning flow you configured, but in the correct way.

How it should integrate:
1. Work happens in the lab worktree first
2. PRs encode the stable implementation knowledge
3. Passing implementation tests become the durable proof layer
4. Once a PR is validated, the architectural lessons can inform:
   - skills
   - learned routing notes
   - future policy design
   - Obsidian/learning sync summaries if you want

What should NOT happen:
- the runtime should not “self-learn” new multi-provider behavior directly into production config without review
- learned adjustments should not bypass explicit policy or tests

Recommended learning-flow rule:
- architecture ideas → plan/docs in worktree
- accepted behavior → code + tests + PR
- reusable operational lesson → skill or memory note
- production behavior change → only after PR review/merge

So yes: this absolutely can proceed within your Hermes learning architecture, but as reviewed, test-backed learning — not silent runtime drift.

---

## Final acceptance criteria for PR1

PR1 is complete only if:
- `agent/provider_policy.py` exists
- normalization tests pass
- decision tests pass
- existing targeted fallback/routing leak tests still pass
- no production config was touched
- review summary includes the future PR roadmap
- review summary explicitly states how this work fits the Hermes learning flow
