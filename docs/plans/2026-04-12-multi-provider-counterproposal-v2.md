# Multi-Provider Counterproposal V2

> For Hermes: execute only inside `~/.hermes/worktrees/routing-lab`. Every PR from this plan must include implementation tests and targeted validation commands.

Goal: Convert Hermes multi-provider behavior from a scattered set of config knobs into a controllable routing platform: predictable in production, testable in isolation, explainable to the operator, and capable of future optimization without constant accidental model switching.

Architecture: Instead of patching fallback, smart routing, auxiliary, and delegation independently, introduce a single provider-orchestration layer with explicit policy, explicit state, and explicit audit output. The system should separate decision-making from execution, so Hermes can answer three questions at any time: why this provider, why this model, and why not the others.

Tech Stack: Python, pytest, Hermes config system, AIAgent runtime, auxiliary client, smart routing module, CLI/gateway/cron entrypoints, structured routing logs, fixture-based config testing.

---

## 1. Why the previous plan is not enough

The previous init plan is safe, but conservative. It improves coverage and semantics, yet it still treats multi-provider as a collection of local fixes.

That leaves several high-potential problems unsolved:
- routing logic remains fragmented across runtime domains
- config still encodes implementation details instead of operator intent
- provider switching remains hard to explain after the fact
- future additions like cost-based routing, reliability scoring, or role-based provider policies will keep increasing complexity
- the user still has to think in terms of low-level toggles instead of high-level behavior

This counterproposal aims higher:
- make multi-provider a first-class subsystem
- centralize decision rules
- support both simple single-provider operation and advanced orchestrated setups
- create a path for long-term optimization, not just bug fixes

---

## 2. Core idea: Provider Orchestration Layer (POL)

Create a unified layer that all routing domains consult before executing model calls.

### POL responsibilities
1. Normalize config into one runtime policy object
2. Resolve provider/model decisions per domain
3. Emit a structured explanation for every routing decision
4. Enforce hard constraints before execution
5. Support simulation and dry-run evaluation without calling providers

### POL domains
- `primary`
- `fallback`
- `auxiliary`
- `delegation`
- later optional:
  - `compression`
  - `verification`
  - `background_jobs`

### POL output contract
Every decision should produce something like:
- selected provider
- selected model
- decision source
- policy matched
- alternatives considered
- reasons rejected
- whether the choice is sticky, opportunistic, or forced

This is the key shift:
Hermes should not merely choose a model; it should produce a routing explanation object.

---

## 3. High-potential product direction

The real opportunity is not just “stop switching models.”
It is to give Hermes three operating modes under one architecture.

### Mode A: Strict single-provider
For users who want stability.
- one provider family
- no opportunistic switching
- fallback disabled or same-provider only
- auxiliary defaults to `main`
- delegation defaults to main provider unless explicitly overridden

### Mode B: Controlled multi-provider
For users who want role separation but predictable behavior.
- provider pinning per domain
- explicit role-based delegation
- explicit auxiliary policy
- limited fallback graph
- explainable routing logs

### Mode C: Optimizing orchestration
For advanced users.
- policies consider cost, latency, failure rate, and task class
- shadow evaluation can compare alternative provider choices
- routing can learn over time, but only within policy guardrails

This architecture gives Hermes a stable base for all three, instead of hardcoding behavior for one style and breaking the others.

---

## 4. Proposed architecture

### Layer 1: Intent-driven config
Replace raw routing knobs as the primary mental model with a higher-level policy structure.

Potential config direction:

```yaml
provider_policy:
  mode: strict-single-provider | controlled-multi-provider | optimizing
  primary:
    provider: copilot
    model: gpt-5.4
  constraints:
    allow_cross_provider_fallback: false
    allow_cross_provider_auxiliary: false
    allow_role_specific_delegation: true
    require_explanation: true
  domains:
    auxiliary:
      strategy: main
    fallback:
      strategy: disabled
    delegation:
      default_provider: copilot
      role_routes:
        planner: anthropic
        reviewer: anthropic
```

Important:
- existing config should keep working
- POL should normalize legacy config into this internal structure first
- migration can come later

### Layer 2: Policy normalization
Create an internal normalized object, e.g.:
- `ProviderPolicy`
- `DomainPolicy`
- `RoutingDecision`

This layer translates:
- `fallback_model`
- `fallback_providers`
- `smart_model_routing`
- `compression.summary_provider`
- `auxiliary.*`
- `delegation.provider_routes`
- `delegation.role_routes`

into a single coherent runtime policy.

### Layer 3: Decision engine
A pure function layer that answers:
- what is allowed?
- what is preferred?
- what is forbidden?
- what is forced?

Inputs:
- domain
- task metadata
- current provider/model
- prior session state
- config-derived policy
- optional live provider health snapshot

Outputs:
- `RoutingDecision`

### Layer 4: Execution adapters
Existing runtime code keeps executing requests, but it no longer invents routing behavior ad hoc.
It consumes a `RoutingDecision`.

### Layer 5: Observability and audit
Add structured routing events so each important call can be inspected later.

Potential event schema:
- timestamp
- domain
- requested provider/model
- selected provider/model
- reason
- policy mode
- fallback used?
- cross-provider jump?
- blocked alternatives

---

## 5. Strategic design principles

### Principle 1: operator intent over implementation knobs
The user should be able to express:
- “I want single-provider stability”
- “I want planner/reviewer on Anthropic only”
- “I want auxiliary to follow main”

without understanding every internal module.

### Principle 2: no silent cross-domain behavior
A config for delegation should not implicitly affect auxiliary behavior.
A fallback rule should not act like smart routing.
Each domain needs separate semantics.

### Principle 3: every provider jump must be explainable
If Hermes changes provider/model, there must be a recorded reason.

### Principle 4: safe defaults, advanced opt-in
The default should favor predictability.
Advanced orchestration should be explicit.

### Principle 5: test domains independently
Primary, fallback, auxiliary, and delegation each need isolated implementation tests before integrated tests.

---

## 6. More ambitious roadmap

## Phase 0 — Baseline capture
Same safety foundations as before.

Deliverables:
- fixture-based config snapshots
- baseline tests passing
- current routing surface inventory

## Phase 1 — Multi-provider kernel
Build the internal policy and decision abstractions first.

Deliverables:
- `ProviderPolicy` internal model
- `RoutingDecision` object
- legacy-config normalization function
- unit tests for normalization and decision outputs

High value:
- future fixes stop being scattered
- gives one place to reason about behavior

## Phase 2 — Explainability mode
Add routing explanations before changing policy much.

Deliverables:
- structured `why_this_route` object
- optional debug output in CLI/logs
- tests asserting explanations for representative decisions

High value:
- makes debugging much easier
- builds user trust

## Phase 3 — Strict single-provider profile
Implement a complete “stability-first” operating mode.

Rules:
- same provider unless explicitly forced
- auxiliary follows main unless pinned otherwise
- no cross-provider fallback unless policy allows it
- delegation stays on default provider unless role route is explicit

Deliverables:
- policy profile preset
- implementation tests across all domains
- migration-safe behavior for users who want zero drift

High value:
- directly addresses current user pain

## Phase 4 — Controlled multi-provider profile
Implement explicit, bounded orchestration.

Rules:
- provider separation is allowed, but only by domain or role
- fallback graph must be explicit
- every cross-provider decision emits an explanation

Deliverables:
- explicit domain policy tests
- route precedence tests
- integration tests for role-routed delegation

High value:
- powerful without becoming chaotic

## Phase 5 — Shadow routing / simulation harness
This is the high-upside feature.

Idea:
Hermes can simulate alternate routing decisions without executing them live, so you can answer:
- what would strict-single-provider have done?
- what would controlled-multi-provider have done?
- how many provider jumps were avoided?

Deliverables:
- dry-run routing simulator
- fixture-driven comparisons
- future dashboard/report support

High value:
- lets us evaluate policy changes safely before rollout
- very useful for PR review and regression analysis

## Phase 6 — Provider scoring and learning
Only after the system is explicit and testable.

Potential inputs:
- provider failure rate
- latency
- task class success history
- user corrections
- token/cost data

Constraint:
- optimization must stay inside policy limits
- no self-learning that silently violates strict mode

High value:
- this is where the architecture gains long-term performance advantage

---

## 7. PR strategy with more potential

Instead of starting with a narrow semantics-only PR, I recommend a better sequence.

### PR 1 — Introduce the Multi-Provider Kernel
Scope:
- add internal policy/decision data structures
- add normalization logic from current config
- no major runtime behavior change yet
- add implementation tests for normalization and route explanation

Why first:
- creates a strong foundation
- low risk if behavior stays compatible
- future PRs become smaller and cleaner

Required tests:
- legacy config normalizes correctly
- absent/empty sections normalize predictably
- explicit vs implicit domain rules stay distinguishable

### PR 2 — Add Explainability and Routing Audit
Scope:
- return structured reason objects from routing decisions
- wire debug output/logging
- no aggressive behavior changes yet

Required tests:
- decision object includes source and reason
- cross-provider jumps are marked explicitly
- blocked routes are visible

### PR 3 — Strict Single-Provider Profile
Scope:
- add a stability-first policy profile
- make auxiliary/delegation/fallback respect it

Required tests:
- no unintended cross-provider jumps
- auxiliary follows main
- delegation sticks to default unless explicitly overridden

### PR 4 — Controlled Multi-Provider Profile
Scope:
- allow explicit role/domain separation under policy
- preserve explanations

Required tests:
- domain-specific provider policies work
- role-route precedence is deterministic
- fallback graph obeys policy constraints

### PR 5 — Shadow Simulator
Scope:
- simulate policy outcomes without live execution
- compare policies from fixtures

Required tests:
- deterministic output for fixture inputs
- diff output between policies

This sequence has much more leverage than small isolated fixes.

---

## 8. Concrete test architecture

### Test layer A: pure policy tests
New test candidates:
- `tests/routing/test_provider_policy_normalization.py`
- `tests/routing/test_provider_policy_decisions.py`

Purpose:
- fast, deterministic, no provider calls

### Test layer B: domain adapter tests
New test candidates:
- `tests/agent/test_auxiliary_provider_policy.py`
- `tests/agent/test_delegation_provider_policy.py`
- `tests/run_agent/test_fallback_policy.py`

Purpose:
- ensure each runtime domain consumes `RoutingDecision` correctly

### Test layer C: config compatibility tests
New test candidates:
- `tests/hermes_cli/test_multi_provider_config_semantics.py`

Purpose:
- protect backward compatibility and absent/empty behavior

### Test layer D: integration regression tests
Existing/related:
- `tests/integration/test_kimi_bad_key_claude_leak.py`

Add more later:
- cross-provider leak prevention
- strict-mode no-jump integration path
- controlled profile explicit-jump path

### Test layer E: simulation tests
Future:
- fixture-based policy comparison tests

---

## 9. Why this has more potential

This proposal is better because it does not just reduce bugs.
It creates a platform for:
- deterministic single-provider mode
- robust multi-provider orchestration
- simulation before rollout
- routing observability
- eventual learned optimization under guardrails

In other words:
The first plan improves the current system.
This counterproposal creates the next version of the system.

---

## 10. Recommended next move

If we choose this stronger path, the best first action is not to patch runtime code yet.
It is to start PR 1 with the kernel foundation.

Recommended immediate work:
1. create tests for policy normalization
2. define internal `ProviderPolicy` and `RoutingDecision`
3. normalize current config into that structure
4. keep runtime behavior compatible at first
5. then add explainability in PR 2

That gives us a clean architecture and a strong runway for the later strict-mode and controlled-mode improvements.

---

## 11. First implementation slice

Suggested first deliverable in this lab:
- add `tests/routing/test_provider_policy_normalization.py`
- add `tests/routing/test_provider_policy_decisions.py`
- add new internal module, likely something like:
  - `agent/provider_policy.py`

Initial functions:
- `normalize_provider_policy(config) -> ProviderPolicy`
- `decide_provider_route(domain, context, policy) -> RoutingDecision`

Initial guarantee:
- no runtime behavior change unless explicitly wired in
- only build the internal decision substrate and test it thoroughly

That is the highest-potential starting point.
