# Delegation Model Profiles — Implementation Plan

> **For Hermes:** implement task-by-task, TDD-first, one conventional commit per task.

**Goal:** Operator-defined worker profiles (`delegation.profiles`) that pin provider/model/effort/fallback per named tier, selectable per task — by config today (Phase 1), by the parent model through `delegate_task.model_profile` behind a default-off gate (Phase 2, `delegation.agent_routing`).

**Architecture:** One resolver module (`agent/delegation_model_routing.py`) parses/validates profiles and returns an immutable route; `delegate_task` and `SubagentLifecycleService` both call it. Precedence: per-task profile > top-level profile > `delegation.default_profile` > legacy `delegation.provider/model` > parent inherit. Route provenance (`requested_profile`, `effective_model`, ...) rides progress events and result entries.

**Repo facts (verified on main 13e72fb205, 2026-09-04):**
- DEFAULT_CONFIG `delegation` section: `hermes_cli/config_defaults.py:1205`
- Credential resolution: `tools/delegate_tool_config.py` — `_resolve_delegation_credentials` (:360, three branches: base_url / provider / inherit), `_resolve_child_runtime` (:411, fallback-chain inheritance at :490, pinned-provider fallback clearing already exists)
- Tool schema: `tools/delegate_tool.py:503-526` (`_build_tasks_param_description`, tasks properties); handler entry :603
- Batch spawn loop: `tools/delegate_tool_dispatch.py::_run_batch`; single-task normalization `tools/delegate_tool_tasks.py:89`
- Lifecycle bypass: `agent/subagent_lifecycle.py:53` (`model: Optional[str]`, unvalidated, :211 coercion table, :263 construction)
- Result entries: `tools/delegate_tool_child_run.py:296,495` (`"model"` only today)
- Capability metadata: `agent/models_dev.py::get_model_capabilities` (supports_tools :642)
- Model validation: `hermes_cli/models.py::validate_requested_model` (:4963)
- Provider resolution: `hermes_cli/runtime_provider.py::resolve_runtime_provider` (:1665)
- Reasoning parse: `hermes_constants.parse_reasoning_effort` (clamp-at-transport doctrine, NS-696)
- House rules: no model-facing raw model IDs (NS-696 maintainer policy); config-gated default-off; #25752 layering (per-call can never widen past config); no new HERMES_* env vars; docs in website/docs/user-guide/features/delegation... (check actual page), reference/cli if config check verb touched.

## Config shape

```yaml
delegation:
  # existing keys unchanged (provider, model, base_url, ...)
  default_profile: ""            # name from profiles; "" = legacy behavior
  agent_routing: false           # Phase 2 gate: expose model_profile enum on delegate_task
  profiles:
    small:
      provider: anthropic
      model: claude-haiku-current
      reasoning_effort: none     # optional; parse_reasoning_effort grammar
      max_iterations: 20         # optional; clamped by delegation.max_iterations semantics
      fallback: []               # [] = no model promotion (default); list of {provider, model}
```

## Resolution contract (route = immutable dataclass)

1. per-task `model_profile` (Phase 2 arg or lifecycle field)
2. top-level `model_profile` (batch-wide)
3. `delegation.default_profile`
4. legacy `delegation.provider`/`model` (existing `_resolve_delegation_credentials` path, byte-identical)
5. parent inherit (existing)

Rules:
- Unknown profile name → ValueError before child construction (actionable message: configured names listed).
- Profile resolves via `resolve_runtime_provider(requested=<provider>, target_model=<model>)` — same path as legacy. No duplicated credential logic.
- supports_tools=False + nonempty child toolset → ValueError (text_only future work, rejected scope).
- Profile selected → parent fallback chain NOT inherited (reuse the existing pinned-provider mechanism at delegate_tool_config.py:490: profile sets override_provider semantics). Profile `fallback` list becomes the child's chain.
- #25752: profiles can't widen toolsets — they don't touch toolsets at all (explicitly out of scope).
- Telemetry: requested_profile / resolved_provider / resolved_model / fallback_policy on result entries + progress events + lifecycle handle.

## Tasks

### T1: resolver module + contract tests
Create `agent/delegation_model_routing.py`: `ProfileRoute` frozen dataclass, `parse_profiles(cfg) -> Dict[str, ProfileSpec]`, `resolve_profile_route(name, cfg, parent_agent) -> ProfileRoute`, `select_profile_name(task_profile, top_profile, cfg) -> Optional[str]`.
Tests `tests/agent/test_delegation_model_routing.py`: parse (unknown keys rejected, malformed fallback rejected, empty profiles {}), precedence (all 5 levels), unknown-name ValueError message contains configured names, reasoning_effort parse (invalid warns+ignores per NS-696 clamp doctrine), frozen route immutability. RED first.

### T2: config defaults + validation
`hermes_cli/config_defaults.py`: add `default_profile`, `agent_routing`, `profiles` keys with house-style comments under delegation (:1205 area).
`hermes config check` lane: find existing delegation validation (grep `config check` handlers) and add profile validation (names, shapes, default_profile exists in profiles). Tests.

### T3: delegate_task wiring (Phase 1: config-driven)
`tools/delegate_tool_config.py`: `_resolve_delegation_credentials` grows a profile branch ABOVE the legacy branches — when a profile route is selected, build the credential bundle from the route (base creds via resolve_runtime_provider), set override_provider semantics so `_resolve_child_runtime` clears fallback chain; profile.fallback list feeds `fallback_model` chain when nonempty.
Per-task selection: `_run_batch`/`_run_single_child` resolve route per task (task dict carries `model_profile` internal field), NOT once per batch. `_strip_model_hidden_task_fields` (delegate_tool.py:603) must NOT strip it when gate on; must strip when gate off (schema honesty).
Tests: batch with two tasks/two profiles resolves two different models; default_profile applies when task silent; legacy config byte-identical when no profiles configured (existing tests must stay green unmodified — they are the contract).

### T4: Phase 2 gate — model_profile on the public schema
`tools/delegate_tool.py` schema build: when `delegation.agent_routing` is truthy AND profiles configured+runnable, add `model_profile` enum (configured names) to task properties + top-level; description per the doc ("small for bounded work; do not select a more expensive profile without a task-specific reason"). Gate off → schema byte-identical to today (test asserts absence).
Handler: reject model_profile args when gate off (defense in depth, tool_error), resolve when on.
Tests: schema presence/absence by gate; enum content matches configured profiles; unknown profile arg → clean tool error before AIAgent construction (mock spawn, assert not called).

### T5: lifecycle unification
`agent/subagent_lifecycle.py`: add `model_profile: Optional[str]` to SubagentLaunchRequest; when set, resolve through the SAME resolver (route wins over raw `model`); raw `model` field retained for plugin trust but routed through validate-requested-model when profiles configured... — decision: keep raw model as-is (documented internal trust surface), add model_profile as the policy path; docstring states the trust split. Validator rejects both-set.
Tests: request with model_profile resolves identically to delegate_task with same profile (parity test); both-set rejected.

### T6: telemetry
`tools/delegate_tool_child_run.py` result entries (:296,:495): add requested_profile, resolved_provider, resolved_model, fallback_policy. Progress events (`delegate_tool_progress.py::DelegateEvent`) grow the same. Lifecycle handle exposes them. When fallback fires mid-run, preserve requested vs effective (effective = child.model at completion — already captured).
Tests: result entry contains fields for profile runs; absent (or None) for legacy runs — assert exact shape.

### T7: docs
Delegation feature page under website/docs (locate: grep "max_concurrent_children" website/docs) — profiles section, YAML example, precedence list, agent_routing gate + policy rationale (operator menu, model picks names not models), fallback isolation semantics. Tool description update = T4. `hermes config check` docs if verb touched.

### T8: E2E + full gates
Real-import E2E against temp HERMES_HOME (house rule for resolution chains): config file with profiles → delegate_task spawn path resolves route (mock only the network/AIAgent run, real config+resolver+credential imports). Full `tests/tools/ tests/agent/ tests/hermes_cli/` relevant dirs + scripts/run_tests.sh if feasible.

## Rejected scope (PR body material)
- No free-form model field on the public tool (NS-696 policy; profiles ARE the boundary).
- No NLP mapping of "use haiku" prompt text (parent converts intent to args).
- No automatic routing optimizer; telemetry first.
- No catalog-driven profile mutation (catalog is discovery, not policy).
- No text_only tool-less profile mode (future; supports_tools=False rejected loudly).
- Profiles never touch toolsets (#25752 stays untouched).

## Verify
- Existing delegation tests green UNMODIFIED (legacy contract).
- New suites RED→GREEN per task; mutation checks on: precedence order, gate-off schema absence, fallback isolation.
- `python3 -c "from hermes_cli.config_defaults import DEFAULT_CONFIG; print(DEFAULT_CONFIG['delegation']['agent_routing'])"` → False.
- Salt adversarial review before PR; PR body: phases, rejected scope, policy reconciliation (NS-696), Coatue pilot framing.
