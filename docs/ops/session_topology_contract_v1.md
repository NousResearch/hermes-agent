# Session Topology Contract + Deterministic Route-Policy Router v1

Date: 2026-03-20  
Status: active (contract + deterministic router)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose
Define explicit **route-policy topology** rules so model route selection is reproducible, auditable, and fail-closed.

This contract decouples:
1. **Route policy** (`session_kind` + `task_class` + `risk_tier` -> `route_class` + rollout stage)
2. **Qualification policy** (which model keys are permitted for that class/stage)

Scope boundary:
- This file governs **route-policy model selection** only.
- Transport/topic routing is governed by `docs/ops/session_topology_transport_contract_v1.md`.
- Authority/lease/fencing semantics are governed by `docs/ops/lane_topology_authority_contract_v1.md`.
- Orchestrator replay/idempotency/resync outer-contract semantics are governed by `docs/ops/orchestrator_api_contract_v1.md`.
- Family pointer map: `docs/ops/SESSION_TOPOLOGY.md`.

## Artifacts
- Topology schema: `docs/ops/schemas/session_topology_contract.schema.json`
- Topology template: `docs/ops/templates/session_topology_contract.template.json`
- Proposal packet schema: `docs/ops/schemas/proposal_packet.v1.schema.json`
- Proposal packet template: `docs/ops/templates/proposal_packet.v1.template.json`
- Proposal archive packet schema: `docs/ops/schemas/proposal_archive_packet.v1.schema.json`
- Proposal archive packet template: `docs/ops/templates/proposal_archive_packet.v1.template.json`
- Delta spec schema: `docs/ops/schemas/delta_spec.v1.schema.json`
- Delta spec template: `docs/ops/templates/delta_spec.v1.template.json`
- Regression-risk packet schema (active): `docs/ops/schemas/regression_risk_packet.v2.schema.json`
- Regression-risk packet template (active): `docs/ops/templates/regression_risk_packet.v2.template.json`
- Regression-risk packet schema (legacy-compatible): `docs/ops/schemas/regression_risk_packet.v1.schema.json`
- Regression-risk packet template (legacy-compatible): `docs/ops/templates/regression_risk_packet.v1.template.json`
- Test-gap packet schema (pilot): `docs/ops/schemas/test_gap_packet.v1.schema.json`
- Test-gap packet template (pilot): `docs/ops/templates/test_gap_packet.v1.template.json`
- Refactor-risk packet schema: `docs/ops/schemas/refactor_risk_packet.v1.schema.json`
- Refactor-risk packet template: `docs/ops/templates/refactor_risk_packet.v1.template.json`
- Code-health + rule-governance packet schema: `docs/ops/schemas/code_health_rule_governance_packet.v1.schema.json`
- Code-health + rule-governance packet template: `docs/ops/templates/code_health_rule_governance_packet.v1.template.json`
- Route-policy router CLI: `scripts/session_topology_router.py`

## Inputs
Router consumes:
1. topology contract JSON (`clawd.session_topology_contract.v1`)
2. routing request (`session_kind`, `task_class`, `risk_tier`, optional `fold_in_target`, optional `support_only`, optional `escalation_evidence`)
3. one or more model-rollout gate decisions (`clawd.model_rollout_gate.decision.v1`)
4. optional transport decision (`clawd.session_topology_transport_routing.decision.v1`) for end-to-end conformance lock
5. optional `workflow_dag` orchestration plan for XE-303 bounded canary (`nodes` + `edges`)

### Proposal-first delta-spec ingress (COD-05 / PR-07)
Router now supports proposal-first coding ingress with bounded approval hooks:
- accepted ingress forms:
  - direct `proposal_packet.v1` payload (router derives `session_kind=worker_slice`, `task_class`, `risk_tier`)
  - envelope request with nested `proposal_packet` (+ optional `proposal_approval`, `delta_spec`, `proposal_phase`, `proposal_archive`)
- proposal-first gate: `proposal_first_delta_spec`
  - validates proposal packet contract alignment (`task_class`, `risk_assessment.initial_tier`)
  - projects risk-tier approval hooks (`low/medium/high/critical`)
  - validates optional delta-spec surface (`delta_spec.instructions` or `delta_spec.prompt`)
  - enforces bounded proposal/apply/archive state flow:
    - `proposal_phase` allowed values: `proposal | apply | archive`
    - `apply` / `archive` phase requires explicit `delta_spec`
    - `archive` phase requires `proposal_archive` packet (`schema_version=proposal_archive_packet.v1`) aligned to `docs/ops/schemas/proposal_archive_packet.v1.schema.json` (task-id alignment + required artifact inventory)
  - emits additive `proposal_flow.state_flow` packet (`declared_phase`, `effective_phase`, `proposal_ready`, `apply_ready`, `archive_ready`)
- strict rollout toggle: `--require-proposal-first-coding`
  - when enabled, coding task classes (`implementation`, `code:*`) fail closed without `proposal_packet`

Approval hook posture by risk tier:
- `low`: automatic proposal approval
- `medium`: optional operator review
- `high`: mandatory operator approval (minimum 1 approver id)
- `critical`: mandatory multi-operator approval (minimum 2 approver ids)

Archive contract projection (emitted in `proposal_flow.archive_contract.required_artifacts`):
- `proposal_packet`
- `approval_records`
- `delta_spec`
- `code_delta`
- `validation_results`
- `routing_audit`

### Regression-risk packet governance ingress (COD-06 / LT-05)
Router now supports bounded regression-risk packet integration for coding governance and route hardening:
- accepted ingress form:
  - envelope request field `regression_risk_packet`
- regression-risk gate: `regression_risk_packet`
  - validates packet contract (`packet_id`, `version=1.0|2.0`, `risk_assessment`, `evidence`, `validation`)
  - v2 contract adds replay-evidence + classification checks (`replay_evidence`, `blocking_classification`) with deterministic classification parity (`blocking|non_blocking`)
  - enforces score-dimension completeness (`blast_radius`, `code_churn`, `dependency_impact`, `test_coverage_delta`, `historical_instability`, `critical_path_impact`)
  - enforces tier->required-approval alignment:
    - `low`: 0
    - `medium`: 1
    - `high`: 2
    - `critical`: 3
  - projects `request.effective_risk_tier` from max(request risk tier, packet overall tier) so downstream routing/model gates are governed by objective post-implementation risk
- strict rollout toggle: `--require-regression-risk-packet-for-coding`
  - when enabled, coding task classes (`implementation`, `code:*`) fail closed without `regression_risk_packet`
  - strict mode also requires `validation.status=approved` before routing pass

Additive telemetry projection:
- top-level `regression_risk`
- mirrored `request.regression_risk`, `route.regression_risk`, and `routing_audit.regression_risk`

### Test-gap packet pilot artifacts (LT-05 / B9Q-005)
Support-only pilot packet contract is now materialized for validator-ready governance without widening runtime route gates yet:
- schema: `docs/ops/schemas/test_gap_packet.v1.schema.json`
- template: `docs/ops/templates/test_gap_packet.v1.template.json`

Pilot contract focus:
- changed-path test-gap findings with explicit code-to-test mapping basis,
- required confidence + uncertainty disclosures,
- release impact classification (`blocking` vs `non_blocking`) with deterministic packet-level parity,
- required `analysis_scope.unprocessed_scope` disclosure so unanalyzed areas are explicit.

### Refactor-risk + decomposition governance ingress (COD-06 / LT-06)
Router now supports bounded refactor-risk packet integration with decomposition governance for coding tasks:
- accepted ingress form:
  - envelope request field `refactor_risk_packet`
- decomposition/risk gate: `refactor_risk_decomposition`
  - validates packet contract (`packet_id`, `version=1.0`, `risk_assessment`, `decomposition_plan`, `rollback_plan`, `validation`)
  - validates required risk dimensions (`complexity`, `blast_radius`, `conceptual_clarity`) in range `1..5`
  - validates objective blast-radius evidence bundle (`risk_assessment.blast_radius_evidence`):
    - `impacted_components` (non-empty)
    - `impacted_interfaces` (non-empty)
    - `evidence_refs` (non-empty)
  - validates decomposition chunk contract:
    - each chunk includes `chunk_id`, `description`, `scope`, `task_class`
    - task class constrained to coding specialization set (`code:generate|code:edit|code:review|code:test|code:docs`)
    - bounded scope guard: max 3 files per chunk
    - non-trivial refactors (high/critical or multi-chunk) require at least one `code:review` chunk
  - validates rollback posture (`rollback_plan`):
    - `strategy` in (`revert_commit|feature_flag_revert|artifact_restore|state_restore`)
    - `trigger_conditions` (non-empty)
    - `verification_evidence_refs` (non-empty)
  - enforces tier->required-approval alignment:
    - `low`: 0
    - `medium`: 0
    - `high`: 1
    - `critical`: 2
  - projects `request.effective_risk_tier` from max(current effective risk tier, refactor packet overall tier)
- strict rollout toggle: `--require-refactor-risk-packet-for-coding`
  - when enabled, coding task classes (`implementation`, `code:*`) fail closed without `refactor_risk_packet`
  - strict mode also requires `validation.status=approved` before routing pass

Additive telemetry projection:
- top-level `refactor_risk`
- mirrored `request.refactor_risk`, `route.refactor_risk`, and `routing_audit.refactor_risk`

### Code-health + rule-governance packet contract (COD-06 / LT-07 / B9Q-008)
Support-only packet contract is now materialized for validator-ready governance without widening runtime route gates yet:
- schema: `docs/ops/schemas/code_health_rule_governance_packet.v1.schema.json`
- template: `docs/ops/templates/code_health_rule_governance_packet.v1.template.json`

Contract focus:
- code-health metric evidence with explicit per-metric evidence refs,
- rule conflict and duplicate-rule surfacing (`rule_findings.conflicts`, `rule_findings.duplicates`),
- merged-violation analytics summary (`rule_findings.merged_violation_analytics`),
- operator-readable governance recommendations and durable audit trail events.

Current posture:
- no new strict router gate is introduced in this slice;
- packet is ready for schema-level validation and downstream gate integration in later LT-07 closure work.

### COD-06 cross-packet bridge validator (LT-05/LT-06/LT-07 followthrough)
Support-only bridge contract/runtime is now available to validate coherence across regression-risk, refactor-risk, and code-health packets before stricter router integration:
- bridge schema: `docs/ops/schemas/coding_risk_packet_bridge.v1.schema.json`
- bridge template: `docs/ops/templates/coding_risk_packet_bridge.v1.template.json`
- runnable validator: `scripts/coding_risk_packet_bridge_validate.py`
- example packet: `reports/cod06_risk_packet_bridge_2026-04-04.example.json`

Bridge validator checks:
- proposal-id alignment across all three packet families,
- risk-tier floor parity (`code_health.change_scope.risk_tier` must not understate regression/refactor max tier),
- approval floor parity (`code_health.validation.required_approvals` >= max(regression/refactor)),
- action-required/blocking posture requires at least one `p0` governance recommendation,
- code-health metric summary parity and merged-violation count parity,
- refactor decomposition retains a review chunk when required by risk/multi-chunk posture.

### Optional route-lock fields
Request may declare `route_lock` (or top-level aliases):
- `route_class`
- `required_rollout_stage`
- `model_key`
- `rule_id`

If provided, these must exactly match deterministic route-policy output or routing blocks (`requested_route_mismatch`).

### Optional escalation evidence fields (XE-102 + XE-305)
Request may include `escalation_evidence` when escalating above taxonomy baseline tiers:
- `quality_gate_failed` (bool)
- `unresolved_blocker` (bool)
- `explicit_criticality` (bool)
- `previous_tier_failed` (bool)
- `artifact_refs` (array of evidence refs)

Request may also declare support-bound intent signals:
- `fold_in_target` (`canonical_doctrine | roadmap_pair | queue_continuity | support_only`)
- `support_only` (bool alias for support-bound helper work)

### Worker-allocation contract enforcement (2026-03-31 production guard)
For `session_kind=worker_slice`, production routing now enforces explicit worker-allocation metadata when invoked via
`ops/openclaw/continuity.sh session-route` (strict default):
- `scope_shape` (`single_surface | multi_surface_disjoint | multi_surface_coupled`)
- `verification_class` (`self_check | validator_required | validator_plus_human`)
- `worker_topology` (`single | parallel_fanout | staged_serial`)
- `fold_in_target` (`canonical_doctrine | roadmap_pair | queue_continuity | support_only`)

Field location:
- top-level request keys, or
- nested `dispatch_contract.{scope_shape,verification_class,worker_topology}` (fold target remains top-level).

Fail-close invariants:
- `risk_tier=high|critical` cannot use `verification_class=self_check`.
- `risk_tier=high|critical` cannot use `worker_topology=parallel_fanout`.
- `scope_shape=multi_surface_coupled` cannot use `worker_topology=parallel_fanout`.

Legacy bypass (bounded, explicit):
- `--legacy-allow-missing-worker-allocation-contract`

Default-down policy: route starts from taxonomy baseline (`NO_LLM`/`SPARK`) and escalates only when gate evidence is present. If escalation is selected without evidence, routing blocks (`escalation_evidence_missing`).

XE-305 bounded-worker hardening:
- support-only helper requests (`fold_in_target=support_only` or `support_only=true`) may still escalate to `HEAVY`, but only with explicit **non-risk** escalation signals plus non-empty `artifact_refs`.
- `risk_tier=high|critical` alone is insufficient evidence for support-only heavy escalation.
- missing non-risk signal or artifact refs on support-only heavy path blocks routing at `tier_escalation_evidence`.

### Task-class -> model-family matrix (XE-102 / XE-306)
XE-102 defines explicit model-family defaults so routing remains explainable even when route classes are unchanged; XE-306 adds bounded DeepSeek helper-lane activation for first-pass support work:
- `reading`, `triage`, `audit_compression` -> `DeepSeek` default
- `research`, `planning` -> `Gemini` default
- `comparison` -> `Kimi` default
- coding specialization classes:
  - `implementation`, `code:generate`, `code:edit`, `code:test` -> `Codex` default
  - `code:review`, `code:docs` -> `DeepSeek` default

Routing policy source of truth (configuration-driven surface):
- `docs/ops/session_topology_routing_policy_v1.json`
- schema: `docs/ops/schemas/session_topology_routing_policy.schema.json`

Router fails closed (`block_gate=routing_policy_alignment`) if this contract is missing/invalid.

Routing compatibility note:
- `code:{generate|edit|review|test|docs}` selectors are backward-compatible with existing `implementation` route rules (selector aliasing), so legacy topology files keep deterministic behavior while explicit coding-class rules are rolled in.

Router decision payloads expose `default_model_family`, `selected_model_family`, and `misrouting_signals` telemetry fields for audit/delta reporting.

XE-307 additive routing audit guardrails:
- `route.task_class` + `route.task_class_guard` expose deterministic task-class tagging guard status (`pass|warn|fail`) with taxonomy/family-matrix warnings.
- top-level `routing_audit` packet (`clawd.session_topology.routing_audit.v1`) mirrors guard + family/misrouting fields for log-side querying without transcript reconstruction.
- unmapped worker-slice task classes are surfaced via deterministic misrouting signals (`task_class_unmapped_taxonomy_profile`, `task_class_unmapped_family_matrix`) instead of silent fallback.

Operational guardrail note (2026-03-29 audit fold-in + XE-306 closeout):
- Exact-model allowlist for `SPARK` now includes bounded helper entry `deepseek/deepseek-chat` in `docs/ops/model_pool_policy_v1.json`.
- Router selection is family-prioritized by task class (`default` then `fallback`) so DeepSeek is first-pass only for bounded helper classes and falls back deterministically to existing qualified helpers when unavailable.
- `watchdog` and recurring cron/scheduler authority remain `NO_LLM` regardless of helper-family availability.
- XE-304 operational guard surfaces (`ops/openclaw/no_llm_watchdog_cron_authority_guard.sh`, `ops/openclaw/harden_no_llm_watchdog_cron_authority.sh`) enforce that model wrappers are non-authoritative (`NO_REPLY` only) while deterministic contract scripts own blocker/health verdicts.
- coding mutation lanes remain Codex-first by default and heavy escalation remains evidence-gated; helper-family activation does not widen mutation/control authority lanes.
- Kill-switch posture for the helper lane is explicit: removing `deepseek/deepseek-chat` from SPARK allowlist/pool immediately forces fallback to existing qualified helpers without route-class rewiring.

### Optional prompt lint/token guardrail fields (XE-103)
Request may include invocation-bound prompt controls:
- `invocation_prompt` (string)
- `prompt_guardrails.requested_output_tokens` (int, optional)
- `prompt_guardrails.max_prompt_tokens` (int, optional)
- `prompt_guardrails.max_total_tokens` (int, optional)

Runtime behavior:
- Router trims duplicated/blank-line boilerplate before token checks.
- Token checks are strict/fail-closed at route invocation boundary.
- Budget violation blocks routing (`prompt_token_budget_exceeded`) and emits append-only violation ledger rows.

### Optional transport conformance fields
Request may also include `transport_route` / `requested_agent_id` style hints.  
By default, route-policy routing blocks unless the supplied transport decision is present and consistent.

### Deterministic prompt/tool cache controls (XE-104)
Router supports deterministic decision caching for repeated prompt/tool inputs:
- `--prompt-tool-cache <path>` (cache store path)
- `--no-prompt-tool-cache` (disable cache)
- `--prompt-tool-cache-ttl-sec <int>` (entry TTL)
- `--prompt-tool-cache-flush` (clear cache before evaluation)

Cache behavior:
- keying: exact and semantic request fingerprints (semantic prompt key uses lint-normalized prompt text)
- invalidation: TTL expiration/stale-entry eviction on lookup
- output telemetry: decision payload includes `cache` metadata (`status`, key refs, lookup counters)

### Delta-context transport controls (XE-201)
Router emits context transport telemetry with delta-by-default behavior for supported multi-turn flows:
- request field: `context_slices` (array of `{slice_id, content}`); if absent, `invocation_prompt` is used as a fallback slice.
- `--context-delta-cache <path>` (per-flow baseline cache)
- `--context-delta-cache-flush` (clear context baseline cache before evaluation)
- `--no-context-delta-transport` (disable delta mode; force full fallback)

Delta behavior:
- first turn for a flow emits full payload with fallback reason `first_turn_no_baseline`
- subsequent turns emit only changed slices + removed slice ids
- response includes `context_transport.tokens` (`full_context_tokens`, `transmitted_tokens`, `saved_tokens`, `saved_pct`)
- response includes integrity proof (`reconstructed_snapshot_hash`, `reconstruction_ok`) to guard against state-loss regressions
- unsupported/legacy paths fail open to full transport with explicit `fallback_reason`

### Anchor-preserving summary compaction controls (XE-202)
Router also emits fail-closed compaction telemetry for deliberation-safe context reduction:
- optional request field: `immutable_anchor_slice_ids` (explicit anchor set; supports directives/acceptance/blocker anchors)
- `--context-compaction-cache <path>` (rolling compaction baseline cache)
- `--context-compaction-cache-flush` (clear compaction cache before evaluation)
- `--no-context-compaction` (disable compaction packet emission)

Compaction behavior:
- compaction output is structured into immutable-anchor layer, deliberation-capsule layer, and rollup layer
- deliberation capsules preserve primitive classes (`claims`, `tradeoffs`, `contradictions`, `decisions`) with provenance pointers
- integrity checks include anchor roundtrip reconstruction and semantic-loss guards; failures are fail-closed (`status=fail_closed`, `mode=full_passthrough`)
- successful evaluations emit `context_compaction.tokens` (`full_context_tokens`, `compacted_tokens`, `saved_tokens`, `saved_pct`) and drift telemetry (`hierarchy.drift`)

### Hybrid retrieval efficiency controls (XE-203)
Router also supports confidence-tiered selective recall for knowledge tasks via request-bound hybrid retrieval:
- optional request object: `knowledge_retrieval`
  - `enabled` / `required` (bool)
  - `query` (defaults to `invocation_prompt` when omitted)
  - `doc_intent` (`auto|policy|spec|code|incident|decision`)
  - `max_results`, `rerank_top_n`, `top_k`, `high_confidence_top_k`
  - `min_top_score`, `min_margin`, `high_confidence_top_score`, `high_confidence_margin`
  - `candidate_results` (deterministic replay/test fixture payload) or `live_search=true` (run `openclaw memory search`)

Hybrid retrieval behavior:
- source candidates come from injected `candidate_results` or live `openclaw memory search --json`
- candidates are reranked deterministically with the existing vector/keyword base score plus local metadata-aware rerank
- confidence tier decides selected recall depth (`high` => tighter top-k, `pass` => standard top-k, otherwise abstain)
- emitted packet `hybrid_retrieval` includes selected recall slices, top-candidate summaries, and token-savings telemetry versus full candidate stuffing
- when `required=true`, abstain/search-error outcomes block routing instead of silently falling back to weak full-context stuffing

### Event backbone dual-write controls (XE-301)
Router now emits typed orchestration events in dual-write mode alongside the legacy decision log:
- optional request object: `event_backbone`
  - `correlation_id` (stable flow/work-unit correlation token)
  - `idempotency_key` (stable retry/reconnect key; when omitted router derives one from deterministic routing context)
  - `sequence` (positive integer ordering hint within a correlated flow)
  - `parent_event_id` (optional causal parent ref)
- CLI controls:
  - `--event-backbone-typed-log <path>` (typed event stream JSONL)
  - `--event-backbone-db <path>` (SQLite idempotency/retry journal)
  - `--event-backbone-dlq <path>` (dead-letter queue JSONL)
  - `--event-backbone-metrics <path>` (backpressure metrics snapshot)
  - `--event-backbone-max-attempts <int>` / `--event-backbone-base-backoff-ms <int>` (bounded retry schedule)
  - `--no-event-backbone` (disable typed dual-write and keep legacy-only decision logging)

Event-backbone behavior:
- successful writes emit the legacy decision row plus a typed `session.route.decision` envelope carrying `idempotency_key`, `correlation_id`, and `legacy_parity_fingerprint`
- duplicate retries with the same `idempotency_key` suppress duplicate legacy + typed writes while preserving the original `event_id`
- conflicting payloads reusing the same `idempotency_key` fail closed as `idempotency_conflict`
- exhausted write attempts route the typed event to DLQ and update metrics with `backpressure_state` (`normal|elevated|critical`)
- router response includes `event_backbone` delivery metadata so operators can inspect publish status without scraping logs

### Replayable workflow state-machine controls (XE-302)
After XE-302, the router also computes an **authoritative workflow lifecycle state** after route evaluation + event-backbone delivery.

Request may include optional `workflow_state_machine` overrides:
- `workflow_id` (stable replay/recovery identity; defaults to explicit workflow id, then event idempotency/correlation, then transport/session binding)
- `expected_current_state` (`INIT|ROUTE_BLOCKED|ACTIVE|RECOVERY_REQUIRED`) to fail closed on stale replay assumptions

Runtime artifacts:
- `--workflow-state-journal <path>` (append-only JSONL transition journal)
- `--workflow-state-latest <path>` (latest authoritative workflow snapshot map)

Workflow-state behavior:
- state machine authority is **code-driven only**; LLM output is never consulted for routing, retry, replay, or lifecycle transition control
- healthy green state is `ACTIVE`; route-policy `PASS` is not authoritative green by itself
- route `PASS` + event publish `published`/`duplicate_suppressed` => authoritative state `ACTIVE`
- route `PASS` + event publish `dlq`/`config_error`/`idempotency_conflict`/disabled tracking => authoritative state `RECOVERY_REQUIRED` and final router verdict blocks
- route `BLOCK` => authoritative state `ROUTE_BLOCKED` (journaled for replay, but does not create a false green)
- stale `expected_current_state` or invalid state transitions fail closed before promotion to green
- workflow replayability is only valid when both the transition journal and latest snapshot update succeed; persistence failure blocks the final verdict

Authoritative output note:
- `route.*` continues to describe deterministic route-policy selection.
- top-level `decision` / `final_state` become the **workflow-authoritative verdict** after XE-302.
- `workflow_state_machine` is the canonical lifecycle packet for replay, failover recovery, and operator inspection.

Operator-default note:
- `ops/openclaw/continuity.sh session-route` is strict/fail-closed by default for transport conformance.
- Telegram direct lane is cockpit-only by default for heavy/non-trivial engine-room work:
  - `HEAVY` routes, multi-surface/parallel worker orchestration shapes, and non-trivial worker execution (`risk_tier in {medium,high,critical}` or `verification_class in {validator_required, validator_plus_human}`) block on `telegram_direct_offload` unless explicit offload is declared.
  - bounded bypass via `--legacy-allow-telegram-direct-heavy-on-dm` applies to heavy-route blocking only; non-trivial/coding offload and direct `session_kind` checks remain fail-closed.
- Use `--legacy-allow-missing-transport-decision` only for explicit bounded legacy compatibility windows.
- Deprecated `--allow-missing-transport-decision` and `--no-require-transport-decision` are ignored at continuity entrypoint.
- Direct `scripts/session_topology_router.py` invocation is also strict by default; the same legacy flags are the only bypasses.

### Declarative DAG orchestration API canary (XE-303)
Request may include optional `workflow_dag` to describe bounded subagent orchestration plans for downstream event workflows.
- `workflow_dag.nodes`: ordered list of string node IDs or objects with `id` (must be unique, non-empty)
- `workflow_dag.edges`: optional list of:
  - objects with `from`/`to`
  - or 2-tuples/2-element arrays
- request fails closed (`routing_request_invalid`) if:
  - cycle detected
  - malformed node/edge spec
  - unknown edge endpoints
  - duplicate nodes/edges

CANARY behavior:
- evaluation is deterministic and side-effect free in the request contract gate
- emitted packet in `request.workflow_dag` and top-level `workflow_dag` includes:
  - `status` (`pass`/`fail`/`not_requested`)
  - topological `execution_order`
  - deterministic `execution_levels` (parallel scheduling stages)
  - `critical_path_length` and `max_parallelism` bounds

If valid, the packet remains informational and does not alter route-policy selection gates.

## Deterministic matching algorithm (v1)
1. Validate topology contract schema.
2. Validate request contract (`session_kind`, `task_class`, `risk_tier`, optional lock fields).
3. Evaluate proposal-first delta-spec gate for coding requests (`proposal_first_delta_spec`), including approval-hook and delta-spec contract checks.
4. Evaluate regression-risk packet gate for coding requests (`regression_risk_packet`), including objective score/validation checks and effective risk-tier projection.
5. Evaluate refactor-risk + decomposition gate for coding requests (`refactor_risk_decomposition`), including bounded chunking and approval checks.
6. Validate worker-allocation contract for worker slices when strict enforcement is enabled.
7. Validate/lock against transport decision (unless explicit legacy bypass mode is enabled).
8. Select candidate rules whose selectors match request values exactly or via `*` wildcard.
9. Choose one winning rule by ordered key:
   - lowest `priority`
   - lexicographically smallest `rule_id` (tie-break)
10. If no rule matches, use `default_route_class` and `default_required_rollout_stage`.
11. Evaluate Telegram direct offload gate; if transport scope is `telegram|direct`, block heavy or non-trivial engine-room execution (`HEAVY`, multi-surface/parallel worker orchestration shape, `risk_tier in {medium,high,critical}`, validator-required worker execution, or coding worker slices not explicitly offloaded). Explicit bounded legacy bypass applies to heavy-route blocking only.
12. Evaluate taxonomy default-down gate; if selected route is above baseline tier, require escalation evidence/signals. For support-only helper requests on a `HEAVY` route, require non-risk escalation signal(s) and artifact refs (risk-tier alone is not sufficient).
13. If route class is `NO_LLM`, return deterministic no-model decision.
14. Otherwise, collect topology `model_pools[route_class]` models with PASS qualification decisions and required rollout stage in `allowed_stages`, then apply deterministic provider-selection rubric:
    - non-coding tasks: task-family preference (`default` then `fallback`) first, then qualification-signal readiness/score tie-breaks.
    - coding tasks (`implementation`, `code:*`): qualification-signal readiness/score first, then family tie-break.
    - coding tasks require explicit qualification signal presence (`effective_score_0_100` + readiness) before a model is eligible; high/critical tiers additionally enforce stricter threshold/readiness gates.
    - coding rubric now projects additive dimensions (`risk_tier`, `complexity_tier`, `verification_class`, `verification_profile`) and emits both legacy `rubric_rule_id` and expanded `rubric_rule_id_v2` for operator/audit stability.
    - medium-risk coding requests with strict readiness profile triggers must meet policy-allowed readiness states from `routing_policy.coding_qualification.strict_readiness_profile.allowed_readiness_states` (default: `qualified|provisional`); lower readiness states are deterministically disqualified.
15. If no rubric-eligible model exists, block (`no_qualified_model_for_route`).
16. If `invocation_prompt` is present, run lint trimming + token guardrails and block on budget violation (`prompt_token_budget_exceeded`).
17. Emit `context_transport` packet: delta for supported flows with baseline cache, otherwise explicit full-fallback reason.
18. Emit `context_compaction` packet: anchor-preserving deliberation capsules + rollup with fail-closed reconstruction/semantic-loss checks.
19. Emit `hybrid_retrieval` packet when request-bound knowledge retrieval is enabled: rerank candidates, apply abstain thresholds, and expose top-k token savings / selective recall slices.
20. Emit `event_backbone` delivery packet: dual-write legacy decision telemetry plus typed orchestration event publish status with idempotency, retry, DLQ, and backpressure metadata.
21. Evaluate `workflow_state_machine`: replay prior authoritative state, enforce expected-state/transition conformance, persist transition journal + latest snapshot, and derive the final fail-closed lifecycle verdict.

## Canonical route-class coverage matrix (Wave 5 helper-aware profile)
- `watchdog/*/*` -> `NO_LLM` (deterministic controls)
- `worker_slice/{reading|triage|audit_compression|planning|comparison|research}/any-risk` -> `SPARK`
- `worker_slice/{implementation|code:generate|code:edit|code:review|code:test|code:docs}/{high|critical}` -> `HEAVY` via deterministic selector alias compatibility
- `worker_slice/{implementation|code:generate|code:edit|code:review|code:test|code:docs}/{low|medium}` -> `SPARK` (helper-first drafting/analysis path)
- no matching rule -> topology defaults (`default_route_class`, `default_required_rollout_stage`)

## Fail-closed requirements
- Unknown route class or rollout stage => BLOCK.
- Missing model pool for non-`NO_LLM` route => BLOCK.
- Missing qualification decisions for non-`NO_LLM` route => BLOCK.
- Non-PASS qualification decisions are ignored and cannot satisfy routing.
- In proposal-first strict mode (`--require-proposal-first-coding`), coding requests without `proposal_packet.v1` => BLOCK (`proposal_packet_required_for_coding`).
- Invalid proposal packet alignment/fields => BLOCK (`proposal_packet_invalid`).
- High/critical proposal hooks without required approval evidence => BLOCK (`proposal_approval_missing`).
- Invalid `proposal_phase` value => BLOCK (`proposal_phase_invalid`).
- `proposal_phase in {apply,archive}` without explicit `delta_spec` => BLOCK (`proposal_apply_missing_delta_spec`).
- Provided `delta_spec` without non-empty instruction surface (`instructions` or `prompt`) => BLOCK (`delta_spec_invalid`).
- `proposal_phase=archive` without `proposal_archive` => BLOCK (`proposal_archive_missing`).
- Invalid `proposal_archive` packet (schema/task_id/artifact inventory mismatch) => BLOCK (`proposal_archive_invalid`).
- In regression-risk strict mode (`--require-regression-risk-packet-for-coding`), coding requests without `regression_risk_packet` => BLOCK (`regression_risk_packet_required_for_coding`).
- Invalid regression-risk packet fields/scores/validation alignment => BLOCK (`regression_risk_packet_invalid`).
- v2 regression-risk packet replay evidence / blocking-classification mismatch => BLOCK (`regression_risk_packet_invalid`).
- In regression-risk strict mode, packet `validation.status` must be `approved` => BLOCK (`regression_risk_not_approved`).
- In refactor-risk strict mode (`--require-refactor-risk-packet-for-coding`), coding requests without `refactor_risk_packet` => BLOCK (`refactor_risk_packet_required_for_coding`).
- Invalid refactor-risk packet fields/dimensions/decomposition/validation alignment => BLOCK (`refactor_risk_packet_invalid`).
- In refactor-risk strict mode, packet `validation.status` must be `approved` => BLOCK (`refactor_risk_not_approved`).
- coding task class (`implementation` or `code:*`) without rubric-eligible qualification signal => BLOCK (`no_qualified_model_for_route`), with stricter threshold/readiness checks for `risk_tier in {high,critical}`.
- coding task classes matching strict readiness triggers from routing policy (`coding_qualification.strict_readiness_profile.trigger_verification_classes` and/or `.trigger_complexity_tiers`) disqualify readiness outside policy allowed states (`.allowed_readiness_states`); if all candidates fail this rule, routing blocks with `no_qualified_model_for_route` and disqualification reason `strict_readiness_required_for_verification_profile`.
- legacy/missing-timestamp qualification packets remain fail-closed by default; tier-specific legacy grace values are ignored unless a global grace window is explicitly enabled in routing policy.
- when legacy grace is enabled, acceptance is bounded to `routing_policy.generated_at + grace_period_seconds` (risk-tier override if configured). Missing/invalid/future `generated_at` or expired window disables grace fail-closed.
- When strict worker-allocation enforcement is enabled, missing/invalid worker-slice dispatch fields => BLOCK (`worker_allocation_contract_violation`).
- Unless explicitly bypassed via `--legacy-allow-missing-transport-decision`, missing/invalid/mismatched transport decision => BLOCK.
- Telegram direct lane heavy-offload guard: `HEAVY` routes or multi-surface/parallel worker orchestration bound to `telegram|direct` => BLOCK (`telegram_direct_heavy_offload_required`) unless explicit bounded bypass (`--legacy-allow-telegram-direct-heavy-on-dm`) is used.
- Telegram direct lane non-trivial worker guard: worker execution bound to `telegram|direct` with `risk_tier in {medium,high,critical}` or `verification_class in {validator_required,validator_plus_human}` must declare `worker_lane=subagent_default` (offloaded); otherwise => BLOCK (`telegram_direct_worker_offload_required`). This guard is not bypassed by `--legacy-allow-telegram-direct-heavy-on-dm`.
- Telegram direct lane coding guard: coding worker slices (`implementation`, `code:*`) bound to `telegram|direct` must declare `worker_lane=subagent_default` (offloaded) or satisfy `main_session_tiny_exception` contract (`low` risk + `single_surface` + `self_check` + `single` + non-empty `delegation_basis`); lane aliases such as `subagent-default` and `tiny-exception` normalize to canonical tokens, while unknown lane tokens are surfaced as `telegram_direct_coding_worker_lane_invalid` in gate details and still fail closed with BLOCK (`telegram_direct_worker_offload_required`).
- Telegram direct lane worker-target attestation guard: when `worker_lane=subagent_default` is declared, worker handoff target evidence (`lane_name`, `agent_id`, `session_key`) must be present in transport binding context; missing evidence => BLOCK (`telegram_direct_worker_target_evidence_missing`).
- Telegram direct lane worker-target conformance guard: declared offload evidence must resolve to a worker-target identity (lane/agent pattern allowlist, no cockpit lane tokens, lane/agent alignment, and `session_key` agent-id parity); invalid evidence => BLOCK (`telegram_direct_worker_target_evidence_invalid`).
- Telegram direct lane session-kind guard: `telegram|direct` accepts only `session_kind=worker_slice`; watchdog/internal kinds fail closed with BLOCK (`telegram_direct_session_kind_invalid`), including when heavy legacy bypass is enabled.
- Prompt token guardrail over budget => BLOCK + violation ledger append.
- Support-only helper requests escalating to `HEAVY` without a non-risk escalation signal and artifact refs => BLOCK (`escalation_evidence_missing`).
- `knowledge_retrieval.required=true` + abstain/search-error outcome => BLOCK (context stuffing must not silently replace selective recall).
- Reused `event_backbone.idempotency_key` with a different `legacy_parity_fingerprint` => fail-closed `idempotency_conflict` (duplicate suppression must never mask divergent payloads).
- `workflow_state_machine.expected_current_state` mismatch => fail-closed `workflow_state_stale` (replay cannot guess prior control state).
- Workflow journal/latest snapshot persistence failure => fail-closed `workflow_state_persistence_failed` (non-replayable control paths cannot go green).
- Route-policy `PASS` + workflow authoritative state not `ACTIVE` => fail-closed at top-level (`block_gate=workflow_state_machine`) so event publish failures cannot surface as false green.

## Operator usage
- Route-policy CLI (strict-by-default):
  - `python3 scripts/session_topology_router.py --topology <topology.json> --request <request.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json`
  - optional explicit policy pins: `--routing-policy docs/ops/session_topology_routing_policy_v1.json --routing-policy-schema docs/ops/schemas/session_topology_routing_policy.schema.json`
- Continuity operator entrypoint (strict-by-default):
  - `bash ops/openclaw/continuity.sh session-route --topology <topology.json> --request <request.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json`
- Legacy bounded compatibility path (explicit opt-in only):
  - `bash ops/openclaw/continuity.sh session-route --legacy-allow-missing-worker-allocation-contract --topology <topology.json> --request <request.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json`
  - `bash ops/openclaw/continuity.sh session-route --legacy-allow-missing-transport-decision --topology <topology.json> --request <request.json> --qualification-decision <decision.json> --json`
  - `bash ops/openclaw/continuity.sh session-route --legacy-allow-telegram-direct-heavy-on-dm --topology <topology.json> --request <request.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json` (heavy-route compatibility only)
  - `python3 scripts/session_topology_router.py --legacy-allow-missing-worker-allocation-contract --topology <topology.json> --request <request.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json`
  - `python3 scripts/session_topology_router.py --legacy-allow-missing-transport-decision --topology <topology.json> --request <request.json> --qualification-decision <decision.json> --json`
  - `python3 scripts/session_topology_router.py --legacy-allow-telegram-direct-heavy-on-dm --topology <topology.json> --request <request.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json` (heavy-route compatibility only)
- Inline request fields (strict default still applies):
  - `python3 scripts/session_topology_router.py --topology <topology.json> --session-kind worker_slice --task-class implementation --risk-tier medium --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json`
- Proposal-first strict coding ingress (COD-05):
  - `python3 scripts/session_topology_router.py --require-proposal-first-coding --topology <topology.json> --request <proposal_packet_or_envelope.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json`
- Regression-risk strict coding governance (COD-06):
  - `python3 scripts/session_topology_router.py --require-regression-risk-packet-for-coding --topology <topology.json> --request <request_with_regression_risk_packet.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json`
- Refactor-risk strict coding governance (COD-06):
  - `python3 scripts/session_topology_router.py --require-refactor-risk-packet-for-coding --topology <topology.json> --request <request_with_refactor_risk_packet.json> --qualification-decision <decision.json> --transport-decision <transport_decision.json> --json`

## Notes
- v1 intentionally avoids probabilistic or latency-adaptive routing logic.
- v1 keeps deterministic selection and now requires deterministic rubric projection (`selection_rubric.rubric_rule_id` + `selection_rubric` + `selected_qualification_signal`) for operator-readable provider choice.
- COD-04 followthrough adds additive rubric dimensions (`complexity_tier`, `verification_profile`) and expanded identity (`rubric_rule_id_v2`) without breaking existing `rubric_rule_id` consumers.
- PR-06 followthrough externalizes strict-readiness triggers and allowed states into routing policy (`coding_qualification.strict_readiness_profile`) while preserving fail-closed defaults and additive rubric telemetry (`strict_readiness_allowed_states`).
- COD-05 adds additive proposal-flow telemetry (`proposal_flow`) so proposal/approval/archive expectations are queryable without widening route-policy scope.
- COD-06 adds additive regression-risk telemetry (`regression_risk`) and objective risk-tier projection (`effective_risk_tier`) so post-implementation risk packets can govern downstream routing posture.
- PR-05 followthrough adds v2 regression-risk packet compatibility with replay-evidence contract checks and deterministic blocking-classification parity while preserving v1 packet compatibility.
- COD-06 extends governance with additive refactor-risk + decomposition telemetry (`refactor_risk`) so internal complexity/chunking review posture can fail closed before coding dispatch.
- LT-07 followthrough adds a validator-ready code-health + rule-governance packet contract (`code_health_rule_governance_packet.v1`) for conflict/duplication analytics and audit-trail evidence, without changing router strict-gate behavior in this slice.
