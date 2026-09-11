# Contract: RouteDecision v1

Status: PROPOSED. Immutable/persisted before dispatch; no secrets.

`CandidateEvaluation{route_id, deterministic_status, rejection_codes[], score?, score_factors[]}`.
`RuntimeErrorClassificationV1{kind,source,evidence_code?,attempted_route_id,quota_pool_id,billing_pool_id?,classified_at}` is PROPOSED and secret-free.
`RouteDecisionV1{schema_version:"route-decision/v1",decision_id,parent_decision_id?,task_id,attempt_id,created_at,policy_version,router_version,capacity_view_id,effort:E0..E4,verification:V0..V4,policy_status,candidates[],selected_route_id?,reservation_id?,reason_codes[],fallback,quality_compensation_plan?,activation_block_reason?}`.

Policy filters before score: identity/policy, privacy/permission, capability/tool, context, freshness/confidence, budget, breaker/cooldown, concurrency/reservation unit. Score only eligible candidates. A `CapacityView` may be reused only while every pool entry used by this decision remains within `valid_until` at `created_at` and no route/pool-scoped capacity event, reservation mutation, breaker/cooldown update, or authorized refresh has occurred since `built_at`; otherwise recompute a fresh view and record its new `capacity_view_id`.

Fallback is a new linked decision, never “next provider”. It records typed trigger, prior route, invalidated observations, reconciliation, recompute ID, candidate/rejections, reason and `QualityCompensationPlanV1` (`contracts/quality-compensation.md`). For `fallback=true`, `quality_compensation_plan` is mandatory and must pass its pre-dispatch enforcement; absent or insufficient compensation sets `policy_status=ACTIVATION_BLOCKED_QUALITY_COMPENSATION`, records `activation_block_reason=quality_compensation_insufficient`, and prohibits reservation/dispatch. A `DECISION REQUIRED` human gate remains blocked until recorded approval satisfies the plan.

The typed mapping `RuntimeErrorClassificationV1.kind=capacity_exhausted -> route_capacity_exhausted` is PROPOSED and route/pool-scoped exclusively from the attempted normalized route (`attempted_route_id`, `quota_pool_id`, optional `billing_pool_id`), never from a provider/model name alone. The current runtime `FailoverReason` has no such typed mapping. **DECISION REQUIRED before wiring:** define the exact provider/runtime error classifications, evidence threshold, retryability and breaker policy that qualify for this mapping; unrecognized/ambiguous failures MUST NOT be promoted to `route_capacity_exhausted`.

`route_capacity_exhausted`: fail attempt; invalidate scoped observation; update cooldown/breaker; reconcile/release idempotently; authorized refresh; recompute; if eligible route exists create fallback, validate compensation, reserve/dispatch/verify; otherwise (and only otherwise) wait with recheck condition.

Invariants: pool distinctions alter route_id; adapter cannot replace route/recompute policy; WAITING cannot coexist with eligible candidate; unknown capacity is never healthy, zero or unlimited; telemetry excludes prompt/secret/raw provider body.
