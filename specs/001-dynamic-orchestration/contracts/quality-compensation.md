# Contract: QualityCompensationPlan v1

Status: PROPOSED. Immutable, versioned and secret-free. This contract is required for every fallback RouteDecision before it may dispatch.

## Shape
`QualityCompensationPlanV1{schema_version:"quality-compensation/v1",plan_id,decision_id,prior_route_id,selected_route_id,trigger_kind,quality_delta_codes[],required_verification:V0..V4,independence_required,required_reviewers[],acceptance_thresholds[],escalation{on_unmet:"BLOCK_DISPATCH"|"HUMAN_GATE",owner,deadline?},evidence_refs[],created_at,policy_version}`.

`acceptance_thresholds[]` contains `{metric,operator,value,unit?,evidence_required}`. `evidence_refs[]` identifies bounded, sanitized contract/result/review evidence; it MUST NOT contain prompts, credentials, raw tool/provider payloads or unbounded error bodies.

## Mandatory fallback semantics
A decision with `fallback=true` MUST contain a syntactically valid `QualityCompensationPlanV1`. Before reservation or dispatch, the policy validator MUST verify: schema version, matching decision/prior/selected route identities, non-empty trigger and quality-delta codes, a verification level at least as strict as the TaskEnvelope minimum, every required independent reviewer, at least one acceptance threshold with evidence requirement, escalation owner/action, and non-empty evidence references when the policy requires pre-dispatch evidence.

If the plan is absent, malformed, cannot meet independence, has an unmet threshold, lacks required evidence, or has no enforceable escalation, the decision status is `ACTIVATION_BLOCKED_QUALITY_COMPENSATION`; it MUST NOT reserve or dispatch. `DECISION REQUIRED` policy may instead select `HUMAN_GATE`, but it remains activation-blocked until recorded human approval satisfies the plan.

## Review, escalation and acceptance
The verification planner creates the required review work from `required_verification` and `required_reviewers`. An independent route/pool/execution cannot satisfy a required independent review. Evidence is evaluated against every threshold. A failed or unattested threshold triggers the declared escalation and records review status `FAILED` or `HUMAN_REJECTED`; it never silently downgrades the fallback requirement.

## Acceptance
- A fallback with complete valid compensation is persisted before dispatch and its reviews/evidence can be audited by ID.
- A fallback with absent, insufficient, non-independent, unattested or non-enforceable compensation is rejected before reservation/dispatch with stable code `quality_compensation_insufficient` and decision status `ACTIVATION_BLOCKED_QUALITY_COMPENSATION`.
