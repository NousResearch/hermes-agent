from __future__ import annotations

from tests.agent._dynamic_orchestration_support import (
    AcceptanceThresholdV1,
    Callable,
    CandidateEvaluation,
    CompensationEscalationV1,
    DecisionRelation,
    DomainValidationError,
    EscalationAction,
    INDEPENDENT_REVIEW_ROUTE,
    INDEPENDENT_REVIEW_ROUTE_ID,
    IndependentReviewAttestationV1,
    InitialSelectionTriggerV1,
    QualityCompensationPlanV1,
    RouteDecisionV1,
    TaskEnvelope,
    UNSELECTED_ROUTE_ID,
    asdict,
    canonical_dataclass_json,
    classification,
    decision_metadata,
    pytest,
    replace,
    replan,
    replan_after_capacity_exhaustion,
    route,
    scored_candidate,
    task_payload,
    valid_plan,
)
def test_candidate_and_initial_decision_contract_are_round_trippable_and_fail_closed():
    selected = route()
    candidate_payload = {
        "route_id": selected.route_id,
        "deterministic_status": "ELIGIBLE",
        "rejection_codes": [],
        "score": 1.0,
        "score_factors": ["healthy_capacity"],
    }
    candidate = CandidateEvaluation.from_mapping(candidate_payload)
    assert candidate.route_id == selected.route_id
    assert candidate.eligible is True
    assert candidate.score_factors == ("healthy_capacity",)

    decision_payload: dict[str, object] = {
        "schema_version": "route-decision/v1",
        "decision_id": "decision-initial",
        "task_id": "task-1",
        "attempt_id": "attempt-initial",
        "created_at": "2026-07-26T17:00:00Z",
        "policy_version": "policy/v1",
        "router_version": "router/v1",
        "capacity_view_id": "capacity-view-1",
        "effort": "E2",
        "verification": "V2",
        "fallback": False,
        "relation": "INITIAL",
        "candidates": [candidate_payload],
        "selected_route_id": selected.route_id,
        "trigger": {
            "schema_version": "initial-selection-trigger/v1",
            "kind": "initial_selection",
            "source": "policy-router",
            "evaluated_at": "2026-07-26T17:00:00Z",
        },
        "reason_codes": ["initial_selection"],
        "policy_status": "ACTIVATION_BLOCKED_INDEPENDENT_REVIEW",
        "activation_block_reason": "independent_review_required",
    }
    trusted_task = TaskEnvelope.from_mapping(
        task_payload(
            verification={
                "minimum": "V2",
                "independent_required": True,
                "human_gate_required": False,
            }
        )
    )
    trusted_initial_context = {
        "trusted_task": trusted_task,
        "trusted_routes": {selected.route_id: selected},
    }
    decision = RouteDecisionV1.from_mapping(
        decision_payload,
        **trusted_initial_context,
    )
    assert decision.dispatchable is False
    assert type(decision.trigger) is InitialSelectionTriggerV1
    assert decision.candidates[0].route_id == selected.route_id

    contradictory_task = TaskEnvelope.from_mapping(
        task_payload(task_id="different-task", policy_version="policy/trusted", effort="E4")
    )
    with pytest.raises(DomainValidationError, match="decision.trusted_task_context_invalid"):
        RouteDecisionV1.from_mapping(
            decision_payload,
            trusted_task=contradictory_task,
            trusted_routes={selected.route_id: selected},
        )

    arbitrary_selection = {
        **decision_payload,
        "candidates": [],
        "selected_route_id": UNSELECTED_ROUTE_ID,
    }
    with pytest.raises(DomainValidationError, match="decision.selected_candidate_ineligible"):
        RouteDecisionV1.from_mapping(arbitrary_selection, **trusted_initial_context)

    malformed_route = {
        **decision_payload,
        "candidates": [
            {
                **candidate_payload,
                "route_id": "not-a-route-v1-id",
            }
        ],
        "selected_route_id": "not-a-route-v1-id",
    }
    with pytest.raises(DomainValidationError, match="route.identity_invalid"):
        RouteDecisionV1.from_mapping(malformed_route, **trusted_initial_context)

    untyped_trigger = {**decision_payload, "trigger": "untyped"}
    with pytest.raises(DomainValidationError, match="decision.trigger_invalid"):
        RouteDecisionV1.from_mapping(untyped_trigger, **trusted_initial_context)

    missing_status = {
        key: value
        for key, value in decision_payload.items()
        if key != "policy_status"
    }
    with pytest.raises(DomainValidationError, match="decision.persisted_status_required"):
        RouteDecisionV1.from_mapping(missing_status)

def test_route_decision_bounds_candidate_and_route_registry_cardinality():
    routes = tuple(
        route(model=f"bounded-model-{index}", quota_pool_id=f"bounded-quota-{index}")
        for index in range(513)
    )
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-bounded-cardinality",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    trigger = InitialSelectionTriggerV1(
        "initial-selection-trigger/v1",
        "initial_selection",
        "policy-router",
        "2026-07-27T09:00:00Z",
    )
    base = {
        "decision_id": "decision-bounded-cardinality",
        "task_id": task.task_id,
        "attempt_id": "attempt-bounded-cardinality",
        **decision_metadata(),
        "verification": "V0",
        "fallback": False,
        "relation": DecisionRelation.INITIAL,
        "selected_route_id": routes[0].route_id,
        "trigger": trigger,
        "reason_codes": ("initial_selection",),
        "trusted_task": task,
    }

    candidates = tuple(
        CandidateEvaluation(
            item,
            True,
            score=float(513 - index),
            score_factors=("quality",),
        )
        for index, item in enumerate(routes[:257])
    )
    with pytest.raises(DomainValidationError, match="decision.candidates_invalid"):
        RouteDecisionV1(
            **base,
            candidates=candidates,
            trusted_routes={item.route_id: item for item in routes[:257]},
        )

    with pytest.raises(DomainValidationError, match="decision.trusted_route_context_invalid"):
        RouteDecisionV1(
            **base,
            candidates=(candidates[0],),
            trusted_routes={item.route_id: item for item in routes},
        )

    valid_decision = RouteDecisionV1(
        **base,
        candidates=(candidates[0],),
        trusted_routes={routes[0].route_id: routes[0]},
    )
    persisted = asdict(valid_decision)
    persisted["candidates"] = [asdict(candidates[0])] * 257
    with pytest.raises(DomainValidationError, match="decision.candidates_invalid"):
        RouteDecisionV1.from_mapping(
            persisted,
            trusted_task=task,
            trusted_routes={routes[0].route_id: routes[0]},
        )

    oversized_registry = {f"route-{index}": object() for index in range(513)}
    with pytest.raises(DomainValidationError, match="decision.trusted_route_context_invalid"):
        RouteDecisionV1.from_mapping(
            asdict(valid_decision),
            trusted_task=task,
            trusted_routes=oversized_registry,  # type: ignore[arg-type]
        )
    with pytest.raises(DomainValidationError, match="decision.trusted_reviewer_context_invalid"):
        RouteDecisionV1(
            **base,
            candidates=(candidates[0],),
            trusted_routes={routes[0].route_id: routes[0]},
            trusted_reviewer_routes=oversized_registry,  # type: ignore[arg-type]
        )

def test_route_decision_requires_stable_reason_codes_but_preserves_bounded_freeform_evidence():
    selected = route(model="bounded-reasons", quota_pool_id="bounded-reasons")
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-bounded-reasons",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    values = {
        "decision_id": "decision-bounded-reasons",
        "task_id": task.task_id,
        "attempt_id": "attempt-bounded-reasons",
        **decision_metadata(),
        "verification": "V0",
        "fallback": False,
        "relation": DecisionRelation.INITIAL,
        "candidates": (scored_candidate(selected),),
        "selected_route_id": selected.route_id,
        "trigger": InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy-router",
            "2026-07-27T09:00:00Z",
        ),
        "trusted_task": task,
        "trusted_routes": {selected.route_id: selected},
    }

    for invalid_reasons in (
        ("freeform reason text",),
        tuple(f"reason_{index}" for index in range(33)),
    ):
        with pytest.raises(DomainValidationError) as exc:
            RouteDecisionV1(
                **values,
                reason_codes=invalid_reasons,
            )
        assert exc.value.code == "decision.reason_codes_invalid"

    freeform_evidence = "shared quota pool remains exhausted until the next cooldown check"
    decision = RouteDecisionV1(
        **values,
        reason_codes=("initial_selection",),
        recheck_evidence=(freeform_evidence,),
    )
    assert decision.recheck_evidence == (freeform_evidence,)

    with pytest.raises(DomainValidationError, match="task.collection_invalid"):
        RouteDecisionV1(
            **values,
            reason_codes=("initial_selection",),
            recheck_evidence=("e" * 8193,),
        )

def test_candidate_audit_records_are_consistent_and_finite():
    candidate_route = route()
    with pytest.raises(DomainValidationError, match="candidate.invalid"):
        CandidateEvaluation(candidate_route, False)
    with pytest.raises(DomainValidationError, match="candidate.invalid"):
        CandidateEvaluation(candidate_route, True, ("policy_denied",))
    with pytest.raises(DomainValidationError, match="candidate.invalid"):
        CandidateEvaluation(candidate_route, True, score=float("nan"))
    with pytest.raises(DomainValidationError, match="candidate.invalid"):
        CandidateEvaluation(candidate_route, True, score=float("inf"))

def test_initial_decision_rejects_unscored_eligible_candidate():
    selected = route(model="unscored-initial", quota_pool_id="unscored-initial")
    trusted_task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-unscored-initial",
            verification={
                "minimum": "V2",
                "independent_required": True,
                "human_gate_required": False,
            },
        )
    )

    with pytest.raises(DomainValidationError, match="decision.candidate_score_required"):
        RouteDecisionV1(
            decision_id="decision-unscored-initial",
            task_id=trusted_task.task_id,
            attempt_id="attempt-unscored-initial",
            **decision_metadata(),  # type: ignore[arg-type]
            fallback=False,
            relation=DecisionRelation.INITIAL,
            candidates=(CandidateEvaluation(selected, True),),
            selected_route_id=selected.route_id,
            trigger=InitialSelectionTriggerV1(
                "initial-selection-trigger/v1",
                "initial_selection",
                "policy-router",
                "2026-07-26T17:00:00Z",
            ),
            reason_codes=("initial_selection",),
            trusted_task=trusted_task,
            trusted_routes={selected.route_id: selected},
        )

def test_valid_quality_compensation_requires_future_sealed_fallback_authority():
    failed = route(provider="anthropic", product="claude", model="opus", quota_pool_id="anthropic-a")
    alternate = route(quota_pool_id="openai-a")
    plan = valid_plan("decision-2", failed, alternate)

    decision = replan(
        task_id="task-1",
        attempt_id="attempt-1",
        decision_id="decision-2",
        failed_route=failed,
        classification=classification(failed),
        candidates=(
            CandidateEvaluation(failed, False, ("failed_route",)),
            scored_candidate(alternate),
        ),
        quality_compensation_plan=plan,
        task_verification_minimum="V2",
    )

    assert decision.relation is DecisionRelation.FALLBACK
    assert decision.selected_route_id == alternate.route_id
    assert decision.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    assert decision.activation_block_reason == "quality_compensation_insufficient"
    assert decision.dispatchable is False
    assert decision.schema_version == "route-decision/v1"
    assert decision.policy_version == "policy/v1"
    assert decision.router_version == "router/pure-v1"
    assert decision.capacity_view_id == "capacity-view:test"
    assert decision.fallback is True
    assert decision.verification == "V2"

    persisted = asdict(decision)
    trusted_task = TaskEnvelope.from_mapping(task_payload())
    trusted_reviewers = {INDEPENDENT_REVIEW_ROUTE.route_id: INDEPENDENT_REVIEW_ROUTE}
    with pytest.raises(DomainValidationError, match="decision.trusted_task_context_required"):
        RouteDecisionV1.from_mapping(persisted)

    trusted_routes = {
        failed.route_id: failed,
        alternate.route_id: alternate,
    }
    restored = RouteDecisionV1.from_mapping(
        persisted,
        trusted_routes=trusted_routes,
        trusted_task=trusted_task,
        trusted_reviewer_routes=trusted_reviewers,
        trusted_execution_routes={"review-execution": INDEPENDENT_REVIEW_ROUTE},
        trusted_execution_evidence={"review-execution": ("evidence:test-run",)},
        trusted_evidence_refs=("evidence:test-run",),
        trusted_threshold_results={"evidence:test-run": True},
    )
    assert restored == decision
    assert restored.dispatchable is False

    downgraded = asdict(decision)
    downgraded["verification"] = "V0"
    downgraded_plan = downgraded["quality_compensation_plan"]
    assert isinstance(downgraded_plan, dict)
    downgraded_plan["required_verification"] = "V0"
    with pytest.raises(DomainValidationError, match="decision.trusted_task_context_invalid"):
        RouteDecisionV1.from_mapping(
            downgraded,
            trusted_routes=trusted_routes,
            trusted_task=trusted_task,
            trusted_reviewer_routes=trusted_reviewers,
        )

    forged = asdict(decision)
    forged_plan = forged["quality_compensation_plan"]
    assert isinstance(forged_plan, dict)
    forged_plan["selected_quota_pool_id"] = "forged-unrelated-quota"
    forged_plan["selected_billing_pool_id"] = "forged-unrelated-billing"
    with pytest.raises(DomainValidationError, match="quality.route_binding_invalid"):
        RouteDecisionV1.from_mapping(
            forged,
            trusted_routes=trusted_routes,
            trusted_task=trusted_task,
            trusted_reviewer_routes=trusted_reviewers,
        )

    forged_trigger_payload = asdict(decision)
    forged_trigger = forged_trigger_payload["trigger"]
    assert isinstance(forged_trigger, dict)
    forged_trigger["attempted_route_id"] = alternate.route_id
    forged_trigger["quota_pool_id"] = alternate.quota_pool_id
    forged_trigger["billing_pool_id"] = alternate.billing_pool_id
    with pytest.raises(DomainValidationError, match="decision.trigger_route_mismatch"):
        RouteDecisionV1.from_mapping(
            forged_trigger_payload,
            trusted_routes=trusted_routes,
            trusted_task=trusted_task,
            trusted_reviewer_routes=trusted_reviewers,
        )

    unregistered = route(model="unregistered", quota_pool_id="other-q")
    typed_candidate_payload = asdict(decision)
    typed_candidate_payload["candidates"] = (
        *decision.candidates,
        CandidateEvaluation(unregistered, False, ("not_selected",)),
    )
    with pytest.raises(DomainValidationError, match="decision.trusted_route_context_required"):
        RouteDecisionV1.from_mapping(
            typed_candidate_payload,
            trusted_routes=trusted_routes,
            trusted_task=trusted_task,
            trusted_reviewer_routes=trusted_reviewers,
        )

def test_secret_bearing_nested_quality_plan_cannot_authorize_dispatch():
    failed = route(model="failed", quota_pool_id="failed-q", billing_pool_id="failed-b")
    alternate = route(model="alternate", quota_pool_id="alternate-q", billing_pool_id="alternate-b")
    plan = valid_plan("decision-sensitive-plan", failed, alternate)
    plan_payload = asdict(plan)
    thresholds = plan_payload["acceptance_thresholds"]
    assert isinstance(thresholds, tuple)
    threshold_payload = thresholds[0]
    assert isinstance(threshold_payload, dict)
    threshold_payload["value"] = "password=[REDACTED]"
    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        QualityCompensationPlanV1.from_mapping(plan_payload)

    typed_plan = valid_plan("decision-sensitive-plan", failed, alternate)
    object.__setattr__(
        typed_plan.acceptance_thresholds[0],
        "value",
        "password=[REDACTED]",
    )
    decision = replan(
        task_id="task-sensitive-plan",
        attempt_id="attempt-sensitive-plan",
        decision_id="decision-sensitive-plan",
        failed_route=failed,
        classification=classification(failed),
        candidates=(
            CandidateEvaluation(failed, False, ("failed_route",)),
            scored_candidate(alternate),
        ),
        quality_compensation_plan=typed_plan,
        task_verification_minimum="V2",
    )
    assert decision.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    assert decision.dispatchable is False

    typed_mutations: tuple[
        tuple[str, Callable[[QualityCompensationPlanV1], object], str, object], ...
    ] = (
        ("bytes secret", lambda plan: plan.acceptance_thresholds[0], "value", b"password=[REDACTED]"),
        ("schema", lambda plan: plan, "schema_version", "quality-compensation/v0"),
        ("deltas", lambda plan: plan, "quality_delta_codes", ()),
        ("owner", lambda plan: plan.escalation, "owner", ""),
        ("action", lambda plan: plan.escalation, "on_unmet", "IGNORE"),
        ("operator", lambda plan: plan.acceptance_thresholds[0], "operator", "ALWAYS"),
    )
    for _name, target, field_name, invalid_value in typed_mutations:
        adversarial_plan = valid_plan("decision-sensitive-plan", failed, alternate)
        object.__setattr__(target(adversarial_plan), field_name, invalid_value)
        blocked = replan(
            task_id="task-sensitive-plan",
            attempt_id="attempt-sensitive-plan",
            decision_id="decision-sensitive-plan",
            failed_route=failed,
            classification=classification(failed),
            candidates=(
                CandidateEvaluation(failed, False, ("failed_route",)),
                scored_candidate(alternate),
            ),
            quality_compensation_plan=adversarial_plan,
            task_verification_minimum="V2",
        )
        assert blocked.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
        assert blocked.dispatchable is False

    nested_threshold_plan = valid_plan("decision-sensitive-plan", failed, alternate)
    object.__setattr__(nested_threshold_plan.acceptance_thresholds[0], "operator", "ALWAYS")
    nested_threshold_payload = asdict(nested_threshold_plan)
    nested_threshold_payload["acceptance_thresholds"] = nested_threshold_plan.acceptance_thresholds
    with pytest.raises(DomainValidationError, match="quality.threshold_invalid"):
        QualityCompensationPlanV1.from_mapping(nested_threshold_payload)

    nested_escalation_plan = valid_plan("decision-sensitive-plan", failed, alternate)
    object.__setattr__(nested_escalation_plan.escalation, "on_unmet", "IGNORE")
    nested_escalation_payload = asdict(nested_escalation_plan)
    nested_escalation_payload["escalation"] = nested_escalation_plan.escalation
    with pytest.raises(DomainValidationError, match="quality.escalation_invalid"):
        QualityCompensationPlanV1.from_mapping(nested_escalation_payload)

def test_fallback_ranking_is_score_based_and_permutation_invariant():
    failed = route(model="failed", quota_pool_id="failed-quota", billing_pool_id="failed-billing")
    low = route(model="low", quota_pool_id="eligible-quota", billing_pool_id="eligible-billing")
    high = route(model="high", quota_pool_id="eligible-quota", billing_pool_id="eligible-billing")
    plan = valid_plan(
        "decision-score",
        failed,
        high,
        reviewed_execution_id="attempt-score",
    )

    def select(
        candidates: tuple[CandidateEvaluation, ...],
        compensation_plan: QualityCompensationPlanV1 = plan,
    ) -> str | None:
        return replan(
            task_id="task-score",
            attempt_id="attempt-score",
            decision_id="decision-score",
            failed_route=failed,
            classification=classification(failed),
            candidates=candidates,
            quality_compensation_plan=compensation_plan,
            task_verification_minimum="V2",
        ).selected_route_id

    low_candidate = CandidateEvaluation(low, True, score=1.0, score_factors=("quality",))
    high_candidate = CandidateEvaluation(high, True, score=100.0, score_factors=("quality",))
    assert select((low_candidate, high_candidate)) == high.route_id
    assert select((high_candidate, low_candidate)) == high.route_id

    tie_selected = min((low, high), key=lambda candidate_route: candidate_route.route_id)
    tie_plan = valid_plan(
        "decision-score",
        failed,
        tie_selected,
        reviewed_execution_id="attempt-score",
    )
    low_tie = CandidateEvaluation(low, True, score=10.0, score_factors=("quality",))
    high_tie = CandidateEvaluation(high, True, score=10.0, score_factors=("quality",))
    assert select((low_tie, high_tie), tie_plan) == tie_selected.route_id
    assert select((high_tie, low_tie), tie_plan) == tie_selected.route_id

def test_direct_decision_candidate_permutations_are_canonical_and_preserve_ranking():
    rejected = route(model="direct-canonical-rejected", quota_pool_id="direct-canonical-rejected")
    low = route(model="direct-canonical-low", quota_pool_id="direct-canonical-low")
    high = route(model="direct-canonical-high", quota_pool_id="direct-canonical-high")
    trusted_task = TaskEnvelope.from_mapping(
        task_payload(task_id="task-direct-canonical")
    )
    candidates = (
        CandidateEvaluation(rejected, False, ("policy_rejected",)),
        CandidateEvaluation(low, True, score=1, score_factors=("quality",)),
        CandidateEvaluation(high, True, score=100, score_factors=("quality",)),
    )
    values: dict[str, object] = {
        "decision_id": "decision-direct-canonical",
        "task_id": trusted_task.task_id,
        "attempt_id": "attempt-direct-canonical",
        **decision_metadata(),
        "fallback": False,
        "relation": DecisionRelation.INITIAL,
        "selected_route_id": high.route_id,
        "trigger": InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy-router",
            "2026-07-26T17:00:00Z",
        ),
        "reason_codes": ("initial_selection",),
        "trusted_task": trusted_task,
        "trusted_routes": {
            rejected.route_id: rejected,
            low.route_id: low,
            high.route_id: high,
        },
    }

    forward = RouteDecisionV1(candidates=candidates, **values)  # type: ignore[arg-type]
    reverse = RouteDecisionV1(candidates=tuple(reversed(candidates)), **values)  # type: ignore[arg-type]

    assert forward.selected_route_id == reverse.selected_route_id == high.route_id
    assert forward == reverse
    assert canonical_dataclass_json(forward) == canonical_dataclass_json(reverse)
    assert tuple(candidate.route_id for candidate in forward.candidates) == tuple(
        sorted(candidate.route_id for candidate in candidates)
    )

def test_mapped_decision_candidate_permutations_are_canonical():
    rejected = route(model="mapped-canonical-rejected", quota_pool_id="mapped-canonical-rejected")
    low = route(model="mapped-canonical-low", quota_pool_id="mapped-canonical-low")
    high = route(model="mapped-canonical-high", quota_pool_id="mapped-canonical-high")
    trusted_task = TaskEnvelope.from_mapping(
        task_payload(task_id="task-mapped-canonical")
    )
    trusted_routes = {
        rejected.route_id: rejected,
        low.route_id: low,
        high.route_id: high,
    }
    decision = RouteDecisionV1(
        decision_id="decision-mapped-canonical",
        task_id=trusted_task.task_id,
        attempt_id="attempt-mapped-canonical",
        **decision_metadata(),
        fallback=False,
        relation=DecisionRelation.INITIAL,
        candidates=(
            CandidateEvaluation(rejected, False, ("policy_rejected",)),
            CandidateEvaluation(low, True, score=1, score_factors=("quality",)),
            CandidateEvaluation(high, True, score=100, score_factors=("quality",)),
        ),
        selected_route_id=high.route_id,
        trigger=InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy-router",
            "2026-07-26T17:00:00Z",
        ),
        reason_codes=("initial_selection",),
        trusted_task=trusted_task,
        trusted_routes=trusted_routes,
    )
    forward_payload = asdict(decision)
    reverse_payload = {
        **forward_payload,
        "candidates": tuple(reversed(forward_payload["candidates"])),
    }

    forward = RouteDecisionV1.from_mapping(
        forward_payload,
        trusted_task=trusted_task,
        trusted_routes=trusted_routes,
    )
    reverse = RouteDecisionV1.from_mapping(
        reverse_payload,
        trusted_task=trusted_task,
        trusted_routes=trusted_routes,
    )

    assert forward == reverse
    assert canonical_dataclass_json(forward) == canonical_dataclass_json(reverse)

def test_replan_candidate_permutations_are_canonical_and_preserve_score_selection():
    failed = route(model="replan-canonical-failed", quota_pool_id="replan-canonical-failed")
    low = route(model="replan-canonical-low", quota_pool_id="replan-canonical-low")
    high = route(model="replan-canonical-high", quota_pool_id="replan-canonical-high")
    candidates = (
        CandidateEvaluation(failed, False, ("failed_route",)),
        CandidateEvaluation(low, True, score=1, score_factors=("quality",)),
        CandidateEvaluation(high, True, score=100, score_factors=("quality",)),
    )
    plan = valid_plan(
        "decision-replan-canonical",
        failed,
        high,
        reviewed_execution_id="attempt-replan-canonical",
    )

    def build(raw_candidates: tuple[CandidateEvaluation, ...]) -> RouteDecisionV1:
        return replan(
            task_id="task-replan-canonical",
            attempt_id="attempt-replan-canonical",
            decision_id="decision-replan-canonical",
            failed_route=failed,
            classification=classification(failed),
            candidates=raw_candidates,
            quality_compensation_plan=plan,
            task_verification_minimum="V2",
        )

    forward = build(candidates)
    reverse = build(tuple(reversed(candidates)))

    assert forward.selected_route_id == reverse.selected_route_id == high.route_id
    assert forward == reverse
    assert canonical_dataclass_json(forward) == canonical_dataclass_json(reverse)

def test_direct_decisions_enforce_ranking_e0_waiting_and_review_authority():
    low = route(model="direct-low", quota_pool_id="direct-low-q")
    high = route(model="direct-high", quota_pool_id="direct-high-q")
    initial_trigger = InitialSelectionTriggerV1(
        schema_version="initial-selection-trigger/v1",
        kind="initial_selection",
        source="policy-router",
        evaluated_at="2026-07-26T17:00:00Z",
    )
    direct_task = TaskEnvelope.from_mapping(
        task_payload(task_id="task-direct-ranking")
    )
    direct_values: dict[str, object] = {
        "decision_id": "decision-direct-ranking",
        "task_id": "task-direct-ranking",
        "attempt_id": "attempt-direct-ranking",
        **decision_metadata(),
        "fallback": False,
        "relation": DecisionRelation.INITIAL,
        "candidates": (
            CandidateEvaluation(low, True, score=1.0, score_factors=("quality",)),
            CandidateEvaluation(high, True, score=100.0, score_factors=("quality",)),
        ),
        "selected_route_id": low.route_id,
        "trigger": initial_trigger,
        "reason_codes": ("initial_selection",),
        "trusted_task": direct_task,
        "trusted_routes": {
            low.route_id: low,
            high.route_id: high,
        },
    }
    with pytest.raises(DomainValidationError, match="decision.selection_rank_invalid"):
        RouteDecisionV1(**direct_values)  # type: ignore[arg-type]

    direct_values["effort"] = "E0"
    direct_values["selected_route_id"] = high.route_id
    direct_values["trusted_task"] = TaskEnvelope.from_mapping(
        task_payload(task_id="task-direct-ranking", effort="E0")
    )
    with pytest.raises(DomainValidationError, match="decision.effort_route_prohibited"):
        RouteDecisionV1(**direct_values)  # type: ignore[arg-type]

    failed = route(model="waiting-failed", quota_pool_id="waiting-failed-q")
    unrelated = route(model="waiting-unrelated", quota_pool_id="waiting-unrelated-q")
    with pytest.raises(DomainValidationError, match="decision.trigger_route_mismatch"):
        RouteDecisionV1(
            decision_id="decision-waiting-mismatch",
            task_id="task-waiting-mismatch",
            attempt_id="attempt-waiting-mismatch",
            **decision_metadata(),  # type: ignore[arg-type]
            fallback=False,
            relation=DecisionRelation.WAITING,
            candidates=(CandidateEvaluation(failed, False, ("capacity",)),),
            selected_route_id=None,
            trigger=classification(unrelated),
            reason_codes=("waiting",),
            prior_route_id=failed.route_id,
            trusted_task=TaskEnvelope.from_mapping(
                task_payload(task_id="task-waiting-mismatch")
            ),
            trusted_routes={failed.route_id: failed},
            trusted_prior_route=failed,
        )

    selected = route(model="review-selected", quota_pool_id="review-selected-q")
    mismatched_review = valid_plan(
        "decision-review-binding",
        failed,
        selected,
        reviewed_execution_id="other-attempt",
    )
    blocked = replan(
        task_id="task-review-binding",
        attempt_id="actual-attempt",
        decision_id="decision-review-binding",
        failed_route=failed,
        classification=classification(failed),
        candidates=(
            CandidateEvaluation(failed, False, ("failed_route",)),
            scored_candidate(selected),
        ),
        quality_compensation_plan=mismatched_review,
        task_verification_minimum="V2",
    )
    assert blocked.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    assert blocked.dispatchable is False

    forged_attestation = replace(
        mismatched_review.review_attestations[0],
        reviewed_execution_id="actual-attempt",
        quota_pool_id="forged-review-pool",
    )
    forged_reviewer_plan = replace(
        mismatched_review,
        review_attestations=(forged_attestation,),
    )
    forged_reviewer_decision = replan(
        task_id="task-review-binding",
        attempt_id="actual-attempt",
        decision_id="decision-review-binding",
        failed_route=failed,
        classification=classification(failed),
        candidates=(
            CandidateEvaluation(failed, False, ("failed_route",)),
            scored_candidate(selected),
        ),
        quality_compensation_plan=forged_reviewer_plan,
        task_verification_minimum="V2",
    )
    assert forged_reviewer_decision.policy_status == (
        "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    )
    assert forged_reviewer_decision.dispatchable is False

    with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
        TaskEnvelope.from_mapping(task_payload(objective="inspect (github_pat_REDACTED)"))

    for objective in (
        "inspect _github_pat_REDACTED",
        "inspect _gho_REDACTED",
        "inspect _xoxb-REDACTED",
        "inspect _sk-proj-REDACTED",
    ):
        with pytest.raises(DomainValidationError, match="decision.sensitive_field_prohibited"):
            TaskEnvelope.from_mapping(task_payload(objective=objective))

def test_task_authority_binds_replan_and_quality_policy_requirements():
    failed = route(model="authority-failed", quota_pool_id="authority-failed-q")
    selected = route(model="authority-selected", quota_pool_id="authority-selected-q")
    candidates = (
        CandidateEvaluation(failed, False, ("failed_route",)),
        scored_candidate(selected),
    )
    authoritative_task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="trusted-task",
            policy_version="policy/trusted",
            effort="E4",
            verification={
                "minimum": "V4",
                "independent_required": True,
                "human_gate_required": True,
            },
        )
    )
    with pytest.raises(DomainValidationError, match="replan.trusted_task_mismatch"):
        replan_after_capacity_exhaustion(
            trusted_task=authoritative_task,
            task_id="forged-task",
            attempt_id="forged-attempt",
            failed_route=failed,
            classification=classification(failed),
            candidates=candidates,
            decision_id="forged-decision",
            parent_decision_id="parent-forged-decision",
            **decision_metadata(),  # type: ignore[arg-type]
            quality_compensation_plan=None,
            trusted_reviewer_routes={
                INDEPENDENT_REVIEW_ROUTE.route_id: INDEPENDENT_REVIEW_ROUTE
            },
        )

    task_independent = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-quality-authority",
            verification={
                "minimum": "V2",
                "independent_required": True,
                "human_gate_required": False,
            },
        )
    )
    task_human = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-quality-authority",
            verification={
                "minimum": "V2",
                "independent_required": True,
                "human_gate_required": True,
            },
        )
    )
    cases = (
        (
            valid_plan(
                "decision-quality-authority",
                failed,
                selected,
                reviewed_execution_id="attempt-quality-authority",
                independence_required=False,
                required_reviewers=(),
                review_attestations=(),
            ),
            task_independent,
        ),
        (
            valid_plan(
                "decision-quality-authority",
                failed,
                selected,
                reviewed_execution_id="attempt-quality-authority",
            ),
            task_human,
        ),
        (
            valid_plan(
                "decision-quality-authority",
                failed,
                selected,
                reviewed_execution_id="attempt-quality-authority",
                human_approval_ref="approval:forged",
            ),
            task_human,
        ),
        (
            valid_plan(
                "decision-quality-authority",
                failed,
                selected,
                reviewed_execution_id="attempt-quality-authority",
                policy_version="policy/attacker",
            ),
            TaskEnvelope.from_mapping(task_payload(task_id="task-quality-authority")),
        ),
        (
            valid_plan(
                "decision-quality-authority",
                failed,
                selected,
                reviewed_execution_id="attempt-quality-authority",
                required_verification="V4",
            ),
            TaskEnvelope.from_mapping(task_payload(task_id="task-quality-authority")),
        ),
    )
    for plan, trusted_task in cases:
        decision = replan(
            trusted_task=trusted_task,
            task_id="task-quality-authority",
            attempt_id="attempt-quality-authority",
            decision_id="decision-quality-authority",
            failed_route=failed,
            classification=classification(failed),
            candidates=candidates,
            quality_compensation_plan=plan,
            task_verification_minimum="V2",
        )
        assert decision.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
        assert decision.dispatchable is False

    approved_plan = valid_plan(
        "decision-quality-authority",
        failed,
        selected,
        reviewed_execution_id="attempt-quality-authority",
        human_approval_ref="approval:trusted",
    )
    approved = replan(
        trusted_task=task_human,
        trusted_human_approval_refs={"approval:trusted": "task-quality-authority"},
        task_id="task-quality-authority",
        attempt_id="attempt-quality-authority",
        decision_id="decision-quality-authority",
        failed_route=failed,
        classification=classification(failed),
        candidates=candidates,
        quality_compensation_plan=approved_plan,
        task_verification_minimum="V2",
    )
    assert approved.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    assert approved.dispatchable is False

@pytest.mark.parametrize(
    "failure_kind",
    (
        "absent",
        "invalid_schema",
        "mismatched",
        "insufficient_verification",
        "non_independent",
        "unattested",
        "unmet",
        "unenforceable",
    ),
)
def test_invalid_quality_compensation_blocks_fallback_without_hiding_candidate(failure_kind: str):
    failed = route(provider="anthropic", product="claude", model="opus", quota_pool_id="anthropic-a")
    alternate = route(quota_pool_id="openai-a")
    plan: QualityCompensationPlanV1 | dict[str, object] | None
    if failure_kind == "absent":
        plan = None
    elif failure_kind == "invalid_schema":
        plan = {"schema_version": "quality-compensation/v0"}
    elif failure_kind == "mismatched":
        plan = valid_plan("decision-2", failed, UNSELECTED_ROUTE_ID)
    elif failure_kind == "insufficient_verification":
        plan = valid_plan(
            "decision-2",
            failed,
            alternate,
            required_verification="V1",
            independence_required=False,
        )
    elif failure_kind == "non_independent":
        plan = valid_plan(
            "decision-2",
            failed,
            alternate,
            required_reviewers=(alternate.route_id,),
        )
    elif failure_kind in {"unattested", "unmet"}:
        plan = valid_plan(
            "decision-2",
            failed,
            alternate,
            acceptance_thresholds=(
                AcceptanceThresholdV1(
                    metric="focused_tests",
                    operator=">=",
                    value=1,
                    evidence_required=True,
                    evidence_ref=(
                        "evidence:test-run" if failure_kind == "unmet" else None
                    ),
                    met=False if failure_kind == "unmet" else None,
                ),
            ),
        )
    else:
        plan = {
            "schema_version": "quality-compensation/v1",
            "plan_id": "plan-bad",
            "decision_id": "decision-2",
            "prior_route_id": failed.route_id,
            "selected_route_id": alternate.route_id,
            "trigger_kind": "capacity_exhausted",
            "quality_delta_codes": ["different_model_family"],
            "required_verification": "V2",
            "independence_required": True,
            "required_reviewers": ["review-route-independent"],
            "acceptance_thresholds": [],
            "escalation": {"on_unmet": "IGNORE", "owner": "nobody"},
            "evidence_refs": ["evidence:test-run"],
            "created_at": "2026-07-26T17:00:00Z",
            "policy_version": "policy/v1",
        }

    decision = replan(
        task_id="task-1",
        attempt_id="attempt-1",
        decision_id="decision-2",
        failed_route=failed,
        classification=classification(failed),
        candidates=(scored_candidate(alternate),),
        quality_compensation_plan=plan,
        task_verification_minimum="V2",
    )

    assert decision.relation is DecisionRelation.FALLBACK
    assert decision.selected_route_id == alternate.route_id
    assert decision.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    assert decision.activation_block_reason == "quality_compensation_insufficient"
    assert "quality_compensation_insufficient" in decision.reason_codes
    assert decision.dispatchable is False

def test_activation_blocked_fallback_cannot_claim_reservation():
    failed = route(model="opus", quota_pool_id="failed-quota")
    alternate = route(model="gpt", quota_pool_id="alternate-quota")
    blocked = replan(
        task_id="task-1",
        attempt_id="attempt-1",
        decision_id="decision-2",
        failed_route=failed,
        classification=classification(failed),
        candidates=(scored_candidate(alternate),),
        quality_compensation_plan=None,
        task_verification_minimum="V2",
    )
    assert blocked.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    with pytest.raises(DomainValidationError, match="decision.blocked_reservation"):
        replace(
            blocked,
            reservation_id="reservation-held",
            trusted_task=TaskEnvelope.from_mapping(task_payload()),
            trusted_reviewer_routes={
                INDEPENDENT_REVIEW_ROUTE.route_id: INDEPENDENT_REVIEW_ROUTE
            },
            trusted_routes={
                failed.route_id: failed,
                alternate.route_id: alternate,
            },
            trusted_prior_route=failed,
        )

def test_quality_independence_requires_route_pool_and_execution_attestation():
    failed = route(
        model="opus",
        quota_pool_id="anthropic-quota",
        billing_pool_id="anthropic-billing",
    )
    alternate = route(quota_pool_id="openai-quota", billing_pool_id="openai-billing")
    invalid_attestations = (
        (),
        (
            IndependentReviewAttestationV1(
                "reviewer",
                alternate.route_id,
                "review-quota",
                "review-billing",
                "fallback-execution",
                "review-execution",
                "evidence:test-run",
            ),
        ),
        (
            IndependentReviewAttestationV1(
                "reviewer",
                INDEPENDENT_REVIEW_ROUTE_ID,
                failed.quota_pool_id,
                "review-billing",
                "fallback-execution",
                "review-execution",
                "evidence:test-run",
            ),
        ),
        (
            IndependentReviewAttestationV1(
                "reviewer",
                INDEPENDENT_REVIEW_ROUTE_ID,
                "review-quota",
                alternate.billing_pool_id,
                "fallback-execution",
                "review-execution",
                "evidence:test-run",
            ),
        ),
        (
            IndependentReviewAttestationV1(
                "reviewer",
                INDEPENDENT_REVIEW_ROUTE_ID,
                "review-quota",
                "review-billing",
                "fallback-execution",
                "attempt-1",
                "evidence:test-run",
            ),
        ),
        (
            IndependentReviewAttestationV1(
                "reviewer",
                INDEPENDENT_REVIEW_ROUTE_ID,
                "review-quota",
                "review-billing",
                "same-execution",
                "same-execution",
                "evidence:test-run",
            ),
        ),
    )
    for attestations in invalid_attestations:
        plan = valid_plan(
            "decision-2",
            failed,
            alternate,
            required_reviewers=("reviewer",),
            review_attestations=attestations,
        )
        decision = replan(
            task_id="task-1",
            attempt_id="attempt-1",
            decision_id="decision-2",
            failed_route=failed,
            classification=classification(failed),
            candidates=(scored_candidate(alternate),),
            quality_compensation_plan=plan,
            task_verification_minimum="V2",
        )
        assert decision.dispatchable is False
        assert decision.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"

def test_human_gate_is_fail_closed_without_external_approval_authority():
    failed = route(model="opus", quota_pool_id="failed-quota")
    alternate = route(model="gpt", quota_pool_id="alternate-quota")
    plan = valid_plan(
        "decision-2",
        failed,
        alternate,
        escalation=CompensationEscalationV1(
            on_unmet=EscalationAction.HUMAN_GATE,
            owner="human-owner",
        ),
        human_approval_ref="evidence:test-run",
    )
    decision = replan(
        task_id="task-1",
        attempt_id="attempt-1",
        decision_id="decision-2",
        failed_route=failed,
        classification=classification(failed),
        candidates=(scored_candidate(alternate),),
        quality_compensation_plan=plan,
        task_verification_minimum="V2",
    )
    assert decision.dispatchable is False
    assert decision.activation_block_reason == "quality_compensation_insufficient"

def test_initial_route_authority_task_gates_reservation_and_e0_contract():
    selected = route(model="initial-authority")
    trigger = InitialSelectionTriggerV1(
        "initial-selection-trigger/v1",
        "initial_selection",
        "policy",
        "2026-07-26T17:00:00Z",
    )
    base: dict[str, object] = {
        "decision_id": "initial-authority-decision",
        "task_id": "initial-authority-task",
        "attempt_id": "initial-authority-attempt",
        **decision_metadata(),
        "fallback": False,
        "relation": DecisionRelation.INITIAL,
        "candidates": (CandidateEvaluation(selected, True, score=1, score_factors=("policy",)),),
        "selected_route_id": selected.route_id,
        "trigger": trigger,
        "reason_codes": (),
    }
    normal_task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="initial-authority-task",
            verification={"minimum": "V2", "independent_required": True, "human_gate_required": False},
        )
    )
    with pytest.raises(DomainValidationError, match="decision.trusted_route_context_required"):
        RouteDecisionV1(**base, trusted_task=normal_task)

    independent_task = TaskEnvelope.from_mapping(task_payload(task_id="initial-authority-task"))
    independent = RouteDecisionV1(
        **base,
        trusted_task=independent_task,
        trusted_routes={selected.route_id: selected},
    )
    assert independent.policy_status == "ACTIVATION_BLOCKED_INDEPENDENT_REVIEW"
    assert independent.dispatchable is False

    human_task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="initial-authority-task",
            verification={"minimum": "V1", "independent_required": False, "human_gate_required": True},
        )
    )
    with pytest.raises(DomainValidationError, match="decision.blocked_reservation"):
        RouteDecisionV1(
            **base,
            reservation_id="reservation-self-claimed",
            trusted_task=human_task,
            trusted_routes={selected.route_id: selected},
        )
    approved = RouteDecisionV1(
        **base,
        trusted_task=human_task,
        trusted_routes={selected.route_id: selected},
        trusted_human_approval_refs={"approval:trusted": "initial-authority-task"},
    )
    assert approved.policy_status == "ACTIVATION_BLOCKED_HUMAN_GATE"
    assert approved.dispatchable is False
    unrelated_approval = RouteDecisionV1(
        **base,
        trusted_task=human_task,
        trusted_routes={selected.route_id: selected},
        trusted_human_approval_refs={"approval:trusted": "some-other-task"},
    )
    assert unrelated_approval.dispatchable is False

    e0_task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="initial-authority-task",
            effort="E0",
            verification={"minimum": "V0", "independent_required": False, "human_gate_required": False},
        )
    )
    e0_values = dict(base)
    e0_values.update(effort="E0", verification="V0", candidates=(), selected_route_id=None)
    e0 = RouteDecisionV1(**e0_values, trusted_task=e0_task, trusted_routes={})
    assert e0.policy_status == "NO_ROUTE_REQUIRED"
    assert e0.dispatchable is False

def test_quality_authority_requires_execution_evidence_threshold_and_rejects_e0_replan():
    failed = route(model="authority-failed", quota_pool_id="authority-failed-q")
    selected = route(model="authority-selected", quota_pool_id="authority-selected-q")
    plan = valid_plan(
        "quality-authority-decision",
        failed,
        selected,
        reviewed_execution_id="quality-authority-attempt",
    )
    common = {
        "task_id": "quality-authority-task",
        "attempt_id": "quality-authority-attempt",
        "decision_id": "quality-authority-decision",
        "failed_route": failed,
        "classification": classification(failed),
        "candidates": (scored_candidate(selected),),
        "quality_compensation_plan": plan,
        "task_verification_minimum": "V2",
    }
    for missing_context in (
        {"trusted_execution_routes": None},
        {"trusted_execution_evidence": None},
        {"trusted_execution_evidence": {"review-execution": ("evidence:unrelated",)}},
        {"trusted_evidence_refs": ()},
        {"trusted_threshold_results": {"evidence:test-run": False}},
    ):
        decision = replan(**common, **missing_context)
        assert decision.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
        assert decision.dispatchable is False

    e0_task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="quality-authority-task",
            effort="E0",
            verification={"minimum": "V0", "independent_required": False, "human_gate_required": False},
        )
    )
    with pytest.raises(DomainValidationError, match="replan.effort_route_prohibited"):
        e0_common = {**common, "task_verification_minimum": "V0"}
        replan(**e0_common, trusted_task=e0_task)

def test_decision_rejects_non_mapping_optional_route_registries():
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="registry-shape-task",
            effort="E0",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    with pytest.raises(
        DomainValidationError,
        match="decision.trusted_reviewer_context_invalid",
    ):
        RouteDecisionV1(
            decision_id="registry-shape-decision",
            task_id=task.task_id,
            attempt_id="registry-shape-attempt",
            **{**decision_metadata(), "effort": "E0", "verification": "V0"},
            fallback=False,
            relation=DecisionRelation.INITIAL,
            candidates=(),
            selected_route_id=None,
            trigger=InitialSelectionTriggerV1(
                "initial-selection-trigger/v1",
                "initial_selection",
                "policy",
                "2026-07-26T17:00:00Z",
            ),
            reason_codes=(),
            trusted_task=task,
            trusted_routes={},
            trusted_reviewer_routes=[],  # type: ignore[arg-type]
        )
