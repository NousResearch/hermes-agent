from __future__ import annotations

from tests.agent._dynamic_orchestration_support import (
    CandidateEvaluation,
    CapacityConflictDisposition,
    CapacityObservationV1,
    CapacityValueV1,
    CapacityViewV1,
    DecisionRelation,
    DomainValidationError,
    ErrorKind,
    PoolCapacityV1,
    RouteDecisionV1,
    RuntimeErrorClassificationV1,
    TaskEnvelope,
    asdict,
    breaker_payload,
    canonical_dataclass_json,
    capacity_observation,
    capacity_reservation,
    capacity_view,
    classification,
    fields,
    orchestration,
    pytest,
    replace,
    replan,
    replan_after_capacity_exhaustion,
    reservation_payload,
    route,
    task_payload,
    valid_plan,
)
def test_capacity_replan_rejects_unscored_eligible_candidate():
    failed = route(model="failed-unscored", quota_pool_id="failed-unscored")
    alternate = route(model="alternate-unscored", quota_pool_id="alternate-unscored")

    with pytest.raises(DomainValidationError, match="replan.candidate_score_required"):
        replan(
            task_id="task-unscored-replan",
            attempt_id="attempt-unscored-replan",
            decision_id="decision-unscored-replan",
            failed_route=failed,
            classification=classification(failed),
            candidates=(CandidateEvaluation(alternate, True),),
            task_verification_minimum="V2",
        )

def test_capacity_replan_bounds_candidates_before_revalidation():
    class OversizedCandidates(list[object]):
        def __iter__(self):
            raise AssertionError("candidate validation must not start above the bound")

    failed = route(model="failed-bounded-replan", quota_pool_id="failed-bounded-replan")
    with pytest.raises(DomainValidationError, match="decision.candidates_invalid"):
        replan(
            task_id="task-bounded-replan",
            attempt_id="attempt-bounded-replan",
            decision_id="decision-bounded-replan",
            failed_route=failed,
            classification=classification(failed),
            candidates=OversizedCandidates([object()] * 257),  # type: ignore[arg-type]
        )

def test_capacity_replan_matches_ordinary_public_decision_construction():
    failed = route(
        model="replan-equivalence-failed",
        billing_pool_id="replan-equivalence-failed-billing",
        quota_pool_id="replan-equivalence-failed-quota",
    )
    same_pool = route(
        model="replan-equivalence-same-pool",
        billing_pool_id="replan-equivalence-same-pool-billing",
        quota_pool_id=failed.quota_pool_id,
    )
    selected = route(
        model="replan-equivalence-selected",
        billing_pool_id="replan-equivalence-selected-billing",
        quota_pool_id="replan-equivalence-selected-quota",
    )
    trusted_task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-replan-equivalence",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    exhausted = classification(failed)
    candidates = (
        CandidateEvaluation(
            same_pool,
            True,
            score=100,
            score_factors=("quality",),
        ),
        CandidateEvaluation(
            selected,
            True,
            score=10,
            score_factors=("quality",),
        ),
    )
    plan = valid_plan(
        "decision-replan-equivalence",
        failed,
        selected,
        reviewed_execution_id="attempt-replan-equivalence",
    )
    actual = replan_after_capacity_exhaustion(
        trusted_task=trusted_task,
        task_id=trusted_task.task_id,
        attempt_id="attempt-replan-equivalence",
        failed_route=failed,
        classification=exhausted,
        candidates=candidates,
        decision_id="decision-replan-equivalence",
        parent_decision_id="parent-replan-equivalence",
        created_at="2026-07-27T17:00:00Z",
        policy_version=trusted_task.policy_version,
        router_version="router/pure-v1",
        capacity_view_id="capacity-view:replan-equivalence",
        effort=trusted_task.effort,
        verification="V0",
        quality_compensation_plan=plan,
    )

    scoped_classification = RuntimeErrorClassificationV1(
        kind=exhausted.kind,
        source=exhausted.source,
        attempted_route_id=exhausted.attempted_route_id,
        quota_pool_id=exhausted.quota_pool_id,
        billing_pool_id=failed.billing_pool_id,
        classified_at=exhausted.classified_at,
        evidence_code=exhausted.evidence_code,
    )
    evaluated_candidates = (
        CandidateEvaluation(
            same_pool,
            False,
            ("same_exhausted_quota_pool",),
        ),
        CandidateEvaluation(
            selected,
            True,
            score=10,
            score_factors=("quality",),
        ),
    )
    ordinary = RouteDecisionV1(
        decision_id="decision-replan-equivalence",
        task_id=trusted_task.task_id,
        attempt_id="attempt-replan-equivalence",
        created_at="2026-07-27T17:00:00Z",
        policy_version=trusted_task.policy_version,
        router_version="router/pure-v1",
        capacity_view_id="capacity-view:replan-equivalence",
        effort=trusted_task.effort,
        verification="V0",
        fallback=True,
        relation=DecisionRelation.FALLBACK,
        candidates=evaluated_candidates,
        selected_route_id=selected.route_id,
        trigger=scoped_classification,
        reason_codes=("route_capacity_exhausted",),
        prior_route_id=failed.route_id,
        parent_decision_id="parent-replan-equivalence",
        quality_compensation_plan=plan,
        trusted_task=trusted_task,
        trusted_routes={
            failed.route_id: failed,
            same_pool.route_id: same_pool,
            selected.route_id: selected,
        },
        trusted_prior_route=failed,
    )

    assert actual == ordinary
    assert canonical_dataclass_json(actual) == canonical_dataclass_json(ordinary)

@pytest.mark.parametrize(
    "location",
    (
        "decision_id",
        "task_id",
        "attempt_id",
        "parent_decision_id",
        "created_at",
        "policy_version",
        "router_version",
        "capacity_view_id",
        "trusted_task",
        "classification",
        "candidate",
        "recheck_evidence",
        "trusted_reviewer_routes",
        "trusted_human_approval_refs",
        "trusted_execution_routes",
        "trusted_execution_evidence_key",
        "trusted_execution_evidence_value",
        "trusted_evidence_refs",
        "trusted_threshold_results",
    ),
)
def test_capacity_replan_sensitive_errors_match_ordinary_public_constructor(
    location: str,
):
    secret = "api_key=wave3a2-secret"
    failed = route(
        model=f"sensitive-parity-failed-{location}",
        billing_pool_id=f"sensitive-parity-failed-billing-{location}",
        quota_pool_id=f"sensitive-parity-failed-quota-{location}",
    )
    selected = route(
        model=f"sensitive-parity-selected-{location}",
        billing_pool_id=f"sensitive-parity-selected-billing-{location}",
        quota_pool_id=f"sensitive-parity-selected-quota-{location}",
    )
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id=f"sensitive-parity-task-{location}",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    trigger = RuntimeErrorClassificationV1(
        kind=ErrorKind.CAPACITY_EXHAUSTED,
        source="typed-runtime-error",
        attempted_route_id=failed.route_id,
        quota_pool_id=failed.quota_pool_id,
        billing_pool_id=failed.billing_pool_id,
        classified_at="2026-07-27T17:00:00Z",
    )
    candidate = CandidateEvaluation(
        selected,
        True,
        score=1,
        score_factors=("quality",),
    )
    public_values: dict[str, object] = {
        "decision_id": f"sensitive-parity-decision-{location}",
        "task_id": task.task_id,
        "attempt_id": f"sensitive-parity-attempt-{location}",
        "created_at": "2026-07-27T17:00:01Z",
        "policy_version": task.policy_version,
        "router_version": "router/pure-v1",
        "capacity_view_id": f"capacity-view:sensitive-parity-{location}",
        "effort": task.effort,
        "verification": "V0",
        "fallback": True,
        "relation": DecisionRelation.FALLBACK,
        "candidates": (candidate,),
        "selected_route_id": selected.route_id,
        "trigger": trigger,
        "reason_codes": ("route_capacity_exhausted",),
        "prior_route_id": failed.route_id,
        "parent_decision_id": f"sensitive-parity-parent-{location}",
        "trusted_task": task,
        "trusted_routes": {
            failed.route_id: failed,
            selected.route_id: selected,
        },
        "trusted_prior_route": failed,
    }
    optimized_values: dict[str, object] = {
        "trusted_task": task,
        "task_id": task.task_id,
        "attempt_id": f"sensitive-parity-attempt-{location}",
        "failed_route": failed,
        "classification": trigger,
        "candidates": (candidate,),
        "decision_id": f"sensitive-parity-decision-{location}",
        "parent_decision_id": f"sensitive-parity-parent-{location}",
        "created_at": "2026-07-27T17:00:01Z",
        "policy_version": task.policy_version,
        "router_version": "router/pure-v1",
        "capacity_view_id": f"capacity-view:sensitive-parity-{location}",
        "effort": task.effort,
        "verification": "V0",
    }

    if location in {
        "decision_id",
        "attempt_id",
        "parent_decision_id",
        "created_at",
        "router_version",
        "capacity_view_id",
    }:
        public_values[location] = secret
        optimized_values[location] = secret
    elif location == "task_id":
        object.__setattr__(task, "task_id", secret)
        public_values["task_id"] = secret
        optimized_values["task_id"] = secret
    elif location == "policy_version":
        object.__setattr__(task, "policy_version", secret)
        public_values["policy_version"] = secret
        optimized_values["policy_version"] = secret
    elif location == "trusted_task":
        object.__setattr__(task, "objective", secret)
    elif location == "classification":
        object.__setattr__(trigger, "source", secret)
    elif location == "candidate":
        object.__setattr__(selected, "model", secret)
    elif location == "recheck_evidence":
        rejected = CandidateEvaluation(selected, False, ("capacity",))
        public_values.update(
            fallback=False,
            relation=DecisionRelation.WAITING,
            candidates=(rejected,),
            selected_route_id=None,
            reason_codes=("no_eligible_routes",),
            quality_compensation_plan=None,
            recheck_evidence=(secret,),
        )
        optimized_values.update(
            candidates=(rejected,),
            quality_compensation_plan=None,
            recheck_evidence=(secret,),
        )
    elif location == "trusted_reviewer_routes":
        public_values[location] = {secret: selected}
        optimized_values[location] = {secret: selected}
    elif location == "trusted_human_approval_refs":
        public_values[location] = {secret: task.task_id}
        optimized_values[location] = {secret: task.task_id}
    elif location == "trusted_execution_routes":
        public_values[location] = {secret: selected}
        optimized_values[location] = {secret: selected}
    elif location == "trusted_execution_evidence_key":
        authority = {"trusted_execution_evidence": {secret: ("evidence:test",)}}
        public_values.update(authority)
        optimized_values.update(authority)
    elif location == "trusted_execution_evidence_value":
        authority = {"trusted_execution_evidence": {"execution": (secret,)}}
        public_values.update(authority)
        optimized_values.update(authority)
    elif location == "trusted_evidence_refs":
        public_values[location] = (secret,)
        optimized_values[location] = (secret,)
    elif location == "trusted_threshold_results":
        public_values[location] = {secret: True}
        optimized_values[location] = {secret: True}
    else:
        raise AssertionError(f"unhandled sensitive parity location: {location}")

    with pytest.raises(DomainValidationError) as public_error:
        RouteDecisionV1(**public_values)  # type: ignore[arg-type]
    with pytest.raises(DomainValidationError) as optimized_error:
        replan_after_capacity_exhaustion(**optimized_values)  # type: ignore[arg-type]

    assert public_error.value.code == "decision.sensitive_field_prohibited"
    assert optimized_error.value.code == public_error.value.code

def test_capacity_replan_secret_bearing_quality_plan_matches_public_safe_downgrade():
    failed = route(
        model="sensitive-plan-parity-failed",
        billing_pool_id="sensitive-plan-parity-failed-billing",
        quota_pool_id="sensitive-plan-parity-failed-quota",
    )
    selected = route(
        model="sensitive-plan-parity-selected",
        billing_pool_id="sensitive-plan-parity-selected-billing",
        quota_pool_id="sensitive-plan-parity-selected-quota",
    )
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="sensitive-plan-parity-task",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    trigger = RuntimeErrorClassificationV1(
        kind=ErrorKind.CAPACITY_EXHAUSTED,
        source="typed-runtime-error",
        attempted_route_id=failed.route_id,
        quota_pool_id=failed.quota_pool_id,
        billing_pool_id=failed.billing_pool_id,
        classified_at="2026-07-27T17:00:00Z",
    )
    candidate = CandidateEvaluation(
        selected,
        True,
        score=1,
        score_factors=("quality",),
    )
    plan = valid_plan(
        "sensitive-plan-parity-decision",
        failed,
        selected,
        reviewed_execution_id="sensitive-plan-parity-attempt",
    )
    object.__setattr__(plan, "plan_id", "api_key=wave3a2-secret")

    ordinary = RouteDecisionV1(
        decision_id="sensitive-plan-parity-decision",
        task_id=task.task_id,
        attempt_id="sensitive-plan-parity-attempt",
        created_at="2026-07-27T17:00:01Z",
        policy_version=task.policy_version,
        router_version="router/pure-v1",
        capacity_view_id="capacity-view:sensitive-plan-parity",
        effort=task.effort,
        verification="V0",
        fallback=True,
        relation=DecisionRelation.FALLBACK,
        candidates=(candidate,),
        selected_route_id=selected.route_id,
        trigger=trigger,
        reason_codes=("route_capacity_exhausted",),
        prior_route_id=failed.route_id,
        parent_decision_id="sensitive-plan-parity-parent",
        quality_compensation_plan=plan,
        trusted_task=task,
        trusted_routes={
            failed.route_id: failed,
            selected.route_id: selected,
        },
        trusted_prior_route=failed,
    )
    optimized = replan_after_capacity_exhaustion(
        trusted_task=task,
        task_id=task.task_id,
        attempt_id="sensitive-plan-parity-attempt",
        failed_route=failed,
        classification=trigger,
        candidates=(candidate,),
        decision_id="sensitive-plan-parity-decision",
        parent_decision_id="sensitive-plan-parity-parent",
        created_at="2026-07-27T17:00:01Z",
        policy_version=task.policy_version,
        router_version="router/pure-v1",
        capacity_view_id="capacity-view:sensitive-plan-parity",
        effort=task.effort,
        verification="V0",
        quality_compensation_plan=plan,
    )

    assert ordinary.quality_compensation_plan is None
    assert optimized == ordinary
    assert canonical_dataclass_json(optimized) == canonical_dataclass_json(ordinary)

@pytest.mark.parametrize(
    ("forged_target", "expected_code"),
    (
        ("trusted_task", "task.verification_invalid"),
        ("candidate_route", "route.endpoint_invalid"),
        ("candidate_score", "candidate.invalid"),
    ),
)
def test_capacity_replan_revalidates_forged_typed_dtos(
    forged_target: str,
    expected_code: str,
):
    failed = route(
        model=f"replan-forged-failed-{forged_target}",
        quota_pool_id=f"replan-forged-failed-{forged_target}",
    )
    alternate = route(
        model=f"replan-forged-alternate-{forged_target}",
        quota_pool_id=f"replan-forged-alternate-{forged_target}",
    )
    trusted_task = TaskEnvelope.from_mapping(
        task_payload(
            task_id=f"task-replan-forged-{forged_target}",
            verification={
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
        )
    )
    candidate = CandidateEvaluation(
        alternate,
        True,
        score=1,
        score_factors=("quality",),
    )
    if forged_target == "trusted_task":
        object.__setattr__(trusted_task.verification, "minimum", "V9")
    elif forged_target == "candidate_route":
        object.__setattr__(alternate, "endpoint", "not-an-absolute-url")
    else:
        object.__setattr__(candidate, "score", float("nan"))

    with pytest.raises(DomainValidationError) as exc:
        replan_after_capacity_exhaustion(
            trusted_task=trusted_task,
            task_id=trusted_task.task_id,
            attempt_id=f"attempt-replan-forged-{forged_target}",
            failed_route=failed,
            classification=classification(failed),
            candidates=(candidate,),
            decision_id=f"decision-replan-forged-{forged_target}",
            parent_decision_id=f"parent-replan-forged-{forged_target}",
            created_at="2026-07-27T17:00:00Z",
            policy_version=trusted_task.policy_version,
            router_version="router/pure-v1",
            capacity_view_id=f"capacity-view:replan-forged-{forged_target}",
            effort=trusted_task.effort,
            verification="V0",
        )

    assert exc.value.code == expected_code

def test_capacity_exhaustion_rejects_mismatched_billing_pool_scope():
    attempted = route(billing_pool_id="billing-a", quota_pool_id="quota-a")
    with pytest.raises(DomainValidationError, match="classification.capacity_scope_required"):
        replan(
            task_id="task-1",
            attempt_id="attempt-1",
            decision_id="decision-2",
            failed_route=attempted,
            classification=classification(attempted, billing_pool_id="billing-b"),
            candidates=(),
            recheck_evidence=("pool evidence",),
        )

def test_capacity_exhaustion_does_not_fallback_within_same_quota_pool():
    failed = route(model="gpt-a", quota_pool_id="shared-pool")
    same_pool = route(model="gpt-b", quota_pool_id="shared-pool")
    decision = replan(
        task_id="task-1",
        attempt_id="attempt-1",
        decision_id="decision-2",
        failed_route=failed,
        classification=classification(failed),
        candidates=(CandidateEvaluation(same_pool, True),),
        recheck_evidence=("shared quota pool exhausted",),
    )
    assert decision.relation is DecisionRelation.WAITING
    assert decision.policy_status == "WAITING_FOR_CAPACITY"
    assert decision.dispatchable is False
    assert all(not candidate.eligible for candidate in decision.candidates)
    assert decision.candidates[0].rejection_codes == ("same_exhausted_quota_pool",)

def test_wait_requires_evidence_when_no_alternate_is_eligible():
    attempted = route()
    with pytest.raises(DomainValidationError, match="replan.recheck_evidence_required"):
        replan(
            task_id="task-1",
            attempt_id="attempt-1",
            decision_id="decision-2",
            failed_route=attempted,
            classification=classification(attempted),
            candidates=(CandidateEvaluation(attempted, False, ("policy_rejected",)),),
        )

def test_unscoped_or_mismatched_capacity_classification_is_rejected():
    attempted = route()
    with pytest.raises(DomainValidationError, match="classification.capacity_scope_required"):
        RuntimeErrorClassificationV1(
            kind=ErrorKind.CAPACITY_EXHAUSTED,
            source="typed-runtime-error",
            attempted_route_id="",
            quota_pool_id=attempted.quota_pool_id,
            classified_at="2026-07-26T17:00:00Z",
        )
    with pytest.raises(DomainValidationError, match="classification.capacity_scope_required"):
        replan(
            task_id="task-1",
            attempt_id="attempt-1",
            decision_id="decision-2",
            failed_route=attempted,
            classification=classification(attempted, "other-pool"),
            candidates=(),
        )

def test_runtime_classification_requires_route_pool_scope_not_provider_labels():
    attempted = route()
    classification_value = RuntimeErrorClassificationV1.from_mapping(
        {
            "kind": "capacity_exhausted",
            "attempted_route_id": attempted.route_id,
            "quota_pool_id": attempted.quota_pool_id,
            "billing_pool_id": attempted.billing_pool_id,
            "source": "typed-runtime-error",
            "classified_at": "2026-07-26T17:00:00Z",
        }
    )
    assert classification_value.source == "typed-runtime-error"
    assert classification_value.classified_at.endswith("Z")

    with pytest.raises(DomainValidationError, match="classification.capacity_scope_required"):
        RuntimeErrorClassificationV1.from_mapping(
            {
                "kind": "capacity_exhausted",
                "provider": "openai",
                "model": "gpt-5",
            }
        )

def test_capacity_observation_uses_exact_canonical_v1_wire_fields():
    observation = CapacityObservationV1.from_mapping({
        "observation_id": "obs-contract",
        "route_id": route().route_id, "quota_pool_id": "pool-contract", "billing_pool_id": "billing-contract",
        "metric": "remaining", "value": 0, "unit": "requests", "source": "provider",
        "captured_at": "2026-07-26T17:00:00Z", "valid_until": "2026-07-26T18:00:00Z",
        "freshness": "fresh", "confidence": 0.9, "is_estimated": False, "plan": "team",
        "entitlement": "paid", "reset_at": "2026-07-26T19:00:00Z", "price": 0,
        "max_concurrency": 1, "reason_code": "provider_remaining",
    })
    assert observation.metric == "remaining" and observation.value == 0
    assert tuple(field_.name for field_ in fields(CapacityObservationV1)) == (
        "observation_id", "quota_pool_id", "route_id", "billing_pool_id",
        "metric", "value", "unit", "source", "captured_at", "valid_until",
        "freshness", "confidence", "is_estimated", "plan", "entitlement",
        "reset_at", "price", "max_concurrency", "reason_code",
    )
    assert CapacityObservationV1.SCHEMA_VERSION == "capacity-observation/v1"
    assert "schema_version" not in asdict(observation)
    with pytest.raises(DomainValidationError, match="capacity.schema_invalid"):
        CapacityObservationV1.from_mapping({
            **asdict(observation),
            "schema_version": "capacity-observation/v1",
        })
    with pytest.raises(DomainValidationError, match="capacity.schema_invalid"):
        CapacityObservationV1.from_mapping({**asdict(observation), "dimensions": []})
    with pytest.raises(DomainValidationError, match="capacity.confidence_invalid"):
        CapacityObservationV1.from_mapping({**asdict(observation), "confidence": 1.1})
    with pytest.raises(DomainValidationError, match="capacity.timestamp_order_invalid"):
        CapacityObservationV1.from_mapping({**asdict(observation), "valid_until": observation.captured_at})

@pytest.mark.parametrize("observation", (
    capacity_observation(freshness="stale"),
    capacity_observation(freshness="unknown"),
    capacity_observation(valid_until="2026-07-26T17:20:00Z"),
))
def test_stale_unknown_and_expired_observations_never_become_healthy(observation: CapacityObservationV1):
    pool = capacity_view(observation).get_pool("pool-a")
    assert pool is not None
    assert pool.derived_remaining is None
    assert pool.is_unknown is True
    assert pool.is_healthy is False

def test_capacity_view_never_compares_incompatible_units():
    daily = capacity_observation(observation_id="daily", source="daily", value=100, unit="requests/day")
    per_minute = capacity_observation(observation_id="minute", source="minute", value=1, unit="requests/min")
    pool = capacity_view(daily, per_minute).get_pool("pool-a")
    assert pool is not None
    assert pool.derived_remaining is None
    assert pool.conflict_code == "incompatible_unit"
    assert pool.is_healthy is False

def test_authoritative_only_requires_external_registry_and_is_input_order_independent():
    untrusted = capacity_observation(observation_id="untrusted", source="untrusted", value=999)
    trusted = capacity_observation(observation_id="trusted", source="trusted", value=1)
    with pytest.raises(DomainValidationError, match="capacity.authority_required"):
        capacity_view(untrusted, trusted, conflict_disposition=CapacityConflictDisposition.AUTHORITATIVE_ONLY)
    kwargs = {"conflict_disposition": CapacityConflictDisposition.AUTHORITATIVE_ONLY, "authoritative_sources": {"pool-a": "trusted"}}
    forward = capacity_view(untrusted, trusted, **kwargs).get_pool("pool-a")
    reverse = capacity_view(trusted, untrusted, **kwargs).get_pool("pool-a")
    assert forward is not None and reverse is not None
    assert forward.derived_remaining == reverse.derived_remaining == 1
    assert forward.conflict_code is reverse.conflict_code is None
    assert forward.is_healthy is reverse.is_healthy is True

def test_capacity_view_uses_the_exact_contract_wire_and_observation_validity():
    observation = capacity_observation()
    view = capacity_view(observation)

    assert set(view.canonical_object()) == {
        "view_id",
        "built_at",
        "pools",
        "source_observation_ids",
    }
    assert CapacityViewV1.from_mapping(view.canonical_object()).canonical_object() == view.canonical_object()
    assert CapacityViewV1.SCHEMA_VERSION == "capacity-view/v1"
    for extra in ("schema_version", "expires_at"):
        with pytest.raises(DomainValidationError, match="capacity.schema_invalid"):
            CapacityViewV1.from_mapping(
                {
                    **view.canonical_object(),
                    extra: (
                        "capacity-view/v1"
                        if extra == "schema_version"
                        else "2026-07-26T19:00:00Z"
                    ),
                }
            )

def test_pool_capacity_set_like_provenance_is_canonical_direct_and_from_mapping():
    dimensions = {
        "request": CapacityValueV1(known=True, value=3, unit="requests"),
        "token": CapacityValueV1(known=False),
        "concurrency": CapacityValueV1(known=False),
    }
    forward_values = {
        "schema_version": "capacity-pool/v1",
        "quota_pool_id": "pool-canonical",
        **dimensions,
        "stale_sources": ("provider-z", "provider-a"),
        "unknown_sources": ("provider-y", "provider-b"),
        "source_observation_ids": ("observation-9", "observation-1"),
    }
    reverse_values = {
        **forward_values,
        "stale_sources": tuple(reversed(forward_values["stale_sources"])),
        "unknown_sources": tuple(reversed(forward_values["unknown_sources"])),
        "source_observation_ids": tuple(
            reversed(forward_values["source_observation_ids"])
        ),
    }

    direct_forward = PoolCapacityV1(**forward_values)  # type: ignore[arg-type]
    direct_reverse = PoolCapacityV1(**reverse_values)  # type: ignore[arg-type]
    mapped_forward = PoolCapacityV1.from_mapping(forward_values)
    mapped_reverse = PoolCapacityV1.from_mapping(reverse_values)

    assert direct_forward == direct_reverse == mapped_forward == mapped_reverse
    assert len(
        {
            canonical_dataclass_json(pool)
            for pool in (
                direct_forward,
                direct_reverse,
                mapped_forward,
                mapped_reverse,
            )
        }
    ) == 1

@pytest.mark.parametrize(
    "field_name",
    ("stale_sources", "unknown_sources", "source_observation_ids"),
)
@pytest.mark.parametrize("from_mapping", (False, True))
def test_pool_capacity_rejects_duplicate_set_like_provenance(
    field_name: str,
    from_mapping: bool,
):
    values: dict[str, object] = {
        "schema_version": "capacity-pool/v1",
        "quota_pool_id": "pool-duplicate-provenance",
        "request": CapacityValueV1(known=True, value=3, unit="requests"),
        "token": CapacityValueV1(known=False),
        "concurrency": CapacityValueV1(known=False),
        field_name: ("duplicate", "duplicate"),
    }

    with pytest.raises(DomainValidationError) as error:
        if from_mapping:
            PoolCapacityV1.from_mapping(values)
        else:
            PoolCapacityV1(**values)  # type: ignore[arg-type]

    assert error.value.code == "capacity.provenance_duplicate"

def test_capacity_view_set_like_provenance_is_canonical_direct_and_from_mapping():
    pool = PoolCapacityV1(
        schema_version="capacity-pool/v1",
        quota_pool_id="pool-view-canonical",
        request=CapacityValueV1(known=True, value=3, unit="requests"),
        token=CapacityValueV1(known=False),
        concurrency=CapacityValueV1(known=False),
        source_observation_ids=("observation-2", "observation-1"),
    )
    forward_values = {
        "view_id": "view-canonical",
        "built_at": "2026-07-26T17:30:00Z",
        "pools": (pool,),
        "source_observation_ids": ("observation-z", "observation-a"),
    }
    reverse_values = {
        **forward_values,
        "source_observation_ids": tuple(
            reversed(forward_values["source_observation_ids"])
        ),
    }

    direct_forward = CapacityViewV1(**forward_values)  # type: ignore[arg-type]
    direct_reverse = CapacityViewV1(**reverse_values)  # type: ignore[arg-type]
    mapped_forward = CapacityViewV1.from_mapping(forward_values)
    mapped_reverse = CapacityViewV1.from_mapping(reverse_values)

    assert direct_forward == direct_reverse == mapped_forward == mapped_reverse
    assert direct_forward.canonical_json == direct_reverse.canonical_json
    assert mapped_forward.canonical_json == mapped_reverse.canonical_json
    assert canonical_dataclass_json(direct_forward) == canonical_dataclass_json(
        mapped_reverse
    )

@pytest.mark.parametrize("from_mapping", (False, True))
def test_capacity_view_rejects_duplicate_set_like_provenance(from_mapping: bool):
    values: dict[str, object] = {
        "view_id": "view-duplicate-provenance",
        "built_at": "2026-07-26T17:30:00Z",
        "pools": (),
        "source_observation_ids": ("duplicate", "duplicate"),
    }

    with pytest.raises(DomainValidationError) as error:
        if from_mapping:
            CapacityViewV1.from_mapping(values)
        else:
            CapacityViewV1(**values)  # type: ignore[arg-type]

    assert error.value.code == "capacity.provenance_duplicate"

@pytest.mark.parametrize(
    ("observation", "expected_conflict"),
    (
        (
            capacity_observation(source="trusted", unit="requests/day"),
            "incompatible_unit",
        ),
        (
            capacity_observation(source="trusted", freshness="stale"),
            "authoritative_source_unavailable",
        ),
        (
            capacity_observation(
                source="trusted",
                freshness="unknown",
                value=None,
                unit=None,
            ),
            "authoritative_source_unavailable",
        ),
    ),
)
def test_authoritative_unavailable_or_incompatible_capacity_is_explicit_unknown(
    observation: CapacityObservationV1,
    expected_conflict: str,
):
    pool = capacity_view(
        observation,
        conflict_disposition=CapacityConflictDisposition.AUTHORITATIVE_ONLY,
        authoritative_sources={"pool-a": "trusted"},
    ).get_pool("pool-a")

    assert pool is not None
    assert pool.derived_remaining is None
    assert pool.is_unknown is True
    assert pool.conflict_code == expected_conflict

def test_divergent_compatible_authoritative_values_are_unknown_not_an_exception():
    pool = capacity_view(
        capacity_observation(observation_id="trusted-a", source="trusted", value=7),
        capacity_observation(observation_id="trusted-b", source="trusted", value=9),
        conflict_disposition=CapacityConflictDisposition.AUTHORITATIVE_ONLY,
        authoritative_sources={"pool-a": "trusted"},
    ).get_pool("pool-a")

    assert pool is not None
    assert pool.derived_remaining is None
    assert pool.conflict_code == "conflicting_remaining_values"

def test_future_capacity_observation_is_unknown_not_healthy():
    future = capacity_observation(
        captured_at="2026-07-26T18:00:00Z",
        valid_until="2026-07-26T19:00:00Z",
    )

    pool = capacity_view(future).get_pool("pool-a")

    assert pool is not None
    assert pool.derived_remaining is None
    assert pool.is_unknown is True
    assert pool.is_healthy is False

def test_authoritative_only_never_uses_an_unregistered_source_when_authority_is_absent():
    attacker = capacity_observation(source="attacker", value=999)

    pool = capacity_view(
        attacker,
        conflict_disposition=CapacityConflictDisposition.AUTHORITATIVE_ONLY,
        authoritative_sources={"pool-a": "trusted"},
    ).get_pool("pool-a")

    assert pool is not None
    assert pool.derived_remaining is None
    assert pool.is_unknown is True
    assert pool.is_healthy is False
    assert pool.conflict_code is not None
    assert "attacker" in pool.unknown_sources

def test_authoritative_registry_missing_observation_emits_unknown_pool():
    view = capacity_view(
        conflict_disposition=CapacityConflictDisposition.AUTHORITATIVE_ONLY,
        authoritative_sources={"registered-pool": "trusted-source"},
    )
    pool = view.get_pool("registered-pool")

    assert pool is not None
    assert pool.derived_remaining is None
    assert pool.is_healthy is False
    assert pool.conflict_code == "authoritative_source_unavailable"

def test_forged_reservation_and_breaker_cycles_fail_without_recursing():
    bound_route = route(model="structural-cycle", quota_pool_id="structural-cycle")
    reservation = capacity_reservation(bound_route)
    object.__setattr__(reservation, "unit", reservation)
    with pytest.raises(DomainValidationError):
        replace(reservation, trusted_route=bound_route)

    breaker = orchestration.CircuitBreakerV1.from_mapping(
        breaker_payload(bound_route),
        trusted_route=bound_route,
    )
    object.__setattr__(breaker, "reason_code", breaker)
    with pytest.raises(DomainValidationError):
        replace(breaker, trusted_route=bound_route)

def test_capacity_pool_identity_canonicalization_prevents_split_exhaustion():
    available = capacity_observation(
        observation_id="available",
        quota_pool_id=" team-a ",
        value=10,
    )
    exhausted = capacity_observation(
        observation_id="exhausted",
        quota_pool_id="TEAM-A",
        value=0,
    )

    forward = capacity_view(available, exhausted)
    reverse = capacity_view(exhausted, available)

    assert available.quota_pool_id == exhausted.quota_pool_id == "team-a"
    assert len(forward.pools) == 1
    assert forward.canonical_json == reverse.canonical_json
    pool = forward.get_pool(" TEAM-A ")
    assert pool is not None
    assert pool.derived_remaining is None
    assert pool.is_healthy is False
    assert pool.conflict_code == "conflicting_remaining_values"

def test_capacity_pool_identity_canonicalizes_billing_authority_and_bindings():
    observed = CapacityObservationV1.from_mapping(
        {
            **asdict(capacity_observation(source="trusted", value=0)),
            "quota_pool_id": " TEAM-A ",
            "billing_pool_id": " BILLING-A ",
        }
    )
    authoritative = capacity_view(
        observed,
        conflict_disposition=CapacityConflictDisposition.AUTHORITATIVE_ONLY,
        authoritative_sources={" team-A ": "trusted"},
    )
    assert observed.quota_pool_id == "team-a"
    assert observed.billing_pool_id == "billing-a"
    assert authoritative.get_pool("TEAM-A").is_exhausted is True  # type: ignore[union-attr]

    bound_route = route(
        model="canonical-capacity-binding",
        quota_pool_id="team-a",
        billing_pool_id="billing-a",
    )
    reservation = orchestration.CapacityReservationV1.from_mapping(
        {
            **reservation_payload(bound_route),
            "quota_pool_id": " TEAM-A ",
            "billing_pool_id": " BILLING-A ",
        },
        trusted_route=bound_route,
    )
    breaker = orchestration.CircuitBreakerV1.from_mapping(
        {
            **breaker_payload(bound_route),
            "quota_pool_id": " TEAM-A ",
        },
        trusted_route=bound_route,
    )
    assert reservation.quota_pool_id == breaker.quota_pool_id == "team-a"
    assert reservation.billing_pool_id == "billing-a"

def test_authoritative_pool_registry_rejects_canonical_identity_collisions():
    with pytest.raises(DomainValidationError) as error:
        capacity_view(
            conflict_disposition=CapacityConflictDisposition.AUTHORITATIVE_ONLY,
            authoritative_sources={
                "team-a": "trusted-primary",
                " TEAM-A ": "trusted-secondary",
            },
        )
    assert error.value.code == "capacity.authority_required"
