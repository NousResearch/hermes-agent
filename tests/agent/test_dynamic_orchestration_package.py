from __future__ import annotations

from dataclasses import asdict

from agent import dynamic_orchestration as orchestration


PUBLIC_API = frozenset(
    {
        "AcceptanceThresholdV1",
        "AttemptState",
        "AttemptStateEvent",
        "AuditedModelJustification",
        "CAPACITY_OBSERVATION_SCHEMA_VERSION",
        "CAPACITY_POOL_SCHEMA_VERSION",
        "CAPACITY_VIEW_SCHEMA_VERSION",
        "CandidateEvaluation",
        "CandidateScoreV1",
        "CapacityCompleteness",
        "CapacityConfidence",
        "CapacityConflictDisposition",
        "CapacityDimension",
        "CapacityFreshness",
        "CapacityMetric",
        "CapacityObservationV1",
        "CapacityReservationV1",
        "CapacityValueV1",
        "CapacityViewV1",
        "CircuitBreakerStatus",
        "CircuitBreakerV1",
        "CompensationEscalationV1",
        "CredentialState",
        "CredentialStateEvent",
        "DecisionRelation",
        "DomainValidationError",
        "E",
        "EXECUTION_ENVELOPE_SCHEMA_VERSION",
        "EligibilityDisposition",
        "ErrorKind",
        "EscalationAction",
        "ExecutionDispatchState",
        "ExecutionEnvelopeV1",
        "ExecutionIdentityAttestationV1",
        "ExecutionOutcome",
        "ExecutionOutcomeV1",
        "ExecutionVerificationAttestationV1",
        "IndependentReviewAttestationV1",
        "InitialSelectionTriggerV1",
        "PoolCapacityV1",
        "QUALITY_COMPENSATION_SCHEMA_VERSION",
        "QualityCompensationPlanV1",
        "ROUTE_CANONICALIZATION_VERSION",
        "ReservationState",
        "ReservationStateEvent",
        "ReviewState",
        "ReviewStateEvent",
        "RouteDecisionV1",
        "RouteEligibilityFactsV1",
        "RouteState",
        "RouteStateEvent",
        "RouteV1",
        "RuntimeErrorClassificationV1",
        "S",
        "TaskBudgetV1",
        "TaskContextV1",
        "TaskEnvelope",
        "TaskPrivacyV1",
        "TaskRiskV1",
        "TaskState",
        "TaskStateEvent",
        "TaskVerificationV1",
        "build_capacity_view",
        "evaluate_route_eligibility",
        "replan_after_capacity_exhaustion",
        "score_eligible_candidates",
        "validate_attempt_transition",
        "validate_credential_transition",
        "validate_reservation_transition",
        "validate_review_transition",
        "validate_route_transition",
        "validate_task_transition",
    }
)


def _route() -> orchestration.RouteV1:
    return orchestration.RouteV1.from_mapping(
        {
            "provider": "OpenAI",
            "product": "ChatGPT",
            "surface": "API",
            "account_id": "package-test",
            "billing_pool_id": "package-billing",
            "quota_pool_id": "package-quota",
            "model": "GPT-5",
            "endpoint": "https://api.example.com/v1",
            "region": "US",
        }
    )


def _task() -> orchestration.TaskEnvelope:
    return orchestration.TaskEnvelope.from_mapping(
        {
            "schema_version": "task-envelope/v1",
            "task_id": "package-task",
            "objective": "validate package compatibility",
            "deliverables": ["compatibility evidence"],
            "capabilities_required": ["filesystem.write"],
            "tools_allowed": ["patch"],
            "permissions_required": ["repository.write"],
            "context": {
                "classification": "internal",
                "max_tokens": 900,
                "token_count": 12,
                "allowed_sources": ["repository"],
            },
            "privacy": {
                "classification": "internal",
                "outbound_allowed": False,
                "retention": "ephemeral",
            },
            "risk": {
                "level": "medium",
                "reversibility": "reversible",
                "impact": "repository-only",
            },
            "effort": "E2",
            "budget": {
                "currency": "USD",
                "paid_allowed": False,
                "soft_cap": 0,
                "hard_cap": 0,
            },
            "verification": {
                "minimum": "V0",
                "independent_required": False,
                "human_gate_required": False,
            },
            "policy_version": "policy/v1",
        }
    )


def test_package_root_preserves_manifest_imports_and_private_compatibility() -> None:
    assert frozenset(orchestration.__all__) == PUBLIC_API
    assert all(not name.startswith("_") for name in orchestration.__all__)
    assert all(getattr(orchestration, name) is not None for name in PUBLIC_API)
    assert orchestration._MAX_CONTEXT_TOKENS == 2**31 - 1
    assert callable(orchestration._safe_asdict)


def test_package_root_supports_route_task_decision_capacity_and_execution_flow() -> None:
    route = _route()
    task = _task()
    candidate = orchestration.CandidateEvaluation(
        route,
        True,
        score=1.0,
        score_factors=("quality",),
    )
    decision = orchestration.RouteDecisionV1(
        decision_id="package-decision",
        task_id=task.task_id,
        attempt_id="package-attempt",
        created_at="2026-07-27T12:00:00Z",
        policy_version=task.policy_version,
        router_version="router/pure-v1",
        capacity_view_id="package-view",
        effort=task.effort,
        verification="V0",
        fallback=False,
        relation=orchestration.DecisionRelation.INITIAL,
        candidates=(candidate,),
        selected_route_id=route.route_id,
        trigger=orchestration.InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy",
            "2026-07-27T12:00:00Z",
        ),
        reason_codes=("initial_selection",),
        trusted_task=task,
        trusted_routes={route.route_id: route},
    )
    observation = orchestration.CapacityObservationV1(
        observation_id="package-observation",
        route_id=route.route_id,
        quota_pool_id=route.quota_pool_id,
        billing_pool_id=route.billing_pool_id,
        metric="remaining",
        value=10,
        unit="requests",
        source="provider",
        captured_at="2026-07-27T12:00:00Z",
        valid_until="2026-07-27T13:00:00Z",
        freshness="fresh",
        confidence=0.9,
        is_estimated=False,
    )
    capacity = orchestration.build_capacity_view(
        view_id=decision.capacity_view_id,
        built_at="2026-07-27T12:30:00Z",
        observations=(observation,),
        conflict_disposition=orchestration.CapacityConflictDisposition.EXCLUDE,
        canonical_units={"remaining": "requests"},
    )
    execution = orchestration.ExecutionEnvelopeV1.from_mapping(
        {
            "schema_version": "execution-envelope/v1",
            "execution_id": "package-execution",
            "task_id": task.task_id,
            "attempt_id": decision.attempt_id,
            "decision_id": decision.decision_id,
            "route_id": route.route_id,
            "reservation_id": None,
            "effort": task.effort,
            "verification": decision.verification,
            "workdir": None,
            "allowed_context_refs": (),
            "allowed_tools": (),
            "permissions": (),
            "timeout": 30,
            "budget": asdict(task.budget),
            "trace_context": None,
            "telemetry_consent_id": None,
            "requested_identity": route.route_id,
            "resolved_identity": route.route_id,
            "effective_identity": route.route_id,
            "dispatch_state": "PENDING",
            "result_evidence_refs": (),
            "normalized_error": None,
            "reconciliation_state": "pending",
        },
        trusted_task=task,
        trusted_decision=decision,
        trusted_routes={route.route_id: route},
    )

    assert decision.selected_route_id == route.route_id
    assert capacity.pools[0].quota_pool_id == route.quota_pool_id
    assert execution.route_id == route.route_id
    assert execution.dispatch_state is orchestration.ExecutionDispatchState.PENDING
