from __future__ import annotations

from tests.agent._dynamic_orchestration_support import (
    CandidateEvaluation,
    DecisionRelation,
    DomainValidationError,
    ExecutionDispatchState,
    ExecutionEnvelopeV1,
    ExecutionOutcomeV1,
    InitialSelectionTriggerV1,
    RouteDecisionV1,
    TaskEnvelope,
    asdict,
    capacity_reservation,
    classification,
    execution_authority,
    execution_payload,
    pytest,
    replan,
    route,
    scored_candidate,
    task_payload,
    valid_plan,
)
def test_execution_envelope_is_canonical_and_rehydrates_trusted_authorities():
    task, decision, routes = execution_authority()
    payload = execution_payload(task, decision)
    envelope = ExecutionEnvelopeV1.from_mapping(payload, trusted_task=task, trusted_decision=decision, trusted_routes=routes)
    assert envelope.outcome is None
    assert envelope.activation_status == "ACTIVATION_BLOCKED_EXTERNAL_AUTHORITY_UNAVAILABLE"
    assert set(envelope.canonical_object()) == set(payload)
    assert "activation_status" not in envelope.canonical_object()
    with pytest.raises(DomainValidationError, match="execution.schema_invalid"):
        ExecutionEnvelopeV1.from_mapping({**payload, "identity": {}}, trusted_task=task, trusted_decision=decision, trusted_routes=routes)
    with pytest.raises(DomainValidationError, match="execution.trusted_task_required"):
        ExecutionEnvelopeV1.from_mapping(payload)

def test_execution_rejects_noncanonical_routes_and_impossible_states():
    task, decision, routes = execution_authority()
    payload = execution_payload(task, decision)
    with pytest.raises(DomainValidationError, match="route.identity_invalid"):
        ExecutionEnvelopeV1.from_mapping({**payload, "route_id": "not-a-route"}, trusted_task=task, trusted_decision=decision, trusted_routes=routes)
    with pytest.raises(DomainValidationError, match="execution.reservation_state_invalid"):
        ExecutionEnvelopeV1.from_mapping({**payload, "reservation_id": "reservation-1"}, trusted_task=task, trusted_decision=decision, trusted_routes=routes)
    with pytest.raises(DomainValidationError, match="execution.external_authority_unavailable"):
        ExecutionEnvelopeV1.from_mapping({**payload, "dispatch_state": "COMPLETED", "reservation_id": "reservation-1", "normalized_error": "must-not-succeed"}, trusted_task=task, trusted_decision=decision, trusted_routes=routes)
    with pytest.raises(DomainValidationError, match="execution.external_authority_unavailable"):
        ExecutionEnvelopeV1.from_mapping({**payload, "dispatch_state": "FAILED", "reservation_id": "reservation-1"}, trusted_task=task, trusted_decision=decision, trusted_routes=routes)

def test_terminal_execution_is_rejected_without_an_external_authority_store():
    task, decision, routes = execution_authority(verification="V4", reservation_id="reservation-1")
    terminal = {**execution_payload(task, decision), "dispatch_state": "COMPLETED", "reservation_id": "reservation-1", "result_evidence_refs": ("evidence:human",)}
    with pytest.raises(DomainValidationError, match="execution.external_authority_unavailable"):
        ExecutionEnvelopeV1.from_mapping(
            terminal,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )

def test_invalid_outcome_mapping_has_stable_domain_error():
    with pytest.raises(DomainValidationError) as exc:
        ExecutionOutcomeV1.from_mapping({"outcome": "INVALID"})
    assert exc.value.code == "execution.outcome_invalid"

def test_execution_envelope_v1_has_no_caller_supplied_terminal_attestation_path_or_timestamps():
    task, decision, routes = execution_authority(verification="V4", reservation_id="reservation-1")
    terminal = {
        **execution_payload(task, decision),
        "dispatch_state": "COMPLETED",
        "reservation_id": "reservation-1",
        "result_evidence_refs": ("evidence:human",),
    }

    with pytest.raises(DomainValidationError, match="execution.schema_invalid"):
        ExecutionEnvelopeV1.from_mapping(
            {**terminal, "created_at": "2026-07-26T17:00:00Z"},
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )

    with pytest.raises(TypeError):
        ExecutionEnvelopeV1.from_mapping(
            terminal,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
            trusted_outcome_states={"exec-1": ExecutionDispatchState.COMPLETED},
            trusted_validator_evidence={"evidence:human": "V4"},
        )

def test_terminal_execution_cannot_be_reintroduced_with_result_evidence():
    task, decision, routes = execution_authority(verification="V4", reservation_id="reservation-1")
    terminal = {
        **execution_payload(task, decision),
        "dispatch_state": "COMPLETED",
        "reservation_id": "reservation-1",
        "result_evidence_refs": ("evidence:human",),
    }
    with pytest.raises(DomainValidationError, match="execution.external_authority_unavailable"):
        ExecutionEnvelopeV1.from_mapping(
            terminal,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )

def test_structural_reservation_never_authorizes_non_pending_execution():
    task, decision, routes = execution_authority(reservation_id="reservation-1")
    selected = next(iter(routes.values()))
    valid = capacity_reservation(
        selected,
        owner_attempt_id=decision.attempt_id,
    )
    dispatched_payload = {
        **execution_payload(task, decision),
        "dispatch_state": "DISPATCHED",
        "reservation_id": valid.reservation_id,
    }

    with pytest.raises(DomainValidationError) as missing_store:
        ExecutionEnvelopeV1.from_mapping(
            dispatched_payload,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )
    assert missing_store.value.code == "execution.external_authority_unavailable"

    mismatched_attempt = capacity_reservation(
        selected,
        owner_attempt_id="other-attempt",
    )
    with pytest.raises(TypeError):
        ExecutionEnvelopeV1.from_mapping(
            dispatched_payload,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
            trusted_reservations={valid.reservation_id: mismatched_attempt},
        )

@pytest.mark.parametrize(
    ("minimum", "independent_required", "human_gate_required"),
    (
        ("V0", False, False),
        ("V2", True, False),
        ("V4", True, True),
    ),
)
def test_pure_phase_route_decisions_never_mint_dispatch_authority(
    minimum: str,
    independent_required: bool,
    human_gate_required: bool,
):
    selected = route(
        model=f"authority-{minimum.casefold()}",
        quota_pool_id=f"authority-{minimum.casefold()}",
    )
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id=f"task-authority-{minimum.casefold()}",
            verification={
                "minimum": minimum,
                "independent_required": independent_required,
                "human_gate_required": human_gate_required,
            },
        )
    )
    decision = RouteDecisionV1(
        decision_id=f"decision-authority-{minimum.casefold()}",
        task_id=task.task_id,
        attempt_id=f"attempt-authority-{minimum.casefold()}",
        created_at="2026-07-27T12:00:00Z",
        policy_version=task.policy_version,
        router_version="router/pure-v1",
        capacity_view_id="view-pure-authority",
        effort=task.effort,
        verification=minimum,
        fallback=False,
        relation=DecisionRelation.INITIAL,
        candidates=(scored_candidate(selected),),
        selected_route_id=selected.route_id,
        trigger=InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy",
            "2026-07-27T12:00:00Z",
        ),
        reason_codes=("initial_selection",),
        trusted_task=task,
        trusted_routes={selected.route_id: selected},
        trusted_human_approval_refs={"approval:caller-created": task.task_id},
    )

    assert decision.policy_status != "AUTHORIZED"
    assert decision.dispatchable is False
    object.__setattr__(decision, "policy_status", "AUTHORIZED")
    assert decision.dispatchable is False

def test_caller_human_approval_map_cannot_elevate_activation_block():
    selected = route(model="human-gate", quota_pool_id="human-gate")
    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-human-gate-pure",
            verification={
                "minimum": "V1",
                "independent_required": False,
                "human_gate_required": True,
            },
        )
    )
    values = {
        "decision_id": "decision-human-gate-pure",
        "task_id": task.task_id,
        "attempt_id": "attempt-human-gate-pure",
        "created_at": "2026-07-27T12:00:00Z",
        "policy_version": task.policy_version,
        "router_version": "router/pure-v1",
        "capacity_view_id": "view-human-gate-pure",
        "effort": task.effort,
        "verification": "V1",
        "fallback": False,
        "relation": DecisionRelation.INITIAL,
        "candidates": (scored_candidate(selected),),
        "selected_route_id": selected.route_id,
        "trigger": InitialSelectionTriggerV1(
            "initial-selection-trigger/v1",
            "initial_selection",
            "policy",
            "2026-07-27T12:00:00Z",
        ),
        "reason_codes": ("initial_selection",),
        "trusted_task": task,
        "trusted_routes": {selected.route_id: selected},
    }
    blocked = RouteDecisionV1(**values)
    caller_approved = RouteDecisionV1(
        **values,
        trusted_human_approval_refs={"approval:forged": task.task_id},
    )

    assert blocked.policy_status == "ACTIVATION_BLOCKED_HUMAN_GATE"
    assert caller_approved.policy_status == blocked.policy_status
    assert caller_approved.dispatchable is False

@pytest.mark.parametrize(
    ("minimum", "independent_required", "human_gate_required"),
    (
        ("V2", False, False),
        ("V3", False, False),
        ("V4", True, False),
        ("V4", False, True),
    ),
)
def test_task_verification_rejects_contradictory_independence_and_gate_flags(
    minimum: str,
    independent_required: bool,
    human_gate_required: bool,
):
    with pytest.raises(DomainValidationError) as error:
        TaskEnvelope.from_mapping(
            task_payload(
                verification={
                    "minimum": minimum,
                    "independent_required": independent_required,
                    "human_gate_required": human_gate_required,
                }
            )
        )
    assert error.value.code == "task.verification_invariant"

def test_execution_cannot_claim_dispatch_from_arbitrary_reservation_string():
    task, decision, routes = execution_authority(
        verification="V0",
        reservation_id="caller-fabricated-reservation",
    )
    payload = {
        **execution_payload(task, decision),
        "dispatch_state": "DISPATCHED",
        "reservation_id": "caller-fabricated-reservation",
    }

    with pytest.raises(DomainValidationError) as error:
        ExecutionEnvelopeV1.from_mapping(
            payload,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )
    assert error.value.code == "execution.external_authority_unavailable"

def test_pure_phase_fallback_stays_activation_blocked_and_pending_boundary_is_explicit():
    failed = route(model="fallback-failed", quota_pool_id="fallback-failed")
    selected = route(model="fallback-selected", quota_pool_id="fallback-selected")
    plan = valid_plan(
        "decision-fallback-pure",
        failed,
        selected,
        reviewed_execution_id="attempt-fallback-pure",
    )
    decision = replan(
        task_id="task-fallback-pure",
        attempt_id="attempt-fallback-pure",
        decision_id="decision-fallback-pure",
        failed_route=failed,
        classification=classification(failed),
        candidates=(
            CandidateEvaluation(failed, False, ("failed_route",)),
            scored_candidate(selected),
        ),
        quality_compensation_plan=plan,
        task_verification_minimum="V2",
    )

    assert decision.policy_status == "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    assert decision.activation_block_reason == "quality_compensation_insufficient"
    assert decision.dispatchable is False

    task = TaskEnvelope.from_mapping(
        task_payload(
            task_id="task-fallback-pure",
            verification={
                "minimum": "V2",
                "independent_required": True,
                "human_gate_required": False,
            },
        )
    )
    routes = {
        failed.route_id: failed,
        selected.route_id: selected,
    }
    structurally_restored = RouteDecisionV1.from_mapping(
        asdict(decision),
        trusted_task=task,
        trusted_routes=routes,
    )
    assert structurally_restored.policy_status == (
        "ACTIVATION_BLOCKED_QUALITY_COMPENSATION"
    )
    assert "quality_compensation_insufficient" in structurally_restored.reason_codes
    envelope = ExecutionEnvelopeV1.from_mapping(
        execution_payload(task, decision),
        trusted_task=task,
        trusted_decision=decision,
        trusted_routes=routes,
    )
    assert envelope.dispatch_state is ExecutionDispatchState.PENDING
    assert envelope.activation_status == (
        "ACTIVATION_BLOCKED_EXTERNAL_AUTHORITY_UNAVAILABLE"
    )

@pytest.mark.parametrize(
    "dispatch_state",
    ("DISPATCHED", "COMPLETED", "FAILED", "CANCELLED"),
)
def test_pure_execution_contract_rejects_every_non_pending_state(
    dispatch_state: str,
):
    task, decision, routes = execution_authority()
    payload = {
        **execution_payload(task, decision),
        "dispatch_state": dispatch_state,
        "reservation_id": "caller-reservation",
    }
    with pytest.raises(DomainValidationError) as error:
        ExecutionEnvelopeV1.from_mapping(
            payload,
            trusted_task=task,
            trusted_decision=decision,
            trusted_routes=routes,
        )
    assert error.value.code == "execution.external_authority_unavailable"
