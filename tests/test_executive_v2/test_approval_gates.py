"""Tests for Phase 4A 8-layer approval gates (12 tests).

Covers ``evaluate_approval_gates`` from
``agent.executive.approval_gates``:

* Layer 1 (default).
* Layer 2 (STRATEGIC).
* Layer 3 (HIGH_RISK).
* Layer 4 (cross-session).
* Layer 5 (Kanban_create R4).
* Layer 6 (Worker_spawn R5).
* Layer 7 (External_call R6).
* Layer 8 (token_expiry).
* Layer 1 subsumes Layer 2.
* Happy path R0 (no approval needed).
* Happy path R3.
* Happy path R4.

Plus 4 integration tests covering the persist path through
``ExecutivePolicyEngine``.

All tests are hermetic: in-memory storage, no providers, no network,
no subprocess, no Kanban, no Orchestrator, no workers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.executive.approval_gates import (
    ApprovalGateEvaluator,
    ApprovalGateResult,
    evaluate_approval_gates,
)
from agent.executive.goalmanager_bridge import BridgeApprovalError
from agent.executive.policy import (
    ExecutivePolicyEngine,
    build_policy_decision,
    policy_persist,
)
from agent.executive.types import (
    GoalLinkage,
    ObjectivePlan,
    PlannerSubgoal,
    PolicyDecision,
    RiskLevel,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_subgoal(
    *,
    id: str = "sg-0",
    title: str = "research X",
    intent: str = "RESEARCH",
    risk_level: str = "low",
    approval_required: bool = False,
    source_criterion_index: int = 0,
) -> PlannerSubgoal:
    return PlannerSubgoal(
        id=id,
        title=title,
        intent=intent,
        constraints=(),
        expected_output="",
        risk_level=risk_level,
        approval_required=approval_required,
        estimated_iterations=1,
        timeout_seconds=60,
        source_criterion_index=source_criterion_index,
    )


def _make_plan(subgoals: tuple = ()) -> ObjectivePlan:
    return ObjectivePlan(
        objective_id="obj-1",
        subgoals=subgoals,
        plan_fingerprint="fp",
        created_at="2026-07-02T10:00:00+00:00",
    )


def _make_linkage(
    goal_text: str = "Research goal",
    session_id: str = "sess-1",
) -> GoalLinkage:
    return GoalLinkage(
        objective_id="obj-1",
        session_id=session_id,
        goal_text=goal_text,
        bridge_applied_at="2026-07-02T10:00:00+00:00",
        bridge_fingerprint="link-fp",
        bridge_applied_by="user",
        bridge_version="phase2.v1",
        bridge_objective_fingerprint="obj-fp",
    )


def _r0_decision() -> PolicyDecision:
    return build_policy_decision(
        "obj-1",
        execution_contract={"risk_score": 0.0, "risk_components": {}, "approval_requirements": []},
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )


def _r3_decision() -> PolicyDecision:
    return build_policy_decision(
        "obj-1",
        execution_contract={"risk_score": 0.4, "risk_components": {}, "approval_requirements": []},
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )


def _r4_decision() -> PolicyDecision:
    return build_policy_decision(
        "obj-1",
        execution_contract={"risk_score": 0.1, "risk_components": {}, "approval_requirements": []},
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan((_make_subgoal(intent="BUILD"),)),
    )


def _r5_decision() -> PolicyDecision:
    return build_policy_decision(
        "obj-1",
        execution_contract={"risk_score": 0.85, "risk_components": {}, "approval_requirements": []},
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )


def _r6_decision() -> PolicyDecision:
    return build_policy_decision(
        "obj-1",
        execution_contract={"risk_score": 0.0, "risk_components": {"financial": 0.5}, "approval_requirements": []},
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )


def _strategic_decision() -> PolicyDecision:
    return build_policy_decision(
        "obj-1",
        execution_contract={
            "risk_score": 0.4,
            "risk_components": {},
            "approval_requirements": [{"gate": "EXPLICIT_STRATEGIC", "approver": "user"}],
        },
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )


# ══════════════════════════════════════════════════════════════════════
# 8 individual layer tests
# ══════════════════════════════════════════════════════════════════════

def test_layer1_default_requires_approver():
    """R3 decision (approval_required=True) without approver_id → Layer 1 fail."""
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(_r3_decision())
    assert "Layer 1" in str(exc_info.value)


def test_layer2_strategic_requires_approver():
    """A STRATEGIC approval_requirement at R0 demands approver_id.

    At R3, Layer 1 would subsume Layer 2 (both require approver_id),
    so to test Layer 2 alone we use an R0 contract with a STRATEGIC
    gate, which makes approval_required=False but STRATEGIC fires.
    """
    decision = build_policy_decision(
        "obj-1",
        execution_contract={
            "risk_score": 0.0,
            "risk_components": {},
            "approval_requirements": [{"gate": "EXPLICIT_STRATEGIC", "approver": "user"}],
        },
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    # Sanity: R0 so Layer 1 doesn't fire first.
    assert decision.risk_level == RiskLevel.R0
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(decision)
    assert "Layer 2" in str(exc_info.value)


def test_layer3_high_risk_requires_approval_token():
    """R4+ requires approval_token."""
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(
            _r4_decision(),
            approver_id="user-1",
            kanban_approver_id="kanban-admin",
        )
    assert "Layer 3" in str(exc_info.value)


def test_layer4_cross_session_requires_cross_session_flag():
    """session_id supplied without cross_session=True → Layer 4 fail."""
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(
            _r3_decision(),
            approver_id="user-1",
            session_id="sess-2",
        )
    assert "Layer 4" in str(exc_info.value)


def test_layer5_kanban_create_requires_kanban_approver():
    """R4 demands kanban_approver_id."""
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(
            _r4_decision(),
            approver_id="user-1",
            approval_token="tok-abc",
        )
    assert "Layer 5" in str(exc_info.value)


def test_layer6_worker_spawn_requires_worker_approver():
    """R5 demands worker_approver_id."""
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(
            _r5_decision(),
            approver_id="user-1",
            approval_token="tok-abc",
        )
    assert "Layer 6" in str(exc_info.value)


def test_layer7_external_call_requires_external_approver():
    """R6 demands external_approver_id."""
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(
            _r6_decision(),
            approver_id="user-1",
            approval_token="tok-abc",
        )
    assert "Layer 7" in str(exc_info.value)


def test_layer8_token_expiry_requires_renewal():
    """Expired expiry without renewal=True → Layer 8 fail."""
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(
            _r3_decision(),
            approver_id="user-1",
            expiry=past,
        )
    assert "Layer 8" in str(exc_info.value)


# ══════════════════════════════════════════════════════════════════════
# Layer subsumption + happy paths
# ══════════════════════════════════════════════════════════════════════

def test_layer1_subsumes_layer2():
    """Layer 1 fail (no approver_id at R3) subsumes Layer 2 STRATEGIC check."""
    with pytest.raises(BridgeApprovalError) as exc_info:
        evaluate_approval_gates(_strategic_decision())
    assert "Layer 1" in str(exc_info.value)


def test_happy_path_r0_no_approval_needed():
    """R0 decision: approval_required=False, no approver_id needed."""
    result = evaluate_approval_gates(_r0_decision())
    assert isinstance(result, ApprovalGateResult)
    assert result.approved is True
    assert result.approval_request is not None
    assert result.approval_request.approver_id is None
    assert result.approval_request.risk_level == RiskLevel.R0


def test_happy_path_r3_with_approver():
    """R3 with approver_id passes Layers 1, 2, 3, 4, 5, 6, 7, 8."""
    result = evaluate_approval_gates(
        _r3_decision(),
        approver_id="user-1",
    )
    assert result.approved is True
    assert result.approval_request is not None
    assert result.approval_request.approver_id == "user-1"
    assert result.approval_request.risk_level == RiskLevel.R3


def test_happy_path_r4_full_approval():
    """R4 with full set of approvers + token passes all gates."""
    result = evaluate_approval_gates(
        _r4_decision(),
        approver_id="user-1",
        approval_token="tok-abc",
        kanban_approver_id="kanban-admin",
    )
    assert result.approved is True
    assert result.approval_request is not None
    assert result.approval_request.approver_id == "user-1"
    assert result.approval_request.kanban_approver_id == "kanban-admin"
    assert result.approval_request.approval_token == "tok-abc"


# ══════════════════════════════════════════════════════════════════════
# Integration: ApprovalGateEvaluator class + persist path
# ══════════════════════════════════════════════════════════════════════

def test_evaluator_class_api_matches_module_function():
    """ApprovalGateEvaluator().evaluate(...) yields the same ApprovalGateResult shape."""
    evaluator = ApprovalGateEvaluator()
    result = evaluator.evaluate(
        _r3_decision(),
        approver_id="user-1",
    )
    assert result.approved is True
    assert result.approval_request is not None
    assert result.approval_request.approver_id == "user-1"


def test_layer_results_records_all_8_layers():
    """ApprovalGateResult.layer_results has 8 entries (one per layer)."""
    result = evaluate_approval_gates(
        _r4_decision(),
        approver_id="user-1",
        approval_token="tok-abc",
        kanban_approver_id="kanban-admin",
    )
    layers = {entry["layer"] for entry in result.layer_results}
    assert layers == {1, 2, 3, 4, 5, 6, 7, 8}


def test_renewal_token_unblocks_layer_8():
    """Expired expiry + renewal=True passes Layer 8."""
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    result = evaluate_approval_gates(
        _r3_decision(),
        approver_id="user-1",
        expiry=past,
        renewal=True,
    )
    assert result.approved is True
    layer8 = next(e for e in result.layer_results if e["layer"] == 8)
    assert layer8["passed"] is True


def test_no_session_id_skips_layer_4():
    """Layer 4 is skipped (passed=True) when no session_id is supplied."""
    result = evaluate_approval_gates(_r3_decision(), approver_id="user-1")
    layer4 = next(e for e in result.layer_results if e["layer"] == 4)
    assert layer4["passed"] is True
    assert "skipped" in layer4["reason"]


def test_executive_policy_engine_persist_round_trip(in_memory_storage):
    """ExecutivePolicyEngine.persist writes both keys; rollback removes both."""
    from agent.executive.types import ObjectiveState, ObjectiveStateData

    in_memory_storage.set_objective_goal_link(_make_linkage())
    in_memory_storage.set_objective_plan(_make_plan((_make_subgoal(intent="RESEARCH"),)))
    state = ObjectiveStateData(
        objective_id="obj-1",
        state=ObjectiveState.DRAFT,
        objective_text="seeded",
        constraints=[],
        user_id="user-1",
        created_at="2026-07-02T10:00:00+00:00",
        contract={"risk_score": 0.4, "risk_components": {}, "approval_requirements": []},
    )
    in_memory_storage.save(state)

    engine = ExecutivePolicyEngine(storage=in_memory_storage)
    decision, request = engine.persist("obj-1", approver_id="user-1")
    assert decision.risk_level == RiskLevel.R3

    persisted = engine._storage.get_objective_policy_decision("obj-1")
    assert persisted is not None
    assert persisted.decision_fingerprint == decision.decision_fingerprint

    cleaned = engine.rollback("obj-1")
    assert cleaned is True
    assert engine._storage.get_objective_policy_decision("obj-1") is None
    assert engine._storage.get_objective_approval_request("obj-1") is None