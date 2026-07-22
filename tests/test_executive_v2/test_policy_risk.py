"""Tests for Phase 4A risk classification + policy building (24 tests).

Covers:

* ``classify_risk_level`` (8 tests): R0-R6 classification heuristics.
* ``build_policy_decision`` (8 tests): allowed/forbidden actions,
  approval_required, warnings, fingerprint stability.
* ``policy_dry_run`` (4 tests): integration with state_meta read.
* ``policy_persist`` + ``policy_rollback`` (4 tests): state_meta
  write/read/cleanup with 8-layer approval gates.

All tests are hermetic: in-memory storage, no providers, no network,
no subprocess, no Kanban, no Orchestrator, no workers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.executive.goalmanager_bridge import (
    BridgeApprovalError,
    BridgeMappingError,
)
from agent.executive.policy import (
    ExecutivePolicyEngine,
    allowed_actions_for,
    build_policy_decision,
    classify_risk_level,
    forbidden_actions_for,
    policy_dry_run,
    policy_persist,
    policy_rollback,
)
from agent.executive.types import (
    ACTION_ASSIGN_KANBAN_TASK,
    ACTION_CREATE_KANBAN_TASK,
    ACTION_NETWORK_CALL,
    ACTION_READ_STATE_META,
    ACTION_SPAWN_WORKER,
    ACTION_WRITE_APPROVAL_REQUEST,
    ACTION_WRITE_KANBAN_METADATA,
    ACTION_WRITE_POLICY_DECISION,
    ApprovalRequest,
    GoalLinkage,
    ObjectivePlan,
    OrchestratorPlanPreview,
    PlannerSubgoal,
    PolicyDecision,
    RiskLevel,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _empty_contract() -> dict:
    return {"risk_score": 0.0, "risk_components": {}, "approval_requirements": []}


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


def _seed_storage(in_memory_storage, *, objective_id: str = "obj-1",
                  contract: dict | None = None):
    """Seed an in-memory storage with Phase 1+2+3 artifacts."""
    from agent.executive.types import ObjectiveState, ObjectiveStateData
    linkage = _make_linkage()
    plan = _make_plan((_make_subgoal(title="investigate", intent="RESEARCH"),))
    in_memory_storage.set_objective_goal_link(linkage)
    in_memory_storage.set_objective_plan(plan)

    state = ObjectiveStateData(
        objective_id=objective_id,
        state=ObjectiveState.DRAFT,
        objective_text="seeded",
        constraints=[],
        user_id="user-1",
        created_at="2026-07-02T10:00:00+00:00",
        contract=dict(contract) if contract else {},
    )
    in_memory_storage.save(state)
    return linkage, plan


# ══════════════════════════════════════════════════════════════════════
# Section 1: classify_risk_level (8 tests)
# ══════════════════════════════════════════════════════════════════════

def test_classify_r0_default():
    level = classify_risk_level(
        execution_contract=_empty_contract(),
        goal_linkage=_make_linkage("list active sessions"),
        objective_plan=_make_plan(()),
    )
    assert level == RiskLevel.R0
    assert int(level) == 0


def test_classify_r3_high_score():
    contract = {"risk_score": 0.5, "risk_components": {}}
    level = classify_risk_level(
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert level == RiskLevel.R3


def test_classify_r4_kanban_intent():
    subgoals = (_make_subgoal(title="build feature", intent="BUILD"),)
    contract = {"risk_score": 0.1, "risk_components": {}}
    level = classify_risk_level(
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(subgoals),
    )
    assert level == RiskLevel.R4


def test_classify_r5_worker_threshold():
    contract = {"risk_score": 0.85, "risk_components": {}}
    level = classify_risk_level(
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert level == RiskLevel.R5
    # Also via irreversibility.
    contract2 = {"risk_score": 0.0, "risk_components": {"irreversibility": 0.5}}
    level2 = classify_risk_level(
        execution_contract=contract2,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert level2 == RiskLevel.R5


def test_classify_r6_external_intent():
    subgoals = (_make_subgoal(title="call external API"),)
    contract = {"risk_score": 0.1, "risk_components": {}}
    level = classify_risk_level(
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(subgoals),
    )
    assert level == RiskLevel.R6


def test_classify_r6_financial_component():
    contract = {"risk_score": 0.0, "risk_components": {"financial": 0.5}}
    level = classify_risk_level(
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert level == RiskLevel.R6


def test_classify_priority_highest_wins():
    contract = {
        "risk_score": 0.7,
        "risk_components": {"financial": 0.1, "customer_facing": 0.5},
    }
    subgoals = (_make_subgoal(title="build dashboard", intent="BUILD"),)
    level = classify_risk_level(
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(subgoals),
    )
    assert level == RiskLevel.R6


def test_classify_invariant_under_repeated_call():
    contract = {"risk_score": 0.5, "risk_components": {}}
    subgoals = (_make_subgoal(title="build X", intent="BUILD"),)
    linkage = _make_linkage()
    first = classify_risk_level(contract, linkage, _make_plan(subgoals))
    second = classify_risk_level(dict(contract), linkage, _make_plan(subgoals))
    third = classify_risk_level(dict(contract), linkage, _make_plan(subgoals))
    assert first == second == third


# ══════════════════════════════════════════════════════════════════════
# Section 2: build_policy_decision (8 tests)
# ══════════════════════════════════════════════════════════════════════

def test_build_decision_r0_autonomous():
    decision = build_policy_decision(
        "obj-1",
        execution_contract=_empty_contract(),
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert decision.risk_level == RiskLevel.R0
    assert decision.approval_required is False
    assert ACTION_READ_STATE_META in decision.allowed_actions
    assert decision.decision_fingerprint  # non-empty


def test_build_decision_r3_state_meta_requires_approval():
    contract = {"risk_score": 0.4, "risk_components": {}, "approval_requirements": []}
    decision = build_policy_decision(
        "obj-1",
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert decision.risk_level == RiskLevel.R3
    assert decision.approval_required is True


def test_build_decision_r4_kanban_writes_allowed():
    subgoals = (_make_subgoal(title="build X", intent="BUILD"),)
    contract = {"risk_score": 0.1, "risk_components": {}, "approval_requirements": []}
    decision = build_policy_decision(
        "obj-1",
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(subgoals),
    )
    assert decision.risk_level == RiskLevel.R4
    assert decision.approval_required is True
    assert ACTION_WRITE_KANBAN_METADATA in decision.allowed_actions
    assert ACTION_CREATE_KANBAN_TASK in decision.allowed_actions
    assert ACTION_ASSIGN_KANBAN_TASK in decision.allowed_actions
    assert ACTION_SPAWN_WORKER in decision.forbidden_actions


def test_build_decision_r5_workers_allowed():
    contract = {"risk_score": 0.85, "risk_components": {}, "approval_requirements": []}
    decision = build_policy_decision(
        "obj-1",
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert decision.risk_level == RiskLevel.R5
    assert ACTION_SPAWN_WORKER in decision.allowed_actions


def test_build_decision_r6_external_allowed():
    contract = {
        "risk_score": 0.0,
        "risk_components": {"financial": 0.5},
        "approval_requirements": [],
    }
    decision = build_policy_decision(
        "obj-1",
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert decision.risk_level == RiskLevel.R6
    assert ACTION_NETWORK_CALL in decision.allowed_actions


def test_build_decision_strategic_warning():
    contract = {
        "risk_score": 0.0,
        "risk_components": {},
        "approval_requirements": [{"gate": "EXPLICIT_STRATEGIC", "approver": "user"}],
    }
    decision = build_policy_decision(
        "obj-1",
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert any("STRATEGIC" in w for w in decision.warnings)


def test_build_decision_high_risk_warning():
    contract = {"risk_score": 0.75, "risk_components": {}, "approval_requirements": []}
    decision = build_policy_decision(
        "obj-1",
        execution_contract=contract,
        goal_linkage=_make_linkage(),
        objective_plan=_make_plan(()),
    )
    assert any("HIGH_RISK" in w for w in decision.warnings)


def test_build_decision_fingerprint_idempotent():
    contract = {"risk_score": 0.4, "risk_components": {}, "approval_requirements": []}
    linkage = _make_linkage()
    plan = _make_plan(())
    d1 = build_policy_decision("obj-1", dict(contract), linkage, plan)
    d2 = build_policy_decision("obj-1", dict(contract), linkage, plan)
    assert d1.decision_fingerprint == d2.decision_fingerprint


# ══════════════════════════════════════════════════════════════════════
# Section 3: policy_dry_run (4 tests)
# ══════════════════════════════════════════════════════════════════════

def test_policy_dry_run_zero_side_effects(in_memory_storage):
    """policy_dry_run does NOT write to state_meta."""
    _seed_storage(in_memory_storage, contract={"risk_score": 0.4, "risk_components": {}, "approval_requirements": []})
    decision = policy_dry_run("obj-1", storage=in_memory_storage)
    assert isinstance(decision, PolicyDecision)
    assert decision.risk_level == RiskLevel.R3

    # No Phase 4A keys should be written by a dry run.
    assert in_memory_storage.get_objective_policy_decision("obj-1") is None
    assert in_memory_storage.get_objective_approval_request("obj-1") is None


def test_policy_dry_run_missing_goal_linkage(in_memory_storage):
    """Missing Phase 2 linkage → BridgeMappingError, no writes."""
    in_memory_storage.set_objective_plan(_make_plan())
    with pytest.raises(BridgeMappingError):
        policy_dry_run("obj-1", storage=in_memory_storage)


def test_policy_dry_run_missing_objective_plan(in_memory_storage):
    """Missing Phase 3 plan → BridgeMappingError, no writes."""
    in_memory_storage.set_objective_goal_link(_make_linkage())
    with pytest.raises(BridgeMappingError):
        policy_dry_run("obj-1", storage=in_memory_storage)


def test_policy_dry_run_r0_no_approval(in_memory_storage):
    """R0 decision needs no approval."""
    from agent.executive.types import ObjectiveState, ObjectiveStateData
    in_memory_storage.set_objective_goal_link(_make_linkage())
    in_memory_storage.set_objective_plan(_make_plan(()))
    state = ObjectiveStateData(
        objective_id="obj-1",
        state=ObjectiveState.DRAFT,
        objective_text="seeded",
        constraints=[],
        user_id="user-1",
        created_at="2026-07-02T10:00:00+00:00",
        contract=_empty_contract(),
    )
    in_memory_storage.save(state)
    decision = policy_dry_run("obj-1", storage=in_memory_storage)
    assert decision.risk_level == RiskLevel.R0
    assert decision.approval_required is False


# ══════════════════════════════════════════════════════════════════════
# Section 4: policy_persist + policy_rollback (4 tests)
# ══════════════════════════════════════════════════════════════════════

def test_policy_persist_writes_both_keys(in_memory_storage):
    """policy_persist writes state_meta with explicit approval."""
    _seed_storage(
        in_memory_storage,
        contract={"risk_score": 0.4, "risk_components": {}, "approval_requirements": []},
    )
    decision, request = policy_persist(
        "obj-1", storage=in_memory_storage, approver_id="user-1"
    )
    assert decision.risk_level == RiskLevel.R3
    assert request.approver_id == "user-1"
    assert in_memory_storage.get_objective_policy_decision("obj-1") is not None
    assert in_memory_storage.get_objective_approval_request("obj-1") is not None


def test_policy_persist_no_writes_on_gate_fail(in_memory_storage):
    """A failing approval gate must not write to state_meta."""
    _seed_storage(
        in_memory_storage,
        contract={"risk_score": 0.85, "risk_components": {}, "approval_requirements": []},
    )
    with pytest.raises(BridgeApprovalError):
        policy_persist("obj-1", storage=in_memory_storage)
    assert in_memory_storage.get_objective_policy_decision("obj-1") is None
    assert in_memory_storage.get_objective_approval_request("obj-1") is None


def test_policy_rollback_removes_both_keys(in_memory_storage):
    """policy_rollback deletes both keys when they exist."""
    _seed_storage(
        in_memory_storage,
        contract={"risk_score": 0.4, "risk_components": {}, "approval_requirements": []},
    )
    policy_persist("obj-1", storage=in_memory_storage, approver_id="user-1")
    cleaned = policy_rollback("obj-1", storage=in_memory_storage)
    assert cleaned is True
    assert in_memory_storage.get_objective_policy_decision("obj-1") is None
    assert in_memory_storage.get_objective_approval_request("obj-1") is None


def test_policy_rollback_idempotent(in_memory_storage):
    """policy_rollback is idempotent: returns False when nothing to clean."""
    _seed_storage(in_memory_storage)
    cleaned = policy_rollback("obj-1", storage=in_memory_storage)
    assert cleaned is False


# ══════════════════════════════════════════════════════════════════════
# Section 5: action matrix sanity (light; gate-by-matrix consistency)
# ══════════════════════════════════════════════════════════════════════

def test_allowed_actions_r0_minimal():
    allowed = allowed_actions_for(RiskLevel.R0)
    assert ACTION_READ_STATE_META in allowed
    assert ACTION_WRITE_APPROVAL_REQUEST not in allowed


def test_allowed_actions_r3_state_meta():
    allowed = allowed_actions_for(RiskLevel.R3)
    assert ACTION_READ_STATE_META in allowed
    assert ACTION_WRITE_POLICY_DECISION in allowed
    assert ACTION_WRITE_APPROVAL_REQUEST in allowed
    assert ACTION_SPAWN_WORKER not in allowed


def test_allowed_actions_r4_kanban():
    allowed = allowed_actions_for(RiskLevel.R4)
    assert ACTION_WRITE_KANBAN_METADATA in allowed
    assert ACTION_CREATE_KANBAN_TASK in allowed


def test_forbidden_actions_complement():
    """forbidden = ALL_ACTIONS - allowed, for every level."""
    from agent.executive.types import ALL_ACTIONS as _ALL
    all_actions_set = set(_ALL)
    for level in (RiskLevel.R0, RiskLevel.R3, RiskLevel.R4, RiskLevel.R5, RiskLevel.R6):
        allowed = set(allowed_actions_for(level))
        forbidden = set(forbidden_actions_for(level))
        assert allowed.isdisjoint(forbidden)
        assert allowed | forbidden == all_actions_set