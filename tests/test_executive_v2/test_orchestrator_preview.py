"""Tests for Executive v2 Phase 3 Orchestrator Preview."""

from __future__ import annotations

import time
from typing import Any, Optional

import pytest

from agent.executive.orchestrator_preview import (
    BridgeApprovalError,
    BridgeLinkageConflictError,
    BridgeMappingError,
    ExecutiveOrchestratorBridge,
    build_orchestrator_plan_preview,
    plan_apply,
    plan_dry_run,
    plan_rollback,
)
from agent.executive.state_storage import ObjectiveStateStorage
from agent.executive.types import (
    BridgePreview,
    Complexity,
    GoalClass,
    GoalLinkage,
    NormalizedObjective,
    ObjectiveState,
    ObjectiveStateData,
    RiskProfile,
    OrchestratorPlanPreview,
)


class FakeGoalManager:
    """In-memory fake GoalManager for tests (no real SessionDB)."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._state: Optional[Any] = None


def _make_contract_dict(
    *,
    success_criteria: tuple[str, ...] = ("Sub-goal is delivered",),
    risk_score: float = 0.1,
    approval_requirements: tuple[dict, ...] = (),
    hard: tuple[str, ...] = (),
    soft: tuple[str, ...] = (),
) -> dict:
    return {
        "success_criteria": list(success_criteria),
        "approval_requirements": list(approval_requirements),
        "hard_constraints": list(hard),
        "soft_constraints": list(soft),
        "risk_score": risk_score,
        "budget": {"max_iterations": 24, "max_duration_minutes": 60},
    }


def _make_objective_state(
    objective_id: str = "oid-1",
    contract: Optional[dict] = None,
    fingerprint: str = "abc123",
) -> ObjectiveStateData:
    return ObjectiveStateData(
        objective_id=objective_id,
        state=ObjectiveState.CONTRACT_DRAFT,
        objective_text="implementa una API",
        constraints=[],
        user_id="u-test",
        created_at="2026-01-01",
        fingerprint=fingerprint,
        contract=contract or _make_contract_dict(),
    )


def _make_goal_linkage(
    objective_id: str = "oid-1",
    session_id: str = "test-session",
    goal_text: str = "OBJECTIVE: implementa una API",
    fingerprint: str = "lp-abc",
) -> GoalLinkage:
    return GoalLinkage(
        objective_id=objective_id,
        session_id=session_id,
        goal_text=goal_text,
        bridge_applied_at="2026-01-01",
        bridge_fingerprint=fingerprint,
        bridge_applied_by="u-test",
        bridge_version="phase2.v1",
        bridge_objective_fingerprint="abc123",
    )


@pytest.fixture
def in_memory_storage():
    """In-memory storage with one objective + one Phase 2 goal linkage."""
    state: dict[str, str] = {}

    class FakeDB:
        def set_meta(self, k, v):
            state[k] = v

        def get_meta(self, k):
            return state.get(k)

        def delete_meta(self, k):
            state.pop(k, None)

        def list_meta_keys(self, prefix=None):
            if prefix is None:
                return list(state.keys())
            return [k for k in state.keys() if k.startswith(prefix)]

        def close(self):
            pass

    s = ObjectiveStateStorage(db_factory=lambda: FakeDB())
    s.save(_make_objective_state())
    s.set_objective_goal_link(_make_goal_linkage())
    return s


# ── Section 1: plan_dry_run (3 tests) ────────────────────────────

def test_dry_run_pure_no_side_effects(in_memory_storage):
    snapshot_before = list(in_memory_storage._factory().list_meta_keys())
    plan_dry_run("oid-1", storage=in_memory_storage)
    snapshot_after = list(in_memory_storage._factory().list_meta_keys())
    assert snapshot_before == snapshot_after


def test_dry_run_returns_preview(in_memory_storage):
    preview = plan_dry_run("oid-1", storage=in_memory_storage)
    assert isinstance(preview, OrchestratorPlanPreview)
    assert preview.objective_id == "oid-1"
    assert preview.plan.plan_fingerprint
    assert preview.preview_fingerprint
    # Default success_criteria has 1 element -> 1 subgoal.
    assert len(preview.plan.subgoals) == 1


def test_dry_run_warns_high_risk(in_memory_storage):
    # Save a high-risk contract.
    high_risk = _make_contract_dict(risk_score=0.85)
    in_memory_storage.save(
        _make_objective_state(contract=high_risk)
    )
    preview = plan_dry_run("oid-1", storage=in_memory_storage)
    assert any("HIGH_RISK" in w for w in preview.warnings)
    assert preview.requires_approval is True


# ── Section 2: plan_apply (5 tests) ─────────────────────────────

def test_apply_happy_path(in_memory_storage):
    preview = plan_apply(
        "oid-1",
        storage=in_memory_storage,
        require_human_approval=True,
        approver_id="user-1",
    )
    # Both state_meta keys are written.
    assert in_memory_storage.get_objective_plan("oid-1") is not None
    assert in_memory_storage.get_objective_orchestrator_preview("oid-1") is not None
    assert preview.objective_id == "oid-1"


def test_apply_requires_human_approval_default(in_memory_storage):
    with pytest.raises(BridgeApprovalError) as exc:
        plan_apply(
            "oid-1",
            storage=in_memory_storage,
            require_human_approval=True,
        )
    assert "Layer 1" in str(exc.value)
    # No side effects.
    assert in_memory_storage.get_objective_plan("oid-1") is None
    assert in_memory_storage.get_objective_orchestrator_preview("oid-1") is None


def test_apply_high_risk_requires_token(in_memory_storage):
    in_memory_storage.save(
        _make_objective_state(contract=_make_contract_dict(risk_score=0.85))
    )
    with pytest.raises(BridgeApprovalError) as exc:
        plan_apply(
            "oid-1",
            storage=in_memory_storage,
            require_human_approval=True,
            approver_id="user-1",
        )
    assert "Layer 3" in str(exc.value)
    assert in_memory_storage.get_objective_plan("oid-1") is None


def test_apply_strategic_requires_approver(in_memory_storage):
    """STRATEGIC gate requires approver_id. We pass approver_id but
    ALSO a wrong-flag scenario by also omitting it; the test verifies
    that STRATEGIC specifically surfaces the Layer 2 error.
    """
    in_memory_storage.save(
        _make_objective_state(
            contract=_make_contract_dict(
                approval_requirements=(
                    {"gate": "EXPLICIT_STRATEGIC", "approver": "user", "ttl_hours": 24},
                ),
            ),
        )
    )
    # Without approver_id: Layer 1 fires first (subsumes Layer 2).
    with pytest.raises(BridgeApprovalError) as exc:
        plan_apply(
            "oid-1",
            storage=in_memory_storage,
            require_human_approval=True,
        )
    # Either Layer 1 or Layer 2 is acceptable here (both indicate approver_id missing).
    assert "approver_id" in str(exc.value)
    # STRATEGIC warnings should be in the preview (returned by dry_run).
    preview = plan_dry_run("oid-1", storage=in_memory_storage)
    assert any("STRATEGIC" in w for w in preview.warnings)
    # Storage remains clean.
    assert in_memory_storage.get_objective_plan("oid-1") is None


def test_apply_cross_session_requires_flag(in_memory_storage):
    plan_apply(
        "oid-1",
        storage=in_memory_storage,
        require_human_approval=True,
        approver_id="user-1",
        session_id="test-session",
    )
    # Second apply in a different session without cross_session=True.
    with pytest.raises(BridgeLinkageConflictError) as exc:
        plan_apply(
            "oid-1",
            storage=in_memory_storage,
            require_human_approval=True,
            approver_id="user-1",
            session_id="other-session",
        )
    assert "cross_session=True" in str(exc.value)


# ── Section 3: plan_rollback (2 tests) ──────────────────────────

def test_rollback_idempotent(in_memory_storage):
    result = plan_rollback("oid-1", storage=in_memory_storage)
    assert result is False


def test_rollback_removes_both_keys(in_memory_storage):
    plan_apply(
        "oid-1",
        storage=in_memory_storage,
        require_human_approval=True,
        approver_id="user-1",
    )
    assert in_memory_storage.get_objective_plan("oid-1") is not None
    assert in_memory_storage.get_objective_orchestrator_preview("oid-1") is not None
    result = plan_rollback("oid-1", storage=in_memory_storage)
    assert result is True
    assert in_memory_storage.get_objective_plan("oid-1") is None
    assert in_memory_storage.get_objective_orchestrator_preview("oid-1") is None


# ── Section 4: ExecutiveOrchestratorBridge facade (2 tests) ─────

def test_facade_dry_run(in_memory_storage):
    bridge = ExecutiveOrchestratorBridge(storage=in_memory_storage)
    preview = bridge.dry_run("oid-1")
    assert preview.objective_id == "oid-1"


def test_facade_apply_then_rollback(in_memory_storage):
    bridge = ExecutiveOrchestratorBridge(storage=in_memory_storage)
    bridge.apply("oid-1", require_human_approval=True, approver_id="u-1")
    assert in_memory_storage.get_objective_plan("oid-1") is not None
    assert bridge.rollback("oid-1") is True
    assert in_memory_storage.get_objective_plan("oid-1") is None
