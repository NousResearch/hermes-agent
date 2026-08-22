"""Tests for Executive v2 Phase 2 — GoalManager Bridge."""

from __future__ import annotations

import time
from typing import Any, Optional

import pytest

from agent.executive.goalmanager_bridge import (
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    BridgeApprovalError,
    BridgeLinkageConflictError,
    BridgeMappingError,
    ExecutiveGoalBridge,
    bridge_apply,
    bridge_dry_run,
    bridge_rollback,
    map_contract_to_goal,
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
)
from hermes_cli.goals import (
    DEFAULT_MAX_TURNS,
    GoalContract,
    GoalManager,
    GoalState,
)


# ── Fixtures ────────────────────────────────────────────────────────

class FakeGoalManager:
    """In-memory fake GoalManager for tests (no real SessionDB)."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._state: Optional[GoalState] = None
        self.set_calls: list[tuple[str, Optional[int], Optional[GoalContract]]] = []
        self.set_contract_calls: list[GoalContract] = []
        self.clear_calls: int = 0

    @property
    def state(self) -> Optional[GoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def has_contract(self) -> bool:
        return (
            self._state is not None
            and self._state.contract is not None
            and not self._state.contract.is_empty()
        )

    def status_line(self) -> str:
        return f"{self._state.goal[:50]}" if self._state else "no goal"

    def set(
        self,
        goal: str,
        *,
        max_turns: Optional[int] = None,
        contract: Optional[GoalContract] = None,
    ) -> GoalState:
        self.set_calls.append((goal, max_turns, contract))
        self._state = GoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else DEFAULT_MAX_TURNS,
            created_at=time.time(),
            last_turn_at=0.0,
            contract=contract if contract is not None else GoalContract(),
        )
        return self._state

    def set_contract(self, contract: GoalContract) -> Optional[GoalState]:
        self.set_contract_calls.append(contract)
        if self._state is None:
            return None
        self._state.contract = contract or GoalContract()
        return self._state

    def clear(self) -> None:
        self.clear_calls += 1
        if self._state:
            self._state.status = "cleared"


def _make_contract_dict(
    *,
    success_criteria: tuple[str, ...] = ("Sub-goal is delivered",),
    approval_requirements: tuple[dict, ...] = (),
    hard_constraints: tuple[str, ...] = (),
    soft_constraints: tuple[str, ...] = (),
    risk_score: float = 0.0,
    budget_max_iterations: int = 25,
    verification_method: str = "judge",
    judge_model: Optional[str] = None,
    verification_timeout_minutes: int = 60,
    evidence_required: bool = True,
    objective_id: str = "oid-1",
    fingerprint: str = "abc123",
) -> dict:
    return {
        "success_criteria": list(success_criteria),
        "approval_requirements": list(approval_requirements),
        "hard_constraints": list(hard_constraints),
        "soft_constraints": list(soft_constraints),
        "risk_score": risk_score,
        "budget": {"max_iterations": budget_max_iterations},
        "verification_method": verification_method,
        "judge_model": judge_model,
        "verification_timeout_minutes": verification_timeout_minutes,
        "evidence_required": evidence_required,
        "objective_id": objective_id,
        "fingerprint": fingerprint,
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
        contract=contract or _make_contract_dict(objective_id=objective_id, fingerprint=fingerprint),
    )


@pytest.fixture
def in_memory_storage():
    """In-memory ObjectiveStateStorage (no real SessionDB)."""
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

    return ObjectiveStateStorage(db_factory=lambda: FakeDB())


@pytest.fixture
def populated_storage(in_memory_storage):
    """Storage with one objective in CONTRACT_DRAFT state."""
    state = _make_objective_state()
    in_memory_storage.save(state)
    return in_memory_storage


@pytest.fixture
def fake_gm():
    return FakeGoalManager("test-session-1")


# ── Section 1: map_contract_to_goal (4 tests) ──────────────────────

def test_map_research_objective(populated_storage, fake_gm):
    contract = _make_contract_dict(
        success_criteria=("Information is gathered", "Sources cited"),
        risk_score=0.1,
    )
    goal_text, gc, max_turns, fp, warnings = map_contract_to_goal(contract)
    assert goal_text.startswith("OBJECTIVE: ")
    assert "Information is gathered" in goal_text
    assert max_turns == 25  # low risk, full budget
    assert len(fp) == 64
    assert warnings == ()


def test_map_strategic_high_risk_warns():
    contract = _make_contract_dict(
        risk_score=0.85,
        approval_requirements=(
            {"gate": "EXPLICIT_STRATEGIC", "approver": "user", "ttl_hours": 24},
        ),
    )
    _, _, _, _, warnings = map_contract_to_goal(contract)
    joined = " ".join(warnings)
    assert "HIGH_RISK" in joined
    assert "STRATEGIC" in joined


def test_map_empty_success_criteria_warns():
    contract = _make_contract_dict(success_criteria=())
    goal_text, _, _, _, warnings = map_contract_to_goal(contract)
    assert goal_text == "(no success_criteria)"
    assert any("EMPTY" in w for w in warnings)


def test_map_invariant_under_repeated_call():
    contract = _make_contract_dict(
        success_criteria=("a", "b"),
        risk_score=0.3,
        budget_max_iterations=30,
    )
    results = [map_contract_to_goal(contract) for _ in range(10)]
    fps = {r[3] for r in results}
    max_turns = {r[2] for r in results}
    goal_texts = {r[0] for r in results}
    assert len(fps) == 1, "fingerprint should be invariant"
    assert len(max_turns) == 1
    assert len(goal_texts) == 1


# ── Section 2: bridge_dry_run (4 tests) ───────────────────────────

def test_dry_run_pure_no_side_effects(populated_storage, fake_gm):
    snapshot_before = list(populated_storage._factory().list_meta_keys())
    bridge_dry_run("oid-1", fake_gm, storage=populated_storage)
    snapshot_after = list(populated_storage._factory().list_meta_keys())
    assert snapshot_before == snapshot_after
    # FakeGoalManager should not have been mutated.
    assert fake_gm.state is None
    assert fake_gm.set_calls == []


def test_dry_run_returns_preview(populated_storage, fake_gm):
    preview = bridge_dry_run("oid-1", fake_gm, storage=populated_storage)
    assert isinstance(preview, BridgePreview)
    assert preview.objective_id == "oid-1"
    assert preview.session_id == "test-session-1"
    assert preview.goal_text.startswith("OBJECTIVE: ")
    assert preview.max_turns > 0
    assert preview.bridge_fingerprint
    assert preview.would_apply_to_existing_goal is False
    assert preview.cross_session_conflict is False


def test_dry_run_warns_high_risk(populated_storage, fake_gm):
    # Re-save with high-risk contract.
    state = _make_objective_state(
        contract=_make_contract_dict(risk_score=0.85),
    )
    populated_storage.save(state)
    preview = bridge_dry_run("oid-1", fake_gm, storage=populated_storage)
    assert any("HIGH_RISK" in w for w in preview.warnings)


def test_dry_run_with_existing_link_flags(populated_storage, fake_gm):
    # Save an existing link.
    link = GoalLinkage(
        objective_id="oid-1",
        session_id="OTHER-session",
        goal_text="OBJECTIVE: x",
        bridge_applied_at="2026-01-01",
        bridge_fingerprint="abc",
        bridge_applied_by="u",
        bridge_version="phase2.v1",
        bridge_objective_fingerprint="",
    )
    populated_storage.set_objective_goal_link(link)
    preview = bridge_dry_run("oid-1", fake_gm, storage=populated_storage)
    assert preview.would_apply_to_existing_goal is True
    assert preview.cross_session_conflict is True


# ── Section 3: bridge_apply (4 tests) ─────────────────────────────

def test_apply_happy_path(populated_storage, fake_gm):
    link = bridge_apply(
        "oid-1",
        fake_gm,
        storage=populated_storage,
        require_human_approval=True,
        approver_id="user-1",
    )
    assert isinstance(link, GoalLinkage)
    assert link.objective_id == "oid-1"
    assert link.session_id == "test-session-1"
    assert link.bridge_applied_by == "user-1"
    # GoalManager was called.
    assert len(fake_gm.set_calls) == 1
    assert fake_gm.set_calls[0][0].startswith("OBJECTIVE: ")
    # Link is persisted.
    stored = populated_storage.get_objective_goal_link("oid-1")
    assert stored is not None
    assert stored.objective_id == "oid-1"


def test_apply_requires_human_approval_default(populated_storage, fake_gm):
    """Without approver_id, raises BridgeApprovalError."""
    with pytest.raises(BridgeApprovalError) as exc:
        bridge_apply(
            "oid-1",
            fake_gm,
            storage=populated_storage,
            require_human_approval=True,
        )
    assert "Layer 1" in str(exc.value)
    # No side effects.
    assert fake_gm.set_calls == []
    assert populated_storage.get_objective_goal_link("oid-1") is None


def test_apply_high_risk_requires_token(populated_storage, fake_gm):
    state = _make_objective_state(
        contract=_make_contract_dict(risk_score=0.85),
    )
    populated_storage.save(state)
    with pytest.raises(BridgeApprovalError) as exc:
        bridge_apply(
            "oid-1",
            fake_gm,
            storage=populated_storage,
            require_human_approval=True,
            approver_id="user-1",
            # approval_token missing
        )
    assert "Layer 3" in str(exc.value)
    assert fake_gm.set_calls == []


def test_apply_cross_session_requires_flag(populated_storage, fake_gm):
    # First apply: ok.
    bridge_apply(
        "oid-1",
        fake_gm,
        storage=populated_storage,
        require_human_approval=True,
        approver_id="user-1",
    )
    # Second apply in a different session without cross_session=True.
    other_gm = FakeGoalManager("other-session")
    with pytest.raises(BridgeLinkageConflictError) as exc:
        bridge_apply(
            "oid-1",
            other_gm,
            storage=populated_storage,
            require_human_approval=True,
            approver_id="user-1",
        )
    assert "cross_session=True" in str(exc.value)
    assert len(other_gm.set_calls) == 0


# ── Section 4: bridge_rollback (4 tests) ─────────────────────────

def test_rollback_idempotent_no_link(populated_storage, fake_gm):
    result = bridge_rollback("oid-1", fake_gm, storage=populated_storage)
    assert result is False
    assert fake_gm.clear_calls == 0


def test_rollback_clears_goal_and_deletes_link(populated_storage, fake_gm):
    bridge_apply(
        "oid-1",
        fake_gm,
        storage=populated_storage,
        require_human_approval=True,
        approver_id="user-1",
    )
    assert fake_gm.state is not None
    assert fake_gm.state.status == "active"
    result = bridge_rollback("oid-1", fake_gm, storage=populated_storage)
    assert result is True
    assert fake_gm.clear_calls == 1
    # Goal is now cleared (audit preserved).
    assert fake_gm.state.status == "cleared"
    # Link is gone.
    assert populated_storage.get_objective_goal_link("oid-1") is None


def test_rollback_preserves_goal_if_text_mismatch(populated_storage, fake_gm):
    bridge_apply(
        "oid-1",
        fake_gm,
        storage=populated_storage,
        require_human_approval=True,
        approver_id="user-1",
    )
    # Tamper with the goal text on the manager.
    fake_gm._state.goal = "DIFFERENT GOAL"
    result = bridge_rollback("oid-1", fake_gm, storage=populated_storage)
    assert result is True
    # Goal is NOT cleared (text mismatch).
    assert fake_gm.clear_calls == 0
    # Link IS deleted.
    assert populated_storage.get_objective_goal_link("oid-1") is None


def test_rollback_preserves_goal_if_session_mismatch(populated_storage, fake_gm):
    bridge_apply(
        "oid-1",
        fake_gm,
        storage=populated_storage,
        require_human_approval=True,
        approver_id="user-1",
    )
    other_gm = FakeGoalManager("other-session")
    result = bridge_rollback("oid-1", other_gm, storage=populated_storage)
    assert result is True
    # Goal is NOT cleared (session mismatch).
    assert other_gm.clear_calls == 0
    # Link IS deleted.
    assert populated_storage.get_objective_goal_link("oid-1") is None


# ── Facade test ────────────────────────────────────────────────────

def test_executive_goal_bridge_facade(in_memory_storage):
    state = _make_objective_state()
    in_memory_storage.save(state)
    bridge = ExecutiveGoalBridge(storage=in_memory_storage)
    gm = FakeGoalManager("sess-x")
    preview = bridge.dry_run("oid-1", gm)
    assert preview.session_id == "sess-x"
    link = bridge.apply(
        "oid-1",
        gm,
        require_human_approval=True,
        approver_id="u-1",
    )
    assert link.objective_id == "oid-1"
    assert bridge.rollback("oid-1", gm) is True
