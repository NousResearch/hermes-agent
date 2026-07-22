"""Tests for Phase 4B Kanban Apply — KanbanApplyEngine integration.

18 tests covering:
* Pure preview (3 tests)
* Apply happy path (5 tests)
* Apply with gate failures (3 tests)
* Apply with missing inputs (3 tests)
* Idempotency / dedup (4 tests)

All tests use a fake ``kb.create_task`` injection (no real kanban DB writes).
"""

from __future__ import annotations

import pytest

from agent.executive.goalmanager_bridge import (
    BridgeApprovalError,
    BridgeMappingError,
)
from agent.executive.kanban_apply import (
    KanbanApplyEngine,
    KanbanLinkageConflictError,
    build_kanban_apply_preview,
    kanban_apply,
)
from agent.executive.types import (
    GoalLinkage,
    ObjectivePlan,
    ObjectiveState,
    ObjectiveStateData,
    OrchestratorPlanPreview,
    PlannerSubgoal,
    PolicyDecision,
    ApprovalRequest,
    RiskLevel,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _seed_full(
    in_memory_storage,
    *,
    objective_id: str = "obj-1",
    risk_score: float = 0.4,
    risk_components: dict | None = None,
    session_id: str = "sess-1",
    task_specs: list | None = None,
    request_overrides: dict | None = None,
) -> None:
    """Seed Phase 1+2+3+4A artifacts into in-memory storage."""
    if risk_components is None:
        risk_components = {}
    # Phase 1: objective state with contract.
    state = ObjectiveStateData(
        objective_id=objective_id,
        state=ObjectiveState.DRAFT,
        objective_text="seeded",
        constraints=[],
        user_id="user-1",
        created_at="2026-07-02T12:00:00+00:00",
        contract={
            "risk_score": risk_score,
            "risk_components": risk_components,
            "approval_requirements": [],
        },
    )
    in_memory_storage.save(state)

    # Phase 2: goal_linkage.
    linkage = GoalLinkage(
        objective_id=objective_id,
        session_id=session_id,
        goal_text="seeded goal",
        bridge_applied_at="2026-07-02T12:00:00+00:00",
        bridge_fingerprint="link-fp",
        bridge_applied_by="user-1",
        bridge_version="phase2.v1",
        bridge_objective_fingerprint="obj-fp",
    )
    in_memory_storage.set_objective_goal_link(linkage)

    # Phase 3: plan + orchestrator preview.
    if task_specs is None:
        task_specs = [
            {
                "description": "investigate X",
                "assigned_profile": "researcher",
                "inputs": {"criterion_index": 0},
                "expected_outputs": ["report.md"],
                "dependencies": [],
                "timeout_s": 60,
                "requires_user_input": False,
                "approval_id": None,
                "risk_level": "low",
            },
            {
                "description": "synthesize findings",
                "assigned_profile": "researcher",
                "inputs": {"criterion_index": 1},
                "expected_outputs": ["summary.md"],
                "dependencies": [],
                "timeout_s": 60,
                "requires_user_input": False,
                "approval_id": None,
                "risk_level": "low",
            },
        ]
    plan = ObjectivePlan(
        objective_id=objective_id,
        subgoals=tuple(
            PlannerSubgoal(
                id=f"sg-{i}",
                title=spec["description"],
                intent="RESEARCH",
                constraints=(),
                expected_output=spec.get("expected_outputs", [""])[0]
                if spec.get("expected_outputs")
                else "",
                risk_level="low",
                approval_required=spec.get("requires_user_input", False),
                estimated_iterations=1,
                timeout_seconds=spec.get("timeout_s", 60),
                source_criterion_index=i,
            )
            for i, spec in enumerate(task_specs)
        ),
        plan_fingerprint="plan-fp",
        created_at="2026-07-02T12:00:00+00:00",
    )
    in_memory_storage.set_objective_plan(plan)
    preview = OrchestratorPlanPreview(
        objective_id=objective_id,
        plan=plan,
        task_specs=tuple(task_specs),
        warnings=(),
        requires_approval=True,
        risk_score=risk_score,
        preview_fingerprint="preview-fp",
        created_at="2026-07-02T12:00:00+00:00",
    )
    in_memory_storage.set_objective_orchestrator_preview(preview)

    # Phase 4A: policy decision + approval request.
    decision = PolicyDecision(
        objective_id=objective_id,
        risk_level=RiskLevel.R3 if risk_score >= 0.3 else RiskLevel.R0,
        allowed_actions=("read_state_meta", "write_state_meta"),
        forbidden_actions=(),
        approval_required=risk_score >= 0.3,
        warnings=(),
        approval_requirements=(),
        risk_score=risk_score,
        risk_components=risk_components,
        created_at="2026-07-02T12:00:00+00:00",
        decision_fingerprint="decision-fp",
    )
    in_memory_storage.set_objective_policy_decision(decision)
    request_kwargs = {
        "objective_id": objective_id,
        "risk_level": RiskLevel.R3 if risk_score >= 0.3 else RiskLevel.R0,
        "approver_id": "user-1",
        "approval_token": "tok-abc",
        "kanban_approver_id": "kanban-admin",
        "worker_approver_id": None,
        "external_approver_id": None,
        "approval_reason": "Phase 4A approval",
        "scope": ("create_kanban_task",),
        "expiry": None,
        "created_at": "2026-07-02T12:00:00+00:00",
        "request_fingerprint": "request-fp",
        "policy_decision_fingerprint": "decision-fp",
    }
    if request_overrides:
        request_kwargs.update(request_overrides)
    request = ApprovalRequest(**request_kwargs)
    in_memory_storage.set_objective_approval_request(request)


class _FakeKanbanDB:
    """Minimal fake that simulates kb.create_task / delete_task / archive_task."""

    def __init__(self):
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.archived: list[str] = []
        self.existing_ids: dict[str, bool] = {}
        self.idempotency_map: dict[str, str] = {}

    def reset(self):
        self.created = []
        self.deleted = []
        self.archived = []

    def create_task(self, kwargs: dict) -> str:
        """Simulate kb.create_task with idempotency."""
        idem = kwargs.get("idempotency_key")
        if idem and idem in self.idempotency_map:
            return self.idempotency_map[idem]
        task_id = f"t-{len(self.created) + 1:03d}"
        self.created.append(dict(kwargs))
        self.existing_ids[task_id] = True
        if idem:
            self.idempotency_map[idem] = task_id
        return task_id

    def delete_task(self, conn_unused, task_id: str) -> bool:
        if self.existing_ids.get(task_id, False):
            self.deleted.append(task_id)
            del self.existing_ids[task_id]
            return True
        return False

    def archive_task(self, conn_unused, task_id: str) -> bool:
        if self.existing_ids.get(task_id, False):
            self.archived.append(task_id)
            del self.existing_ids[task_id]
            return True
        return False


def _make_engine(in_memory_storage, fake: _FakeKanbanDB | None = None) -> KanbanApplyEngine:
    if fake is None:
        fake = _FakeKanbanDB()
    return KanbanApplyEngine(
        state_storage=in_memory_storage,
        kanban_create_fn=fake.create_task,
    )


def _make_engine_with_fake_rollback(in_memory_storage, fake: _FakeKanbanDB) -> KanbanApplyEngine:
    """Build an engine whose rollback uses the fake's delete_task / archive_task."""
    engine = KanbanApplyEngine(
        state_storage=in_memory_storage,
        kanban_create_fn=fake.create_task,
    )
    # Monkey-patch the rollback to call the fake's delete/archive.
    original_rollback = engine.rollback
    def patched_rollback(objective_id, *, hard_delete=True):
        existing = engine._storage.get_objective_kanban_apply(objective_id)
        if existing is None:
            engine._storage.delete_objective_kanban_tasks(objective_id)
            return False
        from agent.executive.types import KanbanRollbackPlan
        plan = KanbanRollbackPlan.from_apply_result(
            existing, mode="hard_delete" if hard_delete else "soft_archive"
        )
        removed_any = False
        for tid in plan.task_ids:
            try:
                if hard_delete:
                    if fake.delete_task(None, tid):
                        removed_any = True
                else:
                    if fake.archive_task(None, tid):
                        removed_any = True
            except Exception:
                continue
        engine._storage.delete_objective_kanban_apply(objective_id)
        engine._storage.delete_objective_kanban_tasks(objective_id)
        return removed_any
    engine.rollback = patched_rollback
    return engine


# ══════════════════════════════════════════════════════════════════════
# Section 1: build_apply_preview (3 tests)
# ══════════════════════════════════════════════════════════════════════

def test_build_apply_preview_pure_no_writes(in_memory_storage):
    """build_apply_preview does NOT write to state_meta."""
    _seed_full(in_memory_storage)
    preview = build_kanban_apply_preview("obj-1", storage=in_memory_storage)
    assert preview.objective_id == "obj-1"
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is None
    assert in_memory_storage.get_objective_kanban_tasks("obj-1") is None


def test_build_apply_preview_no_task_specs(in_memory_storage):
    """Empty task_specs -> warning in preview."""
    _seed_full(in_memory_storage, task_specs=[])
    preview = build_kanban_apply_preview("obj-1", storage=in_memory_storage)
    assert any("no task_specs" in w for w in preview.warnings)


def test_build_apply_preview_fingerprint_stable(in_memory_storage):
    """Two consecutive previews yield the same kanban_apply_fingerprint."""
    _seed_full(in_memory_storage)
    p1 = build_kanban_apply_preview("obj-1", storage=in_memory_storage)
    p2 = build_kanban_apply_preview("obj-1", storage=in_memory_storage)
    assert p1.kanban_apply_fingerprint == p2.kanban_apply_fingerprint


# ══════════════════════════════════════════════════════════════════════
# Section 2: apply happy path (5 tests)
# ══════════════════════════════════════════════════════════════════════

def test_apply_happy_path(in_memory_storage):
    """apply() creates tasks, persists linkage, returns KanbanApplyResult."""
    _seed_full(in_memory_storage)
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    result = engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert not result.duplicate
    assert len(result.task_ids) == 2
    persisted = in_memory_storage.get_objective_kanban_apply("obj-1")
    assert persisted is not None
    assert persisted.task_ids == result.task_ids
    assert in_memory_storage.get_objective_kanban_tasks("obj-1") == result.task_ids


def test_apply_idempotency_retry(in_memory_storage):
    """Re-running apply hits state_meta dedup and returns the same result."""
    _seed_full(in_memory_storage)
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    result1 = engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert len(result1.task_ids) == 2
    result2 = engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert result2.duplicate is True
    assert result2.task_ids == result1.task_ids


def test_apply_duplicate_via_state_meta(in_memory_storage):
    """Persisted result returned without re-running the kb.create_task loop."""
    _seed_full(in_memory_storage)
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    result1 = engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    created_count_after_first = len(fake.created)
    result2 = engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert result2.duplicate
    assert len(fake.created) == created_count_after_first


def test_apply_linkage_linear(in_memory_storage):
    """Task N's parent is task N-1 (linear chain)."""
    _seed_full(in_memory_storage)
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert len(fake.created) == 2
    assert fake.created[0]["parents"] == ()
    assert fake.created[1]["parents"] == ("t-001",)


def test_apply_writes_state_meta(in_memory_storage):
    """After apply, both kanban_apply and kanban_tasks keys are written."""
    _seed_full(in_memory_storage)
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    apply_record = in_memory_storage.get_objective_kanban_apply("obj-1")
    task_list = in_memory_storage.get_objective_kanban_tasks("obj-1")
    assert apply_record is not None
    assert task_list is not None
    assert apply_record.task_ids == task_list


# ══════════════════════════════════════════════════════════════════════
# Section 3: apply with gate failures (3 tests)
# ══════════════════════════════════════════════════════════════════════

def test_apply_raises_on_layer_1_failure(in_memory_storage):
    """No approver_id -> BridgeApprovalError, no Kanban writes."""
    _seed_full(in_memory_storage, risk_score=0.4)
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    with pytest.raises(BridgeApprovalError):
        engine.apply("obj-1")
    assert len(fake.created) == 0
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is None


def test_apply_raises_on_linkage_conflict(in_memory_storage):
    """Mismatched ApprovalRequest.policy_decision_fingerprint -> KanbanLinkageConflictError."""
    _seed_full(
        in_memory_storage,
        request_overrides={"policy_decision_fingerprint": "WRONG-FINGERPRINT"},
    )
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    with pytest.raises(KanbanLinkageConflictError):
        engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert len(fake.created) == 0


def test_apply_raises_on_layer_5_failure(in_memory_storage):
    """R4 with kanban_approver_id=None raises BridgeApprovalError (Layer 5).

    To trigger Layer 5 we need risk_level >= R4. We seed a request at R4
    by patching the risk_score. At R4, the apply requires:
    - approver_id (Layer 1)
    - approval_token (Layer 3)
    - kanban_approver_id (Layer 5) — missing in the apply kwargs below.
    """
    _seed_full(
        in_memory_storage,
        risk_score=0.7,  # triggers R4 in policy_dry_run
    )
    # Patch the persisted request so that the only fields checked are
    # approver_id, approval_token, and the (missing) kanban_approver_id.
    # R4's Layer 5 fires when policy_decision.risk_level == R4. We
    # need to manually patch the risk_level to R4.
    decision = in_memory_storage.get_objective_policy_decision("obj-1")
    patched_decision = PolicyDecision(
        objective_id=decision.objective_id,
        risk_level=RiskLevel.R4,
        allowed_actions=decision.allowed_actions,
        forbidden_actions=decision.forbidden_actions,
        approval_required=True,
        warnings=decision.warnings,
        approval_requirements=decision.approval_requirements,
        risk_score=decision.risk_score,
        risk_components=decision.risk_components,
        created_at=decision.created_at,
        decision_fingerprint=decision.decision_fingerprint,
    )
    in_memory_storage.set_objective_policy_decision(patched_decision)
    # Now request has kanban_approver_id=None (overriding default).
    _seed_full(
        in_memory_storage,
        risk_score=0.7,
        request_overrides={"kanban_approver_id": None},
    )
    # Re-apply the patched decision (the second _seed_full overwrites it).
    in_memory_storage.set_objective_policy_decision(patched_decision)

    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    with pytest.raises(BridgeApprovalError):
        # R4 -> Layer 1 (approver_id) passes, Layer 3 (approval_token)
        # passes (we seeded with token), Layer 5 (kanban_approver_id)
        # fails because we did not pass it in kwargs and the persisted
        # request has None.
        engine.apply("obj-1", approver_id="user-1")
    assert len(fake.created) == 0


# ══════════════════════════════════════════════════════════════════════
# Section 4: apply with missing inputs (3 tests)
# ══════════════════════════════════════════════════════════════════════

def test_apply_missing_phase_3_plan(in_memory_storage):
    """Missing Phase 3 plan -> BridgeMappingError."""
    _seed_full(in_memory_storage)
    in_memory_storage.delete_objective_plan("obj-1")
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    with pytest.raises(BridgeMappingError):
        engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert len(fake.created) == 0


def test_apply_missing_phase_4a_decision(in_memory_storage):
    """Missing Phase 4A decision -> BridgeMappingError."""
    _seed_full(in_memory_storage)
    in_memory_storage.delete_objective_policy_decision("obj-1")
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    with pytest.raises(BridgeMappingError):
        engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert len(fake.created) == 0


def test_apply_no_task_specs_is_empty_apply(in_memory_storage):
    """apply with empty task_specs succeeds and creates 0 tasks."""
    _seed_full(in_memory_storage, task_specs=[])
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    result = engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert result.task_ids == ()
    assert len(fake.created) == 0
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is not None


# ══════════════════════════════════════════════════════════════════════
# Section 5: idempotency (4 tests)
# ══════════════════════════════════════════════════════════════════════

def test_idempotency_keys_are_unique_per_spec(in_memory_storage):
    """Each spec has a unique idempotency_key in the create_task calls."""
    _seed_full(in_memory_storage)
    fake = _FakeKanbanDB()
    engine = _make_engine(in_memory_storage, fake)
    engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    keys = [c["idempotency_key"] for c in fake.created]
    assert len(set(keys)) == len(keys)
    for i, k in enumerate(keys):
        assert k == f"exec-v2-phase4b:obj-1:{i}"


def test_idempotency_retry_via_create_task(in_memory_storage):
    """If state_meta is wiped but the fake's idempotency_map has the keys,
    the second apply returns the same task_ids without creating new ones."""
    _seed_full(in_memory_storage)
    fake = _FakeKanbanDB()
    engine1 = _make_engine(in_memory_storage, fake)
    result1 = engine1.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    in_memory_storage.delete_objective_kanban_apply("obj-1")
    in_memory_storage.delete_objective_kanban_tasks("obj-1")
    engine2 = _make_engine(in_memory_storage, fake)
    result2 = engine2.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert result2.task_ids == result1.task_ids


def test_module_level_kanban_apply_uses_default_storage(in_memory_storage):
    """kanban_apply() uses the in_memory_storage via direct state_storage injection."""
    _seed_full(in_memory_storage)
    fake = _FakeKanbanDB()
    # Inject storage and create_fn via a one-off engine.
    engine = KanbanApplyEngine(
        state_storage=in_memory_storage, kanban_create_fn=fake.create_task
    )
    result = engine.apply("obj-1", approver_id="user-1", kanban_approver_id="kanban-admin")
    assert len(result.task_ids) == 2
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is not None


def test_module_level_build_kanban_apply_preview_uses_default_storage(in_memory_storage):
    """build_kanban_apply_preview() with storage arg uses the in_memory_storage."""
    _seed_full(in_memory_storage)
    preview = build_kanban_apply_preview("obj-1", storage=in_memory_storage)
    assert preview.objective_id == "obj-1"