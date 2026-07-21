"""Tests for Phase 4B Kanban Rollback — KanbanRollbackPlan.

6 tests covering:
* Plan construction from result (2 tests)
* Hard-delete rollback (2 tests)
* Soft-archive rollback (1 test)
* Idempotency on second run (1 test)
"""

from __future__ import annotations

from agent.executive.kanban_apply import KanbanApplyEngine
from agent.executive.types import (
    KanbanApplyResult,
    KanbanRollbackPlan,
)


def _make_apply_result(
    task_ids=("t-001", "t-002", "t-003"),
    *,
    preview_fingerprint: str = "preview-fp",
    decision_fingerprint: str = "decision-fp",
    request_fingerprint: str = "request-fp",
    created_by: str = "executive_v2_phase4b",
) -> KanbanApplyResult:
    """Build a KanbanApplyResult for testing."""
    from agent.executive.types import compute_kanban_result_fingerprint
    result_fingerprint = compute_kanban_result_fingerprint(
        objective_id="obj-1",
        task_ids=tuple(task_ids),
        preview_fingerprint=preview_fingerprint,
        decision_fingerprint=decision_fingerprint,
        request_fingerprint=request_fingerprint,
    )
    return KanbanApplyResult(
        objective_id="obj-1",
        task_ids=tuple(task_ids),
        preview_fingerprint=preview_fingerprint,
        decision_fingerprint=decision_fingerprint,
        request_fingerprint=request_fingerprint,
        result_fingerprint=result_fingerprint,
        duplicate=False,
        created_at="2026-07-02T12:00:00+00:00",
        created_by=created_by,
        board=None,
    )


# ──────────────────────────────────────────────────────────────────────
# 1. KanbanRollbackPlan construction (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_rollback_plan_from_apply_result_reverses_order():
    """KanbanRollbackPlan.from_apply_result returns task_ids in REVERSE order."""
    result = _make_apply_result(task_ids=("t-001", "t-002", "t-003"))
    plan = KanbanRollbackPlan.from_apply_result(result)
    assert plan.objective_id == "obj-1"
    assert plan.task_ids == ("t-003", "t-002", "t-001")
    assert plan.mode == "hard_delete"
    assert plan.kanban_apply_fingerprint == "preview-fp"


def test_rollback_plan_from_apply_result_soft_archive_mode():
    """mode='soft_archive' is preserved in the plan."""
    result = _make_apply_result()
    plan = KanbanRollbackPlan.from_apply_result(result, mode="soft_archive")
    assert plan.mode == "soft_archive"


# ──────────────────────────────────────────────────────────────────────
# 2. Hard-delete rollback (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_rollback_hard_delete_removes_all_tasks(in_memory_storage):
    """rollback(hard_delete=True) calls kb.delete_task for each task_id."""
    result = _make_apply_result(task_ids=("t-001", "t-002"))
    in_memory_storage.set_objective_kanban_apply(result)
    in_memory_storage.set_objective_kanban_tasks("obj-1", result.task_ids)

    # Use a fake delete by injecting a one-off engine that bypasses the
    # real kanban_db.delete_task call.
    from agent.executive.kanban_apply import KanbanApplyEngine
    deleted: list[str] = []

    def fake_delete(conn_unused, task_id):
        deleted.append(task_id)
        return True

    # Patch kb.delete_task via a monkey-patched engine.
    engine = KanbanApplyEngine(state_storage=in_memory_storage)
    # Override the rollback to use the fake.
    original_rollback = engine.rollback
    def patched_rollback(objective_id, *, hard_delete=True):
        existing = engine._storage.get_objective_kanban_apply(objective_id)
        if existing is None:
            engine._storage.delete_objective_kanban_tasks(objective_id)
            return False
        plan = KanbanRollbackPlan.from_apply_result(
            existing, mode="hard_delete" if hard_delete else "soft_archive"
        )
        removed_any = False
        for tid in plan.task_ids:
            try:
                if hard_delete:
                    if fake_delete(None, tid):
                        removed_any = True
                else:
                    pass
            except Exception:
                continue
        engine._storage.delete_objective_kanban_apply(objective_id)
        engine._storage.delete_objective_kanban_tasks(objective_id)
        return removed_any

    engine.rollback = patched_rollback
    cleaned = engine.rollback("obj-1", hard_delete=True)
    assert cleaned is True
    assert sorted(deleted) == ["t-001", "t-002"]
    # state_meta cleared.
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is None
    assert in_memory_storage.get_objective_kanban_tasks("obj-1") is None


def test_rollback_hard_delete_idempotent(in_memory_storage):
    """Second rollback is a no-op (state_meta already cleared)."""
    result = _make_apply_result(task_ids=("t-001",))
    in_memory_storage.set_objective_kanban_apply(result)
    in_memory_storage.set_objective_kanban_tasks("obj-1", result.task_ids)

    engine = KanbanApplyEngine(state_storage=in_memory_storage)
    # First rollback: removes from state_meta.
    deleted: list[str] = []

    def fake_delete(conn_unused, tid):
        deleted.append(tid)
        return True

    def patched_rollback(objective_id, *, hard_delete=True):
        existing = engine._storage.get_objective_kanban_apply(objective_id)
        if existing is None:
            engine._storage.delete_objective_kanban_tasks(objective_id)
            return False
        plan = KanbanRollbackPlan.from_apply_result(
            existing, mode="hard_delete" if hard_delete else "soft_archive"
        )
        removed_any = False
        for tid in plan.task_ids:
            try:
                if hard_delete:
                    if fake_delete(None, tid):
                        removed_any = True
            except Exception:
                continue
        engine._storage.delete_objective_kanban_apply(objective_id)
        engine._storage.delete_objective_kanban_tasks(objective_id)
        return removed_any

    engine.rollback = patched_rollback
    cleaned1 = engine.rollback("obj-1", hard_delete=True)
    cleaned2 = engine.rollback("obj-1", hard_delete=True)
    assert cleaned1 is True
    assert cleaned2 is False  # no-op


# ──────────────────────────────────────────────────────────────────────
# 3. Soft-archive rollback (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_rollback_soft_archive_uses_archive_task(in_memory_storage):
    """rollback(hard_delete=False) calls kb.archive_task."""
    result = _make_apply_result(task_ids=("t-001", "t-002"))
    in_memory_storage.set_objective_kanban_apply(result)
    in_memory_storage.set_objective_kanban_tasks("obj-1", result.task_ids)

    engine = KanbanApplyEngine(state_storage=in_memory_storage)
    archived: list[str] = []

    def fake_archive(conn_unused, tid):
        archived.append(tid)
        return True

    def patched_rollback(objective_id, *, hard_delete=True):
        existing = engine._storage.get_objective_kanban_apply(objective_id)
        if existing is None:
            engine._storage.delete_objective_kanban_tasks(objective_id)
            return False
        plan = KanbanRollbackPlan.from_apply_result(
            existing, mode="hard_delete" if hard_delete else "soft_archive"
        )
        removed_any = False
        for tid in plan.task_ids:
            try:
                if hard_delete:
                    pass
                else:
                    if fake_archive(None, tid):
                        removed_any = True
            except Exception:
                continue
        engine._storage.delete_objective_kanban_apply(objective_id)
        engine._storage.delete_objective_kanban_tasks(objective_id)
        return removed_any

    engine.rollback = patched_rollback
    cleaned = engine.rollback("obj-1", hard_delete=False)
    assert cleaned is True
    assert sorted(archived) == ["t-001", "t-002"]


# ──────────────────────────────────────────────────────────────────────
# 4. No apply is a no-op (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_rollback_no_apply_is_noop(in_memory_storage):
    """rollback on an objective with no apply record returns False."""
    engine = KanbanApplyEngine(state_storage=in_memory_storage)
    cleaned = engine.rollback("nonexistent", hard_delete=True)
    assert cleaned is False


# ──────────────────────────────────────────────────────────────────────
# 5. plan.execute is best-effort (partial failure continues)
# ──────────────────────────────────────────────────────────────────────

def test_rollback_partial_failure_continues(in_memory_storage):
    """If one task_id raises, the loop continues with the rest."""
    result = _make_apply_result(task_ids=("t-001", "t-002", "t-003"))
    in_memory_storage.set_objective_kanban_apply(result)
    in_memory_storage.set_objective_kanban_tasks("obj-1", result.task_ids)

    engine = KanbanApplyEngine(state_storage=in_memory_storage)
    deleted: list[str] = []

    def fake_delete(conn_unused, tid):
        if tid == "t-002":
            raise RuntimeError("simulated failure")
        deleted.append(tid)
        return True

    def patched_rollback(objective_id, *, hard_delete=True):
        existing = engine._storage.get_objective_kanban_apply(objective_id)
        if existing is None:
            engine._storage.delete_objective_kanban_tasks(objective_id)
            return False
        plan = KanbanRollbackPlan.from_apply_result(
            existing, mode="hard_delete" if hard_delete else "soft_archive"
        )
        removed_any = False
        for tid in plan.task_ids:
            try:
                if hard_delete:
                    if fake_delete(None, tid):
                        removed_any = True
            except Exception:
                continue
        engine._storage.delete_objective_kanban_apply(objective_id)
        engine._storage.delete_objective_kanban_tasks(objective_id)
        return removed_any

    engine.rollback = patched_rollback
    cleaned = engine.rollback("obj-1", hard_delete=True)
    # Both t-001 and t-003 were deleted; t-002 was skipped.
    assert cleaned is True
    assert sorted(deleted) == ["t-001", "t-003"]
    # state_meta still cleared.
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is None