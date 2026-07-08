"""Tests for Phase 5 Worker Dispatch — rollback / cancel path.

6 tests covering:
* WorkerDispatchRollbackPlan construction (2 tests)
* Soft archive (default mode) (2 tests)
* Hard delete (opt-in mode) (1 test)
* Idempotency (1 test)

All tests use a fake kanban DB so no real kb.archive_task /
kb.delete_task calls happen.
"""

from __future__ import annotations

import pytest

from agent.executive.types import (
    ApprovalRequest,
    KanbanApplyResult,
    ObjectivePlan,
    OrchestratorPlanPreview,
    PolicyDecision,
    RiskLevel,
    WorkerDispatchResult,
    WorkerDispatchRollbackPlan,
)
from agent.executive.worker_dispatch import (
    WorkerDispatchEngine,
    worker_dispatch_rollback,
)


# ── Test fixtures ──────────────────────────────────────────────

def _make_plan(objective_id="obj-1"):
    return ObjectivePlan(
        objective_id=objective_id,
        subgoals=(),
        plan_fingerprint="plan-fp-1",
        created_at="2026-07-02T00:00:00Z",
    )


def _make_preview(objective_id="obj-1"):
    return OrchestratorPlanPreview(
        objective_id=objective_id,
        plan=_make_plan(objective_id),
        task_specs=(),
        warnings=(),
        requires_approval=True,
        risk_score=0.5,
        preview_fingerprint="plan-fp-1",
        created_at="2026-07-02T00:00:00Z",
    )


def _make_decision(objective_id="obj-1", risk_level=RiskLevel.R3):
    return PolicyDecision(
        objective_id=objective_id,
        risk_level=risk_level,
        allowed_actions=("kanban.create",),
        forbidden_actions=(),
        approval_required=True,
        warnings=(),
        approval_requirements=(),
        risk_score=0.5,
        risk_components={},
        created_at="2026-07-02T00:00:00Z",
        decision_fingerprint="d-fp-1",
    )


def _make_request(objective_id="obj-1", risk_level=RiskLevel.R3):
    return ApprovalRequest(
        objective_id=objective_id,
        risk_level=risk_level,
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
        approval_reason="phase5-rollback-test",
        scope=("apply",),
        expiry="2030-01-01T00:00:00Z",
        created_at="2026-07-02T00:00:00Z",
        request_fingerprint="r-fp-1",
        policy_decision_fingerprint="d-fp-1",
    )


def _make_apply_result(objective_id="obj-1", task_ids=("t-1", "t-2")):
    return KanbanApplyResult(
        objective_id=objective_id,
        task_ids=task_ids,
        preview_fingerprint="k-fp-1",
        decision_fingerprint="d-fp-1",
        request_fingerprint="r-fp-1",
        result_fingerprint="res-fp-1",
        duplicate=False,
        created_at="2026-07-02T00:00:00Z",
        created_by="phase4b",
        board=None,
    )


def _seed_full(in_memory_storage, objective_id="obj-1", task_ids=("t-1", "t-2")):
    in_memory_storage.set_objective_plan(_make_plan(objective_id))
    in_memory_storage.set_objective_orchestrator_preview(_make_preview(objective_id))
    in_memory_storage.set_objective_policy_decision(_make_decision(objective_id))
    in_memory_storage.set_objective_approval_request(_make_request(objective_id))
    in_memory_storage.set_objective_kanban_apply(
        _make_apply_result(objective_id, task_ids=task_ids)
    )


# ── Fake kanban DB ────────────────────────────────────────────

class FakeKanbanDB:
    def __init__(self):
        self.tasks: dict = {}
        self.archived: list = []
        self.deleted: list = []

    def get_task(self, task_id):
        return self.tasks.get(task_id)

    def list_tasks(self, **kwargs):
        return list(self.tasks.values())

    def archive_task(self, task_id):
        self.archived.append(task_id)
        self.tasks.pop(task_id, None)

    def delete_task(self, task_id):
        self.deleted.append(task_id)
        self.tasks.pop(task_id, None)

    def connect_closing(self):
        class _Ctx:
            def __enter__(self_inner):
                return self
            def __exit__(self_inner, *args):
                pass
        return _Ctx()


class _CallCountingFactory:
    def __init__(self, results=None):
        self._call_count = 0
        self._results = list(results or [])

    @property
    def call_count(self):
        return self._call_count

    def __call__(self):
        self._call_count += 1
        d_capture = []

        class _FakeDispatchResult:
            def __init__(self, task_id, worker_id, action_executed="RUN_WORKER"):
                self.task_id = task_id
                self.worker_id = worker_id
                self.action_executed = action_executed
                self.decision = {"next_action": "RUN_WORKER", "selected_worker": worker_id}
                self.timestamp = "2026-07-02T00:00:00Z"
                self.trace_line = {"trace_id": "trace-1"}

            def to_dict(self):
                return {
                    "task_id": self.task_id,
                    "worker_id": self.worker_id,
                    "action_executed": self.action_executed,
                    "decision": self.decision,
                    "timestamp": self.timestamp,
                    "trace_line": self.trace_line,
                }

        class _FakeBatchResult:
            def __init__(self, results=None, errors=None, worker_runs_started=0):
                self.results = list(results or [])
                self.errors = list(errors or [])
                self.worker_runs_started = int(worker_runs_started or 0)

        class _FakeAdapter:
            def __init__(self, board_root=None):
                self.board_root = board_root

        class _D:
            def __init__(self, cap):
                self._cap = cap

            def dispatch(self, task, workers, restrictions=None):
                self._cap.append((task, workers, restrictions))
                if factory._results:
                    return factory._results.pop(0)
                return _FakeDispatchResult(
                    task_id=task.get("task_id", "?"),
                    worker_id=(workers[0]["worker_id"] if workers else "?"),
                )

        class _BR:
            def __init__(self, dispatcher, adapter=None):
                self._d = dispatcher

            def run_batch(self, tasks, workers, restrictions=None):
                results = [self._d.dispatch(t, workers, restrictions) for t in tasks]
                return _FakeBatchResult(results=results, errors=[], worker_runs_started=len(results))

        factory = self
        return {
            "Dispatcher": lambda handlers=None: _D(d_capture),
            "DispatchResult": _FakeDispatchResult,
            "BatchRunner": lambda dispatcher=None, adapter=None: _BR(dispatcher),
            "make_handlers": lambda adapter: {},
            "KanbanAdapter": lambda board_root=None: _FakeAdapter(),
            "KanbanTask": object,
            "run_worker_subprocess": lambda *a, **k: None,
            "WorkerRunResult": dict,
        }


def make_kanban_db_factory(tasks=None):
    db = FakeKanbanDB()
    if tasks:
        for t in tasks:
            db.tasks[t["id"]] = t
    return lambda: db


# ── Tests ──────────────────────────────────────────────────────


def test_rollback_plan_from_dispatch_record():
    """Constructs a WorkerDispatchRollbackPlan from a WorkerDispatchResult."""
    rec = WorkerDispatchResult(
        objective_id="obj-1",
        task_ids=("t-1", "t-2", "t-3"),
        worker_runs=(),
        worker_runs_started=3,
        worker_runs_failed=0,
        dispatch_fingerprint="df-1",
        decision_fingerprint="df-1",
        request_fingerprint="rf-1",
        kanban_apply_fingerprint="kf-1",
        duplicate=False,
        errors=(),
        created_at="2026-07-02T00:00:00Z",
    )
    plan = WorkerDispatchRollbackPlan.from_dispatch_record(rec, mode="archive")
    assert plan.objective_id == "obj-1"
    # Task_ids are reversed so the most-recently-created is rolled back first.
    assert plan.task_ids == ("t-3", "t-2", "t-1")
    assert plan.dispatch_fingerprint == "df-1"
    assert plan.mode == "archive"

    plan_hard = WorkerDispatchRollbackPlan.from_dispatch_record(rec, mode="hard_delete")
    assert plan_hard.mode == "hard_delete"


def test_rollback_plan_invalid_mode_raises():
    rec = WorkerDispatchResult(
        objective_id="obj-1",
        task_ids=("t-1",),
        worker_runs=(),
        worker_runs_started=1,
        worker_runs_failed=0,
        dispatch_fingerprint="df-1",
        decision_fingerprint="df-1",
        request_fingerprint="rf-1",
        kanban_apply_fingerprint="kf-1",
        duplicate=False,
        errors=(),
        created_at="2026-07-02T00:00:00Z",
    )
    with pytest.raises(ValueError):
        WorkerDispatchRollbackPlan.from_dispatch_record(rec, mode="bogus")


def test_rollback_archive_removes_all_tasks(in_memory_storage):
    """rollback(hard_delete=False) calls kb.archive_task for each."""
    _seed_full(in_memory_storage, task_ids=("t-1", "t-2"))
    kanban_db = make_kanban_db_factory(
        tasks=[
            {"id": "t-1", "status": "ready", "assignee": "anthropic"},
            {"id": "t-2", "status": "ready", "assignee": "anthropic"},
        ]
    )()
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=_CallCountingFactory(),
        kanban_db_factory=lambda: kanban_db,
    )
    eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    cleaned = eng.rollback("obj-1", hard_delete=False)
    assert cleaned is True
    # archive_task was called for both tasks.
    assert "t-1" in kanban_db.archived
    assert "t-2" in kanban_db.archived
    # delete_task was NOT called.
    assert kanban_db.deleted == []


def test_rollback_hard_delete_removes_all_tasks(in_memory_storage):
    """rollback(hard_delete=True) calls kb.delete_task (opt-in)."""
    _seed_full(in_memory_storage, task_ids=("t-1",))
    kanban_db = make_kanban_db_factory(
        tasks=[{"id": "t-1", "status": "ready", "assignee": "anthropic"}]
    )()
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=_CallCountingFactory(),
        kanban_db_factory=lambda: kanban_db,
    )
    eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    cleaned = eng.rollback("obj-1", hard_delete=True)
    assert cleaned is True
    assert "t-1" in kanban_db.deleted
    assert kanban_db.archived == []


def test_rollback_idempotent(in_memory_storage):
    """Second rollback returns False (nothing to do)."""
    _seed_full(in_memory_storage, task_ids=("t-1",))
    kanban_db = make_kanban_db_factory(
        tasks=[{"id": "t-1", "status": "ready", "assignee": "anthropic"}]
    )()
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=_CallCountingFactory(),
        kanban_db_factory=lambda: kanban_db,
    )
    eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    cleaned1 = eng.rollback("obj-1", hard_delete=False)
    cleaned2 = eng.rollback("obj-1", hard_delete=False)
    assert cleaned1 is True
    assert cleaned2 is False  # second call: no record, no-op
    # State_meta is empty.
    assert in_memory_storage.get_objective_worker_dispatch("obj-1") is None
    assert in_memory_storage.get_objective_worker_dispatch_tasks("obj-1") is None


def test_rollback_no_record_is_noop(in_memory_storage):
    """Rollback on a non-existent objective returns False (idempotent)."""
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=_CallCountingFactory(),
        kanban_db_factory=make_kanban_db_factory(),
    )
    cleaned = eng.rollback("nonexistent-obj", hard_delete=False)
    assert cleaned is False
