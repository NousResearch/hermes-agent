"""Tests for Phase 5 Worker Dispatch — WorkerDispatchEngine integration.

18 tests covering:
* Pure dry-run (3 tests)
* Apply happy path (5 tests)
* Apply with gate failures (3 tests)
* Apply with missing inputs (3 tests)
* Idempotency (4 tests)

All tests use a fake kanban DB and a fake orchestrator factory so
no real subprocess or real Dispatcher runs.
"""

from __future__ import annotations

import json
import pytest

from agent.executive.types import (
    ApprovalRequest,
    KanbanApplyResult,
    ObjectivePlan,
    OrchestratorPlanPreview,
    PolicyDecision,
    RiskLevel,
)
from agent.executive.worker_dispatch import (
    WorkerDispatchEngine,
    worker_dispatch_dry_run,
    worker_dispatch_apply,
    worker_dispatch_rollback,
    BridgeMappingError,
    BridgeApprovalError,
    KanbanLinkageConflictError,
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


def _make_decision(
    objective_id="obj-1",
    risk_level=RiskLevel.R3,
    approval_required=True,
    decision_fingerprint="d-fp-1",
    approval_requirements=None,
):
    return PolicyDecision(
        objective_id=objective_id,
        risk_level=risk_level,
        allowed_actions=("kanban.create",),
        forbidden_actions=(),
        approval_required=approval_required,
        warnings=(),
        approval_requirements=approval_requirements or (),
        risk_score=0.5,
        risk_components={},
        created_at="2026-07-02T00:00:00Z",
        decision_fingerprint=decision_fingerprint,
    )


def _make_request(
    objective_id="obj-1",
    approval_token="tok-1",
    policy_decision_fingerprint="d-fp-1",
    expiry="2030-01-01T00:00:00Z",
    risk_level=RiskLevel.R3,
):
    return ApprovalRequest(
        objective_id=objective_id,
        risk_level=risk_level,
        approver_id="user-1",
        approval_token=approval_token,
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
        approval_reason="phase5-test",
        scope=("apply",),
        expiry=expiry,
        created_at="2026-07-02T00:00:00Z",
        request_fingerprint="r-fp-1",
        policy_decision_fingerprint=policy_decision_fingerprint,
    )


def _make_apply_result(
    objective_id="obj-1",
    task_ids=("t-1", "t-2"),
    preview_fingerprint="k-fp-1",
    decision_fingerprint="d-fp-1",
    request_fingerprint="r-fp-1",
):
    return KanbanApplyResult(
        objective_id=objective_id,
        task_ids=task_ids,
        preview_fingerprint=preview_fingerprint,
        decision_fingerprint=decision_fingerprint,
        request_fingerprint=request_fingerprint,
        result_fingerprint="res-fp-1",
        duplicate=False,
        created_at="2026-07-02T00:00:00Z",
        created_by="phase4b",
        board=None,
    )


def _seed_full(
    in_memory_storage,
    objective_id="obj-1",
    *,
    risk_level=RiskLevel.R3,
    approval_required=True,
    decision_fingerprint="d-fp-1",
    policy_decision_fingerprint="d-fp-1",
    approval_token="tok-1",
    expiry="2030-01-01T00:00:00Z",
    task_ids=("t-1", "t-2"),
    approval_requirements=None,
):
    """Seed all Phase 3+4A+4B artifacts in the in-memory storage."""
    in_memory_storage.set_objective_plan(_make_plan(objective_id))
    in_memory_storage.set_objective_orchestrator_preview(_make_preview(objective_id))
    in_memory_storage.set_objective_policy_decision(
        _make_decision(
            objective_id,
            risk_level=risk_level,
            approval_required=approval_required,
            decision_fingerprint=decision_fingerprint,
            approval_requirements=approval_requirements,
        )
    )
    in_memory_storage.set_objective_approval_request(
        _make_request(
            objective_id,
            approval_token=approval_token,
            policy_decision_fingerprint=policy_decision_fingerprint,
            expiry=expiry,
            risk_level=risk_level,
        )
    )
    in_memory_storage.set_objective_kanban_apply(
        _make_apply_result(objective_id, task_ids=task_ids)
    )


# ── Fake kanban DB ────────────────────────────────────────────

class FakeKanbanDB:
    """In-memory kanban DB. Phase 4B tasks are stored as dicts."""

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


# ── Fake orchestrator (with a factory that tracks call count) ──

class _CallCountingFactory:
    """Wraps the orchestrator factory in a callable that exposes
    `call_count` (number of times the factory was invoked).

    The factory itself returns a dict like `_try_import_orchestrator`:
    a mapping of names to objects. The engine calls
    `orch["Dispatcher"]`, `orch["BatchRunner"]`, etc.
    """

    def __init__(self, results=None, board_root=None):
        self._call_count = 0
        self._results = results
        self._board_root = board_root
        # Per-task dispatch result state.
        self._task_results = list(results or [])

    @property
    def call_count(self):
        return self._call_count

    def __call__(self):
        self._call_count += 1
        # Build fakes for this invocation.
        captured = {"dispatch_calls": []}

        class FakeDispatcher:
            def __init__(self, captured):
                self._captured = captured
            def dispatch(self, task, workers, restrictions=None):
                self._captured["dispatch_calls"].append((task, workers, restrictions))
                if factory._task_results:
                    return factory._task_results.pop(0)
                return FakeDispatchResult(
                    task_id=task.get("task_id", "?"),
                    worker_id=(workers[0]["worker_id"] if workers else "?"),
                )

        class FakeBatchRunner:
            def __init__(self, dispatcher, adapter=None):
                self._dispatcher = dispatcher
            def run_batch(self, tasks, workers, restrictions=None):
                results = []
                for t in tasks:
                    results.append(self._dispatcher.dispatch(t, workers, restrictions))
                return FakeBatchResult(
                    results=results,
                    errors=[],
                    worker_runs_started=len(results),
                )

        class FakeAdapter:
            def __init__(self, board_root=None):
                self.board_root = board_root

        factory = self
        return {
            "Dispatcher": lambda handlers=None: FakeDispatcher(captured),
            "DispatchResult": FakeDispatchResult,
            "BatchRunner": lambda dispatcher=None, adapter=None: FakeBatchRunner(dispatcher),
            "make_handlers": lambda adapter: {},
            "KanbanAdapter": lambda board_root=None: FakeAdapter(board_root),
            "KanbanTask": object,
            "run_worker_subprocess": lambda *a, **k: None,
            "WorkerRunResult": dict,
        }


class FakeDispatchResult:
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


class FakeBatchResult:
    def __init__(self, results=None, errors=None, worker_runs_started=0):
        self.results = list(results or [])
        self.errors = list(errors or [])
        self.worker_runs_started = int(worker_runs_started or 0)


def make_orchestrator_factory(results=None, board_root=None):
    """Return a _CallCountingFactory."""
    return _CallCountingFactory(results=results, board_root=board_root)


def make_kanban_db_factory(tasks=None):
    """Return a factory that yields a FakeKanbanDB."""
    db = FakeKanbanDB()
    if tasks:
        for t in tasks:
            db.tasks[t["id"]] = t
    return lambda: db


# ── Pure dry-run (3 tests) ─────────────────────────────────────


def test_dry_run_pure_no_writes(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(
            tasks=[
                {"id": "t-1", "status": "ready", "assignee": "anthropic"},
                {"id": "t-2", "status": "ready", "assignee": "minimax"},
            ]
        ),
    )
    preview = eng.dry_run("obj-1")
    assert preview.objective_id == "obj-1"
    assert preview.kanban_task_ids == ("t-1", "t-2")
    # No state_meta write.
    assert in_memory_storage.get_objective_worker_dispatch("obj-1") is None


def test_dry_run_maps_task_states(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(
            tasks=[
                {"id": "t-1", "status": "running", "assignee": "anthropic",
                 "consecutive_failures": 1},
            ]
        ),
    )
    preview = eng.dry_run("obj-1")
    assert len(preview.task_states) == 1
    assert preview.task_states[0]["state"] == "running"
    assert preview.task_states[0]["failure_count"] == 1


def test_dry_run_no_batch_runner_call(in_memory_storage):
    _seed_full(in_memory_storage)
    factory = make_orchestrator_factory()
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=factory,
        kanban_db_factory=make_kanban_db_factory(),
    )
    eng.dry_run("obj-1")
    # The fake factory was called 0 times (dry-run does not import orchestrator).
    assert factory.call_count == 0


# ── Apply happy path (5 tests) ─────────────────────────────────


def test_apply_happy_path(in_memory_storage):
    _seed_full(in_memory_storage)
    factory = make_orchestrator_factory()
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=factory,
        kanban_db_factory=make_kanban_db_factory(
            tasks=[
                {"id": "t-1", "status": "ready", "assignee": "anthropic"},
                {"id": "t-2", "status": "ready", "assignee": "anthropic"},
            ]
        ),
    )
    result = eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    assert result.objective_id == "obj-1"
    assert result.task_ids == ("t-1", "t-2")
    assert result.worker_runs_started == 2
    assert result.duplicate is False
    # State_meta written.
    assert in_memory_storage.get_objective_worker_dispatch("obj-1") is not None


def test_apply_calls_batch_runner(in_memory_storage):
    _seed_full(in_memory_storage)
    factory = make_orchestrator_factory()
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=factory,
        kanban_db_factory=make_kanban_db_factory(
            tasks=[
                {"id": "t-1", "status": "ready", "assignee": "anthropic"},
            ]
        ),
    )
    eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    # The factory was called exactly once.
    assert factory.call_count == 1


def test_apply_does_not_call_execution_router(in_memory_storage):
    _seed_full(in_memory_storage)
    import agent.executive.worker_dispatch as wd_mod
    # CODE-ONLY check (strip docstrings + comments).
    import ast
    src_text = wd_mod.__file__
    raw = open(src_text).read()
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        tree = None
    if tree is not None:
        ranges = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                    end = node.body[0].end_lineno or node.body[0].lineno
                    ranges.append((node.body[0].lineno, end))
        lines = raw.splitlines(keepends=True)
        code_lines = [
            line
            for i, line in enumerate(lines, start=1)
            if not line.lstrip().startswith("#")
            and not any(lo <= i <= hi for lo, hi in ranges)
        ]
        src = "".join(code_lines)
    else:
        src = raw
    assert "ExecutionRouter" not in src
    assert "ExecutionDispatcher" not in src
    assert "OrchestratorInterface" not in src


def test_apply_uses_only_allowed_orchestrator_apis(in_memory_storage):
    _seed_full(in_memory_storage)
    import agent.executive.worker_dispatch as wd_mod
    src = open(wd_mod.__file__).read()
    # Allowed: Dispatcher, BatchRunner, make_handlers, run_worker_subprocess, KanbanAdapter.
    assert "Dispatcher" in src
    assert "BatchRunner" in src
    assert "make_handlers" in src
    assert "run_worker_subprocess" in src
    assert "KanbanAdapter" in src


def test_apply_writes_state_meta(in_memory_storage):
    _seed_full(in_memory_storage, task_ids=("t-1",))
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(
            tasks=[{"id": "t-1", "status": "ready", "assignee": "anthropic"}]
        ),
    )
    eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    rec = in_memory_storage.get_objective_worker_dispatch("obj-1")
    assert rec is not None
    assert rec.objective_id == "obj-1"
    tasks = in_memory_storage.get_objective_worker_dispatch_tasks("obj-1")
    assert tasks is not None
    assert tasks["task_ids"] == ["t-1"]


# ── Apply with gate failures (3 tests) ────────────────────────


def test_apply_raises_on_layer_1_failure(in_memory_storage):
    """R3 with no approver_id raises BridgeApprovalError (Layer 1)."""
    _seed_full(in_memory_storage, risk_level=RiskLevel.R3)
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(),
    )
    with pytest.raises(BridgeApprovalError):
        eng.apply("obj-1")  # no approver_id, no approval_token, etc.


def test_apply_raises_on_layer_6_failure(in_memory_storage):
    """R5 with no worker_approver_id raises BridgeApprovalError (Layer 6)."""
    _seed_full(
        in_memory_storage,
        risk_level=RiskLevel.R5,
        approval_required=True,
        approval_token="tok-1",
    )
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(),
    )
    with pytest.raises(BridgeApprovalError):
        eng.apply(
            "obj-1",
            approver_id="user-1",
            approval_token="tok-1",
            kanban_approver_id="user-1",
            external_approver_id="user-1",
            # no worker_approver_id at R5 -> Layer 6 fails.
        )


def test_apply_raises_on_linkage_conflict(in_memory_storage):
    """Mismatched decision_fingerprint / policy_decision_fingerprint raises."""
    _seed_full(
        in_memory_storage,
        decision_fingerprint="d-fp-1",
        policy_decision_fingerprint="d-fp-DIFFERENT",  # mismatch!
    )
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(),
    )
    with pytest.raises((BridgeApprovalError, KanbanLinkageConflictError)):
        eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )


# ── Apply with missing inputs (3 tests) ───────────────────────


def test_apply_raises_on_missing_plan(in_memory_storage):
    # No plan seeded.
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(),
    )
    with pytest.raises(BridgeMappingError):
        eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )


def test_apply_raises_on_missing_decision(in_memory_storage):
    in_memory_storage.set_objective_plan(_make_plan())
    in_memory_storage.set_objective_orchestrator_preview(_make_preview())
    # No decision/request.
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(),
    )
    with pytest.raises(BridgeMappingError):
        eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )


def test_apply_raises_on_missing_apply_record(in_memory_storage):
    in_memory_storage.set_objective_plan(_make_plan())
    in_memory_storage.set_objective_orchestrator_preview(_make_preview())
    in_memory_storage.set_objective_policy_decision(_make_decision())
    in_memory_storage.set_objective_approval_request(_make_request())
    # No kanban apply record.
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(),
    )
    with pytest.raises(BridgeMappingError):
        eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )


# ── Idempotency (4 tests) ─────────────────────────────────────


def test_apply_idempotency_retry(in_memory_storage):
    _seed_full(in_memory_storage)
    factory = make_orchestrator_factory()
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=factory,
        kanban_db_factory=make_kanban_db_factory(
            tasks=[{"id": "t-1", "status": "ready", "assignee": "anthropic"}]
        ),
    )
    r1 = eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    initial_call_count = factory.call_count
    r2 = eng.apply(
        "obj-1",
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    # r2 should be a duplicate (returns the persisted result).
    assert r2.duplicate is True
    assert r2.objective_id == r1.objective_id
    # Factory was NOT called the second time.
    assert factory.call_count == initial_call_count


def test_module_level_worker_dispatch_dry_run(in_memory_storage):
    _seed_full(in_memory_storage, task_ids=("t-1",))
    preview = worker_dispatch_dry_run(
        "obj-1",
        storage=in_memory_storage,
        kanban_db_factory=make_kanban_db_factory(
            tasks=[{"id": "t-1", "status": "ready", "assignee": "anthropic"}]
        ),
    )
    assert preview.objective_id == "obj-1"
    assert preview.kanban_task_ids == ("t-1",)


def test_module_level_worker_dispatch_apply(in_memory_storage):
    _seed_full(in_memory_storage)
    result = worker_dispatch_apply(
        "obj-1",
        storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
        kanban_db_factory=make_kanban_db_factory(
            tasks=[{"id": "t-1", "status": "ready", "assignee": "anthropic"}]
        ),
        approver_id="user-1",
        approval_token="tok-1",
        kanban_approver_id="user-1",
        worker_approver_id="user-1",
        external_approver_id="user-1",
    )
    assert result.objective_id == "obj-1"
    assert result.worker_runs_started == 1


def test_module_level_worker_dispatch_rollback(in_memory_storage):
    _seed_full(in_memory_storage)
    kanban_db = make_kanban_db_factory(
        tasks=[{"id": "t-1", "status": "ready", "assignee": "anthropic"}]
    )()
    eng = WorkerDispatchEngine(
        state_storage=in_memory_storage,
        orchestrator_factory=make_orchestrator_factory(),
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
    # State_meta cleared.
    assert in_memory_storage.get_objective_worker_dispatch("obj-1") is None
    # Fake kanban DB archived the task.
    assert "t-1" in kanban_db.archived
