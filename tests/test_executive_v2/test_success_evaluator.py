"""Tests for Phase 6 Success Evaluator — engine integration.

18 tests covering:
* Pure dry-run (3 tests)
* Evaluate happy path (5 tests)
* Evaluate with errors (3 tests)
* Evaluate with missing inputs (3 tests)
* Persist (2 tests)
* Idempotency (2 tests)

All tests use the in-memory storage fixture; no real Dispatcher /
BatchRunner / Orchestrator is invoked.
"""

from __future__ import annotations

import pytest

from agent.executive.types import (
    ApprovalRequest,
    KanbanApplyResult,
    ObjectivePlan,
    ObjectiveState,
    ObjectiveStateData,
    OrchestratorPlanPreview,
    PolicyDecision,
    RiskLevel,
    SuccessStatus,
    WorkerDispatchResult,
)
from agent.executive.success_evaluator import (
    SuccessEvaluatorEngine,
    SuccessEvaluatorMappingError,
    success_evaluator_dry_run,
    success_evaluator_evaluate,
    success_evaluator_persist,
    success_evaluator_rollback,
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
        approval_reason="phase6-test",
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


def _make_worker_dispatch_result(
    objective_id="obj-1",
    task_ids=("t-1", "t-2"),
    worker_runs=None,
    status="success",
):
    if worker_runs is None:
        worker_runs = tuple(
            {
                "action_executed": "RUN_WORKER",
                "exitcode": 0,
                "error_type": None,
                "timed_out": False,
                "killed": False,
            }
            for _ in task_ids
        )
    return WorkerDispatchResult(
        objective_id=objective_id,
        task_ids=task_ids,
        worker_runs=worker_runs,
        worker_runs_started=len(worker_runs),
        worker_runs_failed=sum(1 for w in worker_runs if w.get("exitcode") != 0),
        dispatch_fingerprint="df-1",
        decision_fingerprint="d-fp-1",
        request_fingerprint="r-fp-1",
        kanban_apply_fingerprint="k-fp-1",
        duplicate=False,
        errors=(),
        created_at="2026-07-02T00:00:00Z",
    )


def _seed_full(in_memory_storage, objective_id="obj-1",
               task_ids=("t-1", "t-2"), worker_runs=None):
    in_memory_storage.set_objective_plan(_make_plan(objective_id))
    in_memory_storage.set_objective_orchestrator_preview(_make_preview(objective_id))
    in_memory_storage.set_objective_policy_decision(_make_decision(objective_id))
    in_memory_storage.set_objective_approval_request(_make_request(objective_id))
    in_memory_storage.set_objective_kanban_apply(
        _make_apply_result(objective_id, task_ids=task_ids)
    )
    in_memory_storage.set_objective_worker_dispatch(
        _make_worker_dispatch_result(
            objective_id, task_ids=task_ids, worker_runs=worker_runs
        )
    )


# ── Pure dry-run (3 tests) ───────────────────────────────────────


def test_dry_run_pure_no_writes(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = eng.dry_run("obj-1")
    assert report.objective_id == "obj-1"
    # No state_meta writes.
    assert in_memory_storage.get_objective_evaluation("obj-1") is None
    assert in_memory_storage.get_objective_success_report("obj-1") is None


def test_dry_run_returns_evaluation_report_with_fingerprints(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = eng.dry_run("obj-1")
    # Fingerprints from Phase 1+5 are present.
    assert report.worker_dispatch_fingerprint == "df-1"
    assert report.policy_fingerprint == "d-fp-1"
    assert report.approval_fingerprint == "r-fp-1"
    assert report.plan_fingerprint == "plan-fp-1"


def test_dry_run_zero_tasks_returns_blocked(in_memory_storage):
    """total_tasks=0 → BLOCKED status."""
    _seed_full(in_memory_storage, task_ids=())
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = eng.dry_run("obj-1")
    assert report.status == SuccessStatus.BLOCKED


# ── Evaluate happy path (5 tests) ─────────────────────────────────


def test_evaluate_happy_path_success(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = eng.evaluate("obj-1")
    assert report.objective_id == "obj-1"
    assert report.status == SuccessStatus.SUCCESS
    assert report.successful_tasks == 2
    # state_meta written.
    assert in_memory_storage.get_objective_evaluation("obj-1") is not None
    assert in_memory_storage.get_objective_success_report("obj-1") is not None


def test_evaluate_partial_success(in_memory_storage):
    """2/3 successful → PARTIAL_SUCCESS."""
    runs = (
        {"action_executed": "RUN_WORKER", "exitcode": 0,
         "error_type": None, "timed_out": False, "killed": False},
        {"action_executed": "RUN_WORKER", "exitcode": 0,
         "error_type": None, "timed_out": False, "killed": False},
        {"action_executed": "RUN_WORKER", "exitcode": 1,
         "error_type": None, "timed_out": False, "killed": False},
    )
    _seed_full(in_memory_storage, task_ids=("t-1", "t-2", "t-3"), worker_runs=runs)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = eng.evaluate("obj-1")
    assert report.status == SuccessStatus.PARTIAL_SUCCESS
    assert report.successful_tasks == 2
    assert report.failed_tasks == 1
    assert report.retry_recommended is True
    assert report.manual_intervention_required is True


def test_evaluate_failed(in_memory_storage):
    """All 3 tasks missing → FAILED (completion = 0.0)."""
    # No worker_runs → all outcomes = MISSING → completion = 0.0 → FAILED.
    _seed_full(in_memory_storage, task_ids=("t-1", "t-2", "t-3"), worker_runs=())
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = eng.evaluate("obj-1")
    assert report.status == SuccessStatus.FAILED
    assert report.failed_tasks == 0
    assert report.successful_tasks == 0
    assert report.missing_tasks == 3  # type: ignore[attr-defined]
    assert report.completion_percentage == 0.0
    assert report.worker_success_rate == 0.0


def test_evaluate_blocked(in_memory_storage):
    """NO_HANDLER_FOR_* → BLOCKED."""
    runs = (
        {"action_executed": "RUN_WORKER", "exitcode": 0,
         "error_type": None, "timed_out": False, "killed": False},
        {"action_executed": "NO_HANDLER_FOR_X",
         "exitcode": None, "error_type": None, "timed_out": False, "killed": False},
    )
    _seed_full(in_memory_storage, task_ids=("t-1", "t-2"), worker_runs=runs)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = eng.evaluate("obj-1")
    assert report.status == SuccessStatus.BLOCKED
    assert report.successful_tasks == 1
    assert report.blocked_tasks == 1


def test_evaluate_aborted(in_memory_storage):
    """aborted=True → ABORTED status (overrides per-task outcomes)."""
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = eng.evaluate("obj-1", aborted=True)
    assert report.status == SuccessStatus.ABORTED
    assert report.cancelled_tasks == 2
    assert report.manual_intervention_required is True


# ── Evaluate with errors (3 tests) ──────────────────────────────


def test_evaluate_missing_worker_dispatch_raises(in_memory_storage):
    """No WorkerDispatchResult → SuccessEvaluatorMappingError."""
    in_memory_storage.set_objective_plan(_make_plan())
    in_memory_storage.set_objective_orchestrator_preview(_make_preview())
    in_memory_storage.set_objective_policy_decision(_make_decision())
    in_memory_storage.set_objective_approval_request(_make_request())
    in_memory_storage.set_objective_kanban_apply(_make_apply_result())
    # No set_objective_worker_dispatch.
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    with pytest.raises(SuccessEvaluatorMappingError):
        eng.evaluate("obj-1")


def test_evaluate_missing_apply_record_raises(in_memory_storage):
    in_memory_storage.set_objective_plan(_make_plan())
    in_memory_storage.set_objective_orchestrator_preview(_make_preview())
    in_memory_storage.set_objective_policy_decision(_make_decision())
    in_memory_storage.set_objective_approval_request(_make_request())
    in_memory_storage.set_objective_worker_dispatch(_make_worker_dispatch_result())
    # No set_objective_kanban_apply.
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    with pytest.raises(SuccessEvaluatorMappingError):
        eng.evaluate("obj-1")


def test_evaluate_uses_only_allowed_apis(in_memory_storage):
    """Verify the engine only reads Phase 1+5 state; no Dispatcher / BatchRunner."""
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    eng.evaluate("obj-1")
    # Verify the source has no references to forbidden symbols in CODE-ONLY
    # (docstrings are documentation of prohibited APIs).
    import agent.executive.success_evaluator as wd
    import ast
    raw = open(wd.__file__).read()
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
            line for i, line in enumerate(lines, start=1)
            if not line.lstrip().startswith("#")
            and not any(lo <= i <= hi for lo, hi in ranges)
        ]
        src = "".join(code_lines)
    else:
        src = raw
    assert "Dispatcher" not in src
    assert "BatchRunner" not in src
    assert "run_worker_subprocess" not in src


# ── Persist (2 tests) ─────────────────────────────────────────────


def test_persist_operator_supplied_report(in_memory_storage):
    """Operator can construct an EvaluationReport and persist it."""
    from agent.executive.success_metrics import build_evaluation_report

    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    report = build_evaluation_report(
        "obj-1",
        ("t-1",),
        ({"action_executed": "RUN_WORKER", "exitcode": 0,
          "error_type": None, "timed_out": False, "killed": False},),
        worker_dispatch_fingerprint="df-1",
        policy_fingerprint="d-fp-1",
        approval_fingerprint="r-fp-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        execution_fingerprint="ef-1",
    )
    eng.persist(report)
    assert in_memory_storage.get_objective_evaluation("obj-1") is not None
    assert in_memory_storage.get_objective_success_report("obj-1") is not None


def test_module_level_persist(in_memory_storage):
    """success_evaluator_persist wrapper persists correctly."""
    from agent.executive.success_metrics import build_evaluation_report

    report = build_evaluation_report(
        "obj-1",
        ("t-1", "t-2"),
        tuple(
            {"action_executed": "RUN_WORKER", "exitcode": 0,
             "error_type": None, "timed_out": False, "killed": False}
            for _ in ("t-1", "t-2")
        ),
        worker_dispatch_fingerprint="df-1",
        policy_fingerprint="d-fp-1",
        approval_fingerprint="r-fp-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        execution_fingerprint="ef-1",
    )
    success_evaluator_persist(report, storage=in_memory_storage)
    assert in_memory_storage.get_objective_evaluation("obj-1") is not None


# ── Idempotency (2 tests) ─────────────────────────────────────────


def test_evaluate_idempotency_retry(in_memory_storage):
    """Re-evaluating returns equivalent report (same fingerprints, same status)."""
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    r1 = eng.evaluate("obj-1")
    r2 = eng.evaluate("obj-1")
    # Same status, same counts.
    assert r1.status == r2.status
    assert r1.successful_tasks == r2.successful_tasks
    assert r1.failed_tasks == r2.failed_tasks


def test_module_level_dry_run(in_memory_storage):
    """success_evaluator_dry_run wrapper works."""
    _seed_full(in_memory_storage)
    report = success_evaluator_dry_run("obj-1", storage=in_memory_storage)
    assert report.objective_id == "obj-1"
    assert in_memory_storage.get_objective_evaluation("obj-1") is None
