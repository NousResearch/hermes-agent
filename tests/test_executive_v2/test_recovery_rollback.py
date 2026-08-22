"""Tests for Phase 7 Recovery Engine — rollback path.

6 tests covering:
* Rollback no record is noop (1 test)
* Rollback idempotent (1 test)
* Rollback removes diagnosis key (1 test)
* Rollback removes plan key (1 test)
* Rollback does not touch Phase 6 state (1 test)
* Rollback does not touch kanban DB (1 test)
"""

from __future__ import annotations

import pytest

from agent.executive.recovery_engine import (
    ObjectiveRecoveryEngine,
    RecoveryMappingError,
    recovery_rollback,
)
from agent.executive.types import (
    ApprovalRequest,
    EvaluationReport,
    KanbanApplyResult,
    ObjectivePlan,
    OrchestratorPlanPreview,
    PolicyDecision,
    RiskLevel,
    SuccessStatus,
    SuccessMetricBreakdown,
    WorkerDispatchResult,
)


def _make_evaluation():
    return EvaluationReport(
        objective_id="obj-1",
        execution_fingerprint="ef-1",
        worker_dispatch_fingerprint="wdf-1",
        policy_fingerprint="pf-1",
        approval_fingerprint="af-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        status=SuccessStatus.PARTIAL_SUCCESS,
        completion_percentage=0.667,
        successful_tasks=2, failed_tasks=1, blocked_tasks=0, cancelled_tasks=0,
        worker_success_rate=0.667, evidence_score=1.0, confidence_score=0.833,
        retry_recommended=True, retry_reason="partial",
        manual_intervention_required=False, remaining_tasks=("t-3",),
        summary="partial",
        metrics=SuccessMetricBreakdown(
            successful_tasks=2, failed_tasks=1, blocked_tasks=0, cancelled_tasks=0,
            missing_tasks=0, total_tasks=3, per_task_completion_sum=2.5, coverage=1.0,
            worker_success_rate=0.667, mean_score=0.667, evidence_score=1.0,
            confidence_score=0.833, completion_percentage=0.667),
        created_at="2026-07-02T00:00:00Z", created_by="phase6",
    )


def _seed_full(in_memory_storage, objective_id="obj-1"):
    in_memory_storage.set_objective_plan(ObjectivePlan(
        objective_id=objective_id, subgoals=(), plan_fingerprint="plan-fp-1",
        created_at="2026-07-02T00:00:00Z"))
    in_memory_storage.set_objective_orchestrator_preview(OrchestratorPlanPreview(
        objective_id=objective_id,
        plan=in_memory_storage.get_objective_plan(objective_id),
        task_specs=(), warnings=(), requires_approval=True, risk_score=0.5,
        preview_fingerprint="plan-fp-1", created_at="2026-07-02T00:00:00Z"))
    in_memory_storage.set_objective_policy_decision(PolicyDecision(
        objective_id=objective_id, risk_level=RiskLevel.R3,
        allowed_actions=("kanban.create",), forbidden_actions=(),
        approval_required=True, warnings=(), approval_requirements=(),
        risk_score=0.5, risk_components={}, created_at="2026-07-02T00:00:00Z",
        decision_fingerprint="d-fp-1"))
    in_memory_storage.set_objective_approval_request(ApprovalRequest(
        objective_id=objective_id, risk_level=RiskLevel.R3, approver_id="user-1",
        approval_token="tok-1", kanban_approver_id="user-1",
        worker_approver_id="user-1", external_approver_id="user-1",
        approval_reason="phase7-test", scope=("apply",), expiry="2030-01-01T00:00:00Z",
        created_at="2026-07-02T00:00:00Z", request_fingerprint="r-fp-1",
        policy_decision_fingerprint="d-fp-1"))
    in_memory_storage.set_objective_kanban_apply(KanbanApplyResult(
        objective_id=objective_id, task_ids=("t-1", "t-2", "t-3"),
        preview_fingerprint="k-fp-1", decision_fingerprint="d-fp-1",
        request_fingerprint="r-fp-1", result_fingerprint="res-fp-1",
        duplicate=False, created_at="2026-07-02T00:00:00Z", created_by="phase4b",
        board=None))
    ok_run = {
        "action_executed": "RUN_WORKER", "exitcode": 0, "error_type": None,
        "timed_out": False, "killed": False,
    }
    failed_run = {
        "action_executed": "RUN_WORKER", "exitcode": 1, "error_type": None,
        "timed_out": False, "killed": False,
    }
    in_memory_storage.set_objective_worker_dispatch(WorkerDispatchResult(
        objective_id=objective_id, task_ids=("t-1", "t-2", "t-3"),
        worker_runs=(ok_run, ok_run, failed_run),
        worker_runs_started=3, worker_runs_failed=1,
        dispatch_fingerprint="df-1", decision_fingerprint="d-fp-1",
        request_fingerprint="r-fp-1", kanban_apply_fingerprint="k-fp-1",
        duplicate=False, errors=(), created_at="2026-07-02T00:00:00Z"))
    in_memory_storage.set_objective_evaluation(_make_evaluation())


def test_rollback_no_record_is_noop(in_memory_storage):
    """No recovery record → returns False."""
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    result = eng.rollback("obj-1")
    assert result is False


def test_rollback_idempotent(in_memory_storage):
    """Second call returns False (idempotent)."""
    _seed_full(in_memory_storage)
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    eng.evaluate("obj-1")
    # First rollback deletes.
    assert eng.rollback("obj-1") is True
    # Second rollback: no record to delete.
    assert eng.rollback("obj-1") is False


def test_rollback_removes_diagnosis_key(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    eng.evaluate("obj-1")
    assert in_memory_storage.get_objective_recovery_diagnosis("obj-1") is not None
    eng.rollback("obj-1")
    assert in_memory_storage.get_objective_recovery_diagnosis("obj-1") is None


def test_rollback_removes_plan_key(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    eng.evaluate("obj-1")
    assert in_memory_storage.get_objective_recovery_plan("obj-1") is not None
    eng.rollback("obj-1")
    assert in_memory_storage.get_objective_recovery_plan("obj-1") is None


def test_rollback_does_not_touch_phase_6_state(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    eng.evaluate("obj-1")
    # Phase 6 state should still be present.
    assert in_memory_storage.get_objective_evaluation("obj-1") is not None
    eng.rollback("obj-1")
    # Phase 6 state should be untouched.
    assert in_memory_storage.get_objective_evaluation("obj-1") is not None
    assert in_memory_storage.get_objective_worker_dispatch("obj-1") is not None
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is not None


def test_rollback_does_not_touch_kanban_db(in_memory_storage):
    """Verify the rollback code does not import or call any kanban API."""
    import agent.executive.recovery_engine as re_mod
    import ast
    raw = open(re_mod.__file__).read()
    # CODE-ONLY check: strip docstrings and comments.
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        tree = None
    if tree is not None:
        ranges = []
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str)):
                    end = n.body[0].end_lineno or n.body[0].lineno
                    ranges.append((n.body[0].lineno, end))
        lines = raw.splitlines(keepends=True)
        src = "".join(
            line for i, line in enumerate(lines, start=1)
            if not line.lstrip().startswith("#")
            and not any(lo <= i <= hi for lo, hi in ranges)
        )
    else:
        src = raw
    for tok in (
        "kanban_command", "_cmd_create", "_cmd_swarm",
        "create_swarm", "kanban_decompose", "kanban_specify",
        "kanban_swarm", "kb.create_task", "kb.delete_task",
        "kb.archive_task", "write_approval_commands",
    ):
        assert tok not in src, f"recovery_engine references forbidden {tok}"
