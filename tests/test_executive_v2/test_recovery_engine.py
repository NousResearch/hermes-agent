"""Tests for Phase 7 Recovery Engine — engine integration.

8 tests covering:
* Pure dry-run (2 tests)
* Evaluate happy path (3 tests)
* Evaluate with errors (2 tests)
* Idempotency (1 test)
"""

from __future__ import annotations

import pytest

from agent.executive.recovery_engine import (
    ObjectiveRecoveryEngine,
    RecoveryMappingError,
    recovery_dry_run,
    recovery_preview,
    recovery_evaluate,
    recovery_persist,
    recovery_rollback,
)
from agent.executive.types import (
    ApprovalRequest,
    EvaluationReport,
    KanbanApplyResult,
    ObjectivePlan,
    OrchestratorPlanPreview,
    PolicyDecision,
    RecoveryAction,
    RecoveryStatus,
    RiskLevel,
    SuccessStatus,
    SuccessMetricBreakdown,
    WorkerDispatchResult,
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
        approval_reason="phase7-test",
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


def _make_evaluation(objective_id="obj-1", status=SuccessStatus.PARTIAL_SUCCESS,
                     successful=2, failed=1, blocked=0, cancelled=0,
                     completion=0.667, manual=False):
    return EvaluationReport(
        objective_id=objective_id,
        execution_fingerprint="ef-1",
        worker_dispatch_fingerprint="wdf-1",
        policy_fingerprint="pf-1",
        approval_fingerprint="af-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        status=status,
        completion_percentage=completion,
        successful_tasks=successful,
        failed_tasks=failed,
        blocked_tasks=blocked,
        cancelled_tasks=cancelled,
        worker_success_rate=0.667,
        evidence_score=1.0,
        confidence_score=0.833,
        retry_recommended=(status == SuccessStatus.PARTIAL_SUCCESS),
        retry_reason="partial" if status == SuccessStatus.PARTIAL_SUCCESS else "",
        manual_intervention_required=manual,
        remaining_tasks=("t-3",) if status == SuccessStatus.PARTIAL_SUCCESS else (),
        summary="partial",
        metrics=SuccessMetricBreakdown(
            successful_tasks=successful, failed_tasks=failed,
            blocked_tasks=blocked, cancelled_tasks=cancelled, missing_tasks=0,
            total_tasks=successful + failed + blocked + cancelled,
            per_task_completion_sum=successful + 0.5 * failed, coverage=1.0,
            worker_success_rate=0.667, mean_score=0.667, evidence_score=1.0,
            confidence_score=0.833, completion_percentage=completion,
        ),
        created_at="2026-07-02T00:00:00Z",
        created_by="phase6",
    )


def _make_worker_dispatch_result(
    objective_id="obj-1",
    task_ids=("t-1", "t-2"),
    worker_runs=None,
):
    if worker_runs is None:
        worker_runs = (
            {"action_executed": "RUN_WORKER", "exitcode": 0,
             "error_type": None, "timed_out": False, "killed": False},
            {"action_executed": "RUN_WORKER", "exitcode": 0,
             "error_type": None, "timed_out": False, "killed": False},
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
               task_ids=("t-1", "t-2"), worker_runs=None,
               status=SuccessStatus.PARTIAL_SUCCESS,
               successful=2, failed=1, blocked=0, cancelled=0,
               completion=0.667, manual=False):
    in_memory_storage.set_objective_plan(_make_plan(objective_id))
    in_memory_storage.set_objective_orchestrator_preview(_make_preview(objective_id))
    in_memory_storage.set_objective_policy_decision(_make_decision(objective_id))
    in_memory_storage.set_objective_approval_request(_make_request(objective_id))
    in_memory_storage.set_objective_kanban_apply(
        _make_apply_result(objective_id, task_ids=task_ids)
    )
    in_memory_storage.set_objective_worker_dispatch(
        _make_worker_dispatch_result(objective_id, task_ids=task_ids, worker_runs=worker_runs)
    )
    in_memory_storage.set_objective_evaluation(
        _make_evaluation(objective_id, status=status, successful=successful,
                         failed=failed, blocked=blocked, cancelled=cancelled,
                         completion=completion, manual=manual)
    )


# ── Pure dry-run (2 tests) ───────────────────────────────────────


def test_dry_run_pure_no_writes(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    plan = eng.dry_run("obj-1")
    assert plan.objective_id == "obj-1"
    # No state_meta writes.
    assert in_memory_storage.get_objective_recovery_diagnosis("obj-1") is None
    assert in_memory_storage.get_objective_recovery_plan("obj-1") is None


def test_module_level_dry_run(in_memory_storage):
    _seed_full(in_memory_storage)
    plan = recovery_dry_run("obj-1", storage=in_memory_storage)
    assert plan.objective_id == "obj-1"
    assert in_memory_storage.get_objective_recovery_plan("obj-1") is None


# ── Evaluate happy path (3 tests) ─────────────────────────────────


def test_evaluate_happy_path(in_memory_storage):
    """PARTIAL_SUCCESS with 1 actual failed task → RECOVERABLE + RETRY_FAILED_TASKS."""
    # Use 3 task_ids with 1 failed run to match the evaluation's failed=1.
    failed_run = {
        "action_executed": "RUN_WORKER",
        "exitcode": 1,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    ok_run = {
        "action_executed": "RUN_WORKER",
        "exitcode": 0,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    _seed_full(
        in_memory_storage,
        task_ids=("t-1", "t-2", "t-3"),
        worker_runs=(ok_run, ok_run, failed_run),
        status=SuccessStatus.PARTIAL_SUCCESS, successful=2, failed=1,
        blocked=0, cancelled=0, completion=0.667, manual=False,
    )
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    plan = eng.evaluate("obj-1")
    assert plan.objective_id == "obj-1"
    assert plan.recommended_action == RecoveryAction.RETRY_FAILED_TASKS
    # state_meta written.
    assert in_memory_storage.get_objective_recovery_diagnosis("obj-1") is not None
    assert in_memory_storage.get_objective_recovery_plan("obj-1") is not None
    # Verify the diagnosis has the expected status.
    diagnosis = in_memory_storage.get_objective_recovery_diagnosis("obj-1")
    assert diagnosis.recovery_status == RecoveryStatus.RECOVERABLE


def test_evaluate_success_no_action(in_memory_storage):
    """SUCCESS with 2 successful tasks and 0 failures → NO_ACTION_NEEDED."""
    _seed_full(in_memory_storage,
               status=SuccessStatus.SUCCESS, successful=2, failed=0,
               blocked=0, cancelled=0, completion=1.0, manual=False)
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    plan = eng.evaluate("obj-1")
    assert plan.recommended_action == RecoveryAction.NOOP
    diagnosis = in_memory_storage.get_objective_recovery_diagnosis("obj-1")
    assert diagnosis.recovery_status == RecoveryStatus.NO_ACTION_NEEDED


def test_evaluate_blocked_needs_human(in_memory_storage):
    _seed_full(in_memory_storage,
               status=SuccessStatus.BLOCKED, successful=1, failed=0,
               blocked=1, cancelled=0, completion=0.5, manual=True)
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    plan = eng.evaluate("obj-1")
    assert plan.recommended_action in (RecoveryAction.REQUEST_WORKER,
                                       RecoveryAction.REQUEST_APPROVAL)
    diagnosis = in_memory_storage.get_objective_recovery_diagnosis("obj-1")
    assert diagnosis.recovery_status == RecoveryStatus.NEEDS_HUMAN


# ── Evaluate with errors (2 tests) ──────────────────────────────


def test_evaluate_missing_evaluation_raises(in_memory_storage):
    """No EvaluationReport → RecoveryMappingError."""
    in_memory_storage.set_objective_plan(_make_plan())
    in_memory_storage.set_objective_orchestrator_preview(_make_preview())
    in_memory_storage.set_objective_policy_decision(_make_decision())
    in_memory_storage.set_objective_approval_request(_make_request())
    in_memory_storage.set_objective_kanban_apply(_make_apply_result())
    in_memory_storage.set_objective_worker_dispatch(_make_worker_dispatch_result())
    # No set_objective_evaluation.
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    with pytest.raises(RecoveryMappingError):
        eng.evaluate("obj-1")


def test_evaluate_missing_worker_dispatch_raises(in_memory_storage):
    in_memory_storage.set_objective_plan(_make_plan())
    in_memory_storage.set_objective_orchestrator_preview(_make_preview())
    in_memory_storage.set_objective_policy_decision(_make_decision())
    in_memory_storage.set_objective_approval_request(_make_request())
    in_memory_storage.set_objective_kanban_apply(_make_apply_result())
    in_memory_storage.set_objective_evaluation(_make_evaluation())
    # No set_objective_worker_dispatch.
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    with pytest.raises(RecoveryMappingError):
        eng.evaluate("obj-1")


# ── Idempotency (1 test) ─────────────────────────────────────────


def test_evaluate_idempotency_retry(in_memory_storage):
    """Re-evaluating returns equivalent plan."""
    _seed_full(in_memory_storage)
    eng = ObjectiveRecoveryEngine(state_storage=in_memory_storage)
    p1 = eng.evaluate("obj-1")
    p2 = eng.evaluate("obj-1")
    assert p1.recommended_action == p2.recommended_action
    assert p1.summary == p2.summary
    d1 = in_memory_storage.get_objective_recovery_diagnosis("obj-1")
    d2 = in_memory_storage.get_objective_recovery_diagnosis("obj-1")
    assert d1.recovery_status == d2.recovery_status
