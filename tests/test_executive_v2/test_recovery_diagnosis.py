"""Tests for Phase 7 Recovery Diagnosis — pure aggregation.

8 tests covering:
* classify_worker_run (3 tests)
* aggregate_task_outcomes (1 test)
* _classify_recovery_status (3 tests)
* build_recovery_diagnosis (1 test)
"""

from __future__ import annotations

import pytest

from agent.executive.recovery_diagnosis import (
    aggregate_task_outcomes,
    build_recovery_diagnosis,
    classify_worker_run,
)
from agent.executive.types import (
    RecoveryStatus,
    SuccessStatus,
)


# ── classify_worker_run (3 tests) ────────────────────────────────────


def test_classify_worker_run_successful():
    """exitcode=0, no error → successful."""
    wr = {
        "action_executed": "RUN_WORKER",
        "exitcode": 0,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    assert classify_worker_run(wr) == "successful"


def test_classify_worker_run_blocked():
    """NO_HANDLER_FOR_X → blocked."""
    wr = {
        "action_executed": "NO_HANDLER_FOR_X",
        "exitcode": None,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    assert classify_worker_run(wr) == "blocked"


def test_classify_worker_run_cancelled():
    """aborted=True → cancelled."""
    wr = {
        "aborted": True,
        "action_executed": "RUN_WORKER",
        "exitcode": 0,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    assert classify_worker_run(wr) == "cancelled"


# ── aggregate_task_outcomes (1 test) ───────────────────────────────


def test_aggregate_task_outcomes_mixed():
    """Mixed outcomes: 1 successful, 1 failed (transient), 1 blocked, 1 missing."""
    task_ids = ("t-1", "t-2", "t-3", "t-4")
    worker_runs = (
        {"action_executed": "RUN_WORKER", "exitcode": 0, "error_type": None, "timed_out": False, "killed": False},
        {"action_executed": "RUN_WORKER", "exitcode": 1, "error_type": None, "timed_out": False, "killed": False},
        {"action_executed": "NO_HANDLER_FOR_X", "exitcode": None, "error_type": None, "timed_out": False, "killed": False},
        # No run for t-4
    )
    (failed, blocked, cancelled, missing, transient, permanent, reasons, aborted) = aggregate_task_outcomes(task_ids, worker_runs)
    assert failed == ["t-2"]
    assert blocked == ["t-3"]
    assert cancelled == []
    assert missing == ["t-4"]
    assert transient == 1
    assert permanent == 0
    assert reasons == ["NO_HANDLER_FOR_X"]
    assert aborted is False


# ── build_recovery_diagnosis (1 test) ─────────────────────────────


def test_build_recovery_diagnosis_all_successful():
    """All 3 tasks successful → NO_ACTION_NEEDED."""
    from agent.executive.types import EvaluationReport, SuccessMetricBreakdown

    task_ids = ("t-1", "t-2", "t-3")
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
    evaluation = EvaluationReport(
        objective_id="obj-1",
        execution_fingerprint="ef-1",
        worker_dispatch_fingerprint="wdf-1",
        policy_fingerprint="pf-1",
        approval_fingerprint="af-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        status=SuccessStatus.SUCCESS,
        completion_percentage=1.0,
        successful_tasks=3,
        failed_tasks=0,
        blocked_tasks=0,
        cancelled_tasks=0,
        worker_success_rate=1.0,
        evidence_score=1.0,
        confidence_score=1.0,
        retry_recommended=False,
        retry_reason="",
        manual_intervention_required=False,
        remaining_tasks=(),
        summary="All OK",
        metrics=SuccessMetricBreakdown(
            successful_tasks=3, failed_tasks=0, blocked_tasks=0,
            cancelled_tasks=0, missing_tasks=0, total_tasks=3,
            per_task_completion_sum=3.0, coverage=1.0, worker_success_rate=1.0,
            mean_score=1.0, evidence_score=1.0, confidence_score=1.0,
            completion_percentage=1.0,
        ),
        created_at="2026-07-02T00:00:00Z",
        created_by="phase6",
    )
    diagnosis = build_recovery_diagnosis(
        "obj-1",
        evaluation=evaluation,
        worker_dispatch_fingerprint="wdf-1",
        policy_fingerprint="pf-1",
        approval_fingerprint="af-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        task_ids=task_ids,
        worker_runs=worker_runs,
    )
    assert diagnosis.recovery_status == RecoveryStatus.NO_ACTION_NEEDED


# ── Additional status classification (3 tests) ───────────────────


def test_diagnose_partial_success_recoverable():
    """PARTIAL_SUCCESS with 1 transient failure and 0 blocked → RECOVERABLE."""
    from agent.executive.types import EvaluationReport, SuccessMetricBreakdown

    task_ids = ("t-1", "t-2", "t-3")
    worker_runs = (
        {"action_executed": "RUN_WORKER", "exitcode": 0, "error_type": None, "timed_out": False, "killed": False},
        {"action_executed": "RUN_WORKER", "exitcode": 0, "error_type": None, "timed_out": False, "killed": False},
        {"action_executed": "RUN_WORKER", "exitcode": 1, "error_type": None, "timed_out": False, "killed": False},
    )
    evaluation = EvaluationReport(
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
        successful_tasks=2,
        failed_tasks=1,
        blocked_tasks=0,
        cancelled_tasks=0,
        worker_success_rate=0.667,
        evidence_score=1.0,
        confidence_score=0.833,
        retry_recommended=True,
        retry_reason="partial",
        manual_intervention_required=True,
        remaining_tasks=("t-3",),
        summary="partial",
        metrics=SuccessMetricBreakdown(
            successful_tasks=2, failed_tasks=1, blocked_tasks=0,
            cancelled_tasks=0, missing_tasks=0, total_tasks=3,
            per_task_completion_sum=2.5, coverage=1.0, worker_success_rate=0.667,
            mean_score=0.667, evidence_score=1.0, confidence_score=0.833,
            completion_percentage=0.667,
        ),
        created_at="2026-07-02T00:00:00Z",
        created_by="phase6",
    )
    diagnosis = build_recovery_diagnosis(
        "obj-1", evaluation=evaluation,
        worker_dispatch_fingerprint="wdf-1", policy_fingerprint="pf-1",
        approval_fingerprint="af-1", plan_fingerprint="plf-1",
        goal_fingerprint="gf-1", objective_fingerprint="of-1",
        task_ids=task_ids, worker_runs=worker_runs,
    )
    assert diagnosis.recovery_status == RecoveryStatus.RECOVERABLE
    assert diagnosis.failed_task_ids == ("t-3",)
    assert diagnosis.transient_failures == 1


def test_diagnose_blocked_needs_human():
    """BLOCKED with NO_HANDLER_FOR_X → NEEDS_HUMAN."""
    from agent.executive.types import EvaluationReport, SuccessMetricBreakdown

    task_ids = ("t-1", "t-2")
    worker_runs = (
        {"action_executed": "RUN_WORKER", "exitcode": 0, "error_type": None, "timed_out": False, "killed": False},
        {"action_executed": "NO_HANDLER_FOR_X", "exitcode": None, "error_type": None, "timed_out": False, "killed": False},
    )
    evaluation = EvaluationReport(
        objective_id="obj-1",
        execution_fingerprint="ef-1",
        worker_dispatch_fingerprint="wdf-1",
        policy_fingerprint="pf-1",
        approval_fingerprint="af-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        status=SuccessStatus.BLOCKED,
        completion_percentage=0.5,
        successful_tasks=1,
        failed_tasks=0,
        blocked_tasks=1,
        cancelled_tasks=0,
        worker_success_rate=1.0,
        evidence_score=1.0,
        confidence_score=1.0,
        retry_recommended=False,
        retry_reason="",
        manual_intervention_required=True,
        remaining_tasks=("t-2",),
        summary="blocked",
        metrics=SuccessMetricBreakdown(
            successful_tasks=1, failed_tasks=0, blocked_tasks=1,
            cancelled_tasks=0, missing_tasks=0, total_tasks=2,
            per_task_completion_sum=1.0, coverage=1.0, worker_success_rate=1.0,
            mean_score=0.5, evidence_score=1.0, confidence_score=1.0,
            completion_percentage=0.5,
        ),
        created_at="2026-07-02T00:00:00Z",
        created_by="phase6",
    )
    diagnosis = build_recovery_diagnosis(
        "obj-1", evaluation=evaluation,
        worker_dispatch_fingerprint="wdf-1", policy_fingerprint="pf-1",
        approval_fingerprint="af-1", plan_fingerprint="plf-1",
        goal_fingerprint="gf-1", objective_fingerprint="of-1",
        task_ids=task_ids, worker_runs=worker_runs,
    )
    assert diagnosis.recovery_status == RecoveryStatus.NEEDS_HUMAN
    assert diagnosis.blocked_task_ids == ("t-2",)
    assert diagnosis.blocked_reasons == ("NO_HANDLER_FOR_X",)


def test_diagnose_aborted_abort_recommended():
    """ABORTED (no manual intervention) → ABORT_RECOMMENDED."""
    from agent.executive.types import EvaluationReport, SuccessMetricBreakdown

    task_ids = ("t-1",)
    worker_runs = ()
    evaluation = EvaluationReport(
        objective_id="obj-1",
        execution_fingerprint="ef-1",
        worker_dispatch_fingerprint="wdf-1",
        policy_fingerprint="pf-1",
        approval_fingerprint="af-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        status=SuccessStatus.ABORTED,
        completion_percentage=0.0,
        successful_tasks=0,
        failed_tasks=0,
        blocked_tasks=0,
        cancelled_tasks=1,
        worker_success_rate=0.0,
        evidence_score=0.0,
        confidence_score=0.0,
        retry_recommended=False,
        retry_reason="aborted",
        manual_intervention_required=False,
        remaining_tasks=(),
        summary="aborted",
        metrics=SuccessMetricBreakdown(
            successful_tasks=0, failed_tasks=0, blocked_tasks=0,
            cancelled_tasks=1, missing_tasks=0, total_tasks=1,
            per_task_completion_sum=0.0, coverage=0.0, worker_success_rate=0.0,
            mean_score=0.0, evidence_score=0.0, confidence_score=0.0,
            completion_percentage=0.0,
        ),
        created_at="2026-07-02T00:00:00Z",
        created_by="phase6",
    )
    diagnosis = build_recovery_diagnosis(
        "obj-1", evaluation=evaluation,
        worker_dispatch_fingerprint="wdf-1", policy_fingerprint="pf-1",
        approval_fingerprint="af-1", plan_fingerprint="plf-1",
        goal_fingerprint="gf-1", objective_fingerprint="of-1",
        task_ids=task_ids, worker_runs=worker_runs,
        aborted=True,
    )
    assert diagnosis.recovery_status == RecoveryStatus.ABORT_RECOMMENDED
    assert diagnosis.aborted_flag is True
