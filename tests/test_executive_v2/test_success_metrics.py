"""Tests for Phase 6 Success Metrics — pure aggregation.

8 tests covering:
* evaluate_worker_run (5 tests)
* aggregate_task_outcomes (1 test)
* compute_completion_percentage (1 test)
* build_evaluation_report (1 test)
"""

from __future__ import annotations

import pytest

from agent.executive.success_metrics import (
    aggregate_task_outcomes,
    build_evaluation_report,
    compute_completion_percentage,
    evaluate_worker_run,
)
from agent.executive.types import (
    TaskOutcome,
    SuccessStatus,
)


# ── evaluate_worker_run (5 tests) ────────────────────────────────────


def test_evaluate_worker_run_successful():
    """exitcode=0, error_type=None, timed_out=False, killed=False → SUCCESSFUL."""
    wr = {
        "action_executed": "RUN_WORKER",
        "exitcode": 0,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    assert evaluate_worker_run(wr) == TaskOutcome.SUCCESSFUL


def test_evaluate_worker_run_failed_nonzero_exitcode():
    wr = {
        "action_executed": "RUN_WORKER",
        "exitcode": 1,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    assert evaluate_worker_run(wr) == TaskOutcome.FAILED


def test_evaluate_worker_run_failed_error_type_set():
    wr = {
        "action_executed": "RUN_WORKER",
        "exitcode": 0,
        "error_type": "RuntimeError",
        "timed_out": False,
        "killed": False,
    }
    assert evaluate_worker_run(wr) == TaskOutcome.FAILED


def test_evaluate_worker_run_blocked_no_handler():
    """action_executed starts with NO_HANDLER_FOR_ → BLOCKED."""
    wr = {
        "action_executed": "NO_HANDLER_FOR_X",
        "exitcode": None,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    assert evaluate_worker_run(wr) == TaskOutcome.BLOCKED


def test_evaluate_worker_run_cancelled_external_flag():
    """aborted=True → CANCELLED (overrides action_executed)."""
    wr = {
        "aborted": True,
        "action_executed": "RUN_WORKER",
        "exitcode": 0,
        "error_type": None,
        "timed_out": False,
        "killed": False,
    }
    assert evaluate_worker_run(wr) == TaskOutcome.CANCELLED


# ── aggregate_task_outcomes (1 test) ───────────────────────────────


def test_aggregate_task_outcomes_mixed():
    """3 tasks with different outcomes → correct counts + outcomes list."""
    task_ids = ("t-1", "t-2", "t-3")
    worker_runs = (
        {"action_executed": "RUN_WORKER", "exitcode": 0,
         "error_type": None, "timed_out": False, "killed": False},  # SUCCESSFUL
        {"action_executed": "RUN_WORKER", "exitcode": 1,
         "error_type": None, "timed_out": False, "killed": False},  # FAILED
        {"action_executed": "NO_HANDLER_FOR_X",
         "exitcode": None, "error_type": None, "timed_out": False, "killed": False},  # BLOCKED
    )
    s, f, b, c, m, outcomes = aggregate_task_outcomes(task_ids, worker_runs)
    assert s == 1
    assert f == 1
    assert b == 1
    assert c == 0
    assert m == 0
    assert outcomes == [TaskOutcome.SUCCESSFUL, TaskOutcome.FAILED, TaskOutcome.BLOCKED]


# ── compute_completion_percentage (1 test) ──────────────────────────


def test_compute_completion_percentage_mixed():
    """2 successful + 1 failed out of 3 → (1+1+0.5)/3 = 0.833..."""
    outcomes = [TaskOutcome.SUCCESSFUL, TaskOutcome.SUCCESSFUL, TaskOutcome.FAILED]
    pct = compute_completion_percentage(outcomes)
    assert abs(pct - (2.5 / 3)) < 1e-9


# ── build_evaluation_report (1 test) ───────────────────────────────


def test_build_evaluation_report_all_successful():
    """All 3 tasks successful → SUCCESS status."""
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
    report = build_evaluation_report(
        "obj-1",
        task_ids,
        worker_runs,
        worker_dispatch_fingerprint="df-1",
        policy_fingerprint="pf-1",
        approval_fingerprint="af-1",
        plan_fingerprint="plf-1",
        goal_fingerprint="gf-1",
        objective_fingerprint="of-1",
        execution_fingerprint="ef-1",
    )
    assert report.objective_id == "obj-1"
    assert report.status == SuccessStatus.SUCCESS
    assert report.successful_tasks == 3
    assert report.failed_tasks == 0
    assert report.blocked_tasks == 0
    assert report.cancelled_tasks == 0
    assert report.completion_percentage == 1.0
    assert report.worker_success_rate == 1.0
    assert report.evidence_score == 1.0
    assert report.confidence_score == 1.0
    assert report.retry_recommended is False
    assert report.manual_intervention_required is False
    assert report.remaining_tasks == ()
    assert "Status: success" in report.summary
