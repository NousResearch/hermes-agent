"""Phase 6 Success Metrics — pure aggregation.

This module is the **only** place that converts Phase 5's persisted
``WorkerDispatchResult.worker_runs`` (and the underlying
``WorkerRunResult.to_dict()``) into a per-task ``TaskOutcome`` and
aggregates them into ``SuccessMetricBreakdown`` /
``EvaluationReport``.

It does NOT spawn workers, dispatch, or call the orchestrator. It
only reads data and produces pure-data structures.

Forbidden APIs (PROHIBITED):
* ``hermes_cli.kanban.kanban_command``
* ``hermes_cli.kanban._cmd_create`` / ``_cmd_swarm``
* ``hermes_cli.kanban_db.create_task`` / ``delete_task``
* ``hermes_cli.kanban_decompose.*`` / ``kanban_specify.*`` / ``kanban_swarm.*``
* ``hermes_cli.write_approval_commands``
* ``agent.execution_router.ExecutionRouter``
* ``agent.execution_dispatcher.ExecutionDispatcher``
* ``agent.orchestrator_interface.OrchestratorInterface.execute``
* ``agent.orchestrator.dispatcher.Dispatcher.dispatch``
* ``agent.orchestrator.batch_runner.BatchRunner.run_batch``
* ``agent.orchestrator.worker_runner.run_worker_subprocess``
* ``agent.orchestrator.handlers.make_handlers``
* ``agent.orchestrator.kanban_adapter.KanbanAdapter``
* ``delegate_task`` / ``execute()`` / ``worker_runner.real`` /
  ``pilot_bridge.real`` / ``batch_runner.real`` /
  ``run_kanban_goal_loop`` / ``evaluate_after_turn``
* Any LLM call (anthropic, openai, auxiliary_client, urllib, requests, httpx)
* Any subprocess / os.system / os.popen
* Any DB DDL (CREATE TABLE / ALTER TABLE / CREATE INDEX)
* gbrain / obsidian / notebooklm
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from .types import (
    EvaluationReport,
    SuccessMetricBreakdown,
    SuccessReport,
    SuccessStatus,
    TaskOutcome,
)


# ── Per-task outcome classification ──────────────────────────────────


def evaluate_worker_run(worker_run: Any) -> TaskOutcome:
    """Classify a single ``WorkerRunResult.to_dict()`` into a TaskOutcome.

    Pure: no I/O, no LLM, no provider.

    The criteria:
    * ``action_executed`` indicates a NO_HANDLER_FOR or BLOCK → BLOCKED
    * ``action_executed`` indicates RUN_WORKER AND the run was clean
      (exitcode 0, no error_type, not timed_out, not killed) → SUCCESSFUL
    * ``action_executed`` indicates RUN_WORKER but the run was dirty
      (non-zero exitcode, error_type set, timed_out, killed) → FAILED
    * If an external "aborted" flag was set in the worker_run
      ("aborted": true) → CANCELLED
    * Otherwise (no ``action_executed`` at all, or a malformed record)
      → MISSING

    The input is expected to be a ``WorkerRunResult.to_dict()`` (a
    plain dict). The function does NOT raise; it returns the most
    conservative outcome.
    """
    if not isinstance(worker_run, dict):
        return TaskOutcome.MISSING

    # External "aborted" flag (set by the rollback path or by an
    # external operator signal).
    if worker_run.get("aborted") is True:
        return TaskOutcome.CANCELLED

    action = worker_run.get("action_executed") or ""
    if action.startswith("NO_HANDLER_FOR_") or action == "BLOCK":
        return TaskOutcome.BLOCKED

    if action == "RUN_WORKER" or not action:
        # The run actually happened. Check the run outcome.
        exitcode = worker_run.get("exitcode")
        error_type = worker_run.get("error_type")
        timed_out = bool(worker_run.get("timed_out", False))
        killed = bool(worker_run.get("killed", False))
        if (
            exitcode == 0
            and error_type is None
            and not timed_out
            and not killed
        ):
            return TaskOutcome.SUCCESSFUL
        return TaskOutcome.FAILED

    # Unknown action: treat as missing (conservative).
    return TaskOutcome.MISSING


# ── Aggregation ──────────────────────────────────────────────────────


def aggregate_task_outcomes(
    task_ids: Tuple[str, ...],
    worker_runs: Tuple[dict, ...],
    *,
    aborted: bool = False,
) -> Tuple[
    int, int, int, int, int,  # successful, failed, blocked, cancelled, missing
    List[TaskOutcome],
]:
    """Aggregate per-task outcomes.

    Pure: no I/O.

    Returns:
    * successful_tasks, failed_tasks, blocked_tasks, cancelled_tasks,
      missing_tasks (all ints)
    * outcomes: a list of per-task TaskOutcome (parallel to task_ids)

    Edge cases:
    * If ``task_ids`` is empty, all counters are 0 and ``outcomes``
      is empty.
    * If a ``task_id`` has no corresponding ``worker_run`` (i.e. the
      parallel ``worker_runs`` list is shorter than ``task_ids``),
      the missing tasks are counted as ``MISSING``.
    * If ``aborted`` is True, all tasks are reported as
      ``CANCELLED`` (the abort flag overrides the per-task outcome).
    """
    if aborted:
        n = len(task_ids)
        outcomes = [TaskOutcome.CANCELLED] * n
        return 0, 0, 0, n, 0, outcomes

    outcomes: List[TaskOutcome] = []
    successful = failed = blocked = cancelled = missing = 0
    for i, _task_id in enumerate(task_ids):
        if i < len(worker_runs):
            outcome = evaluate_worker_run(worker_runs[i])
        else:
            outcome = TaskOutcome.MISSING
        outcomes.append(outcome)
        if outcome == TaskOutcome.SUCCESSFUL:
            successful += 1
        elif outcome == TaskOutcome.FAILED:
            failed += 1
        elif outcome == TaskOutcome.BLOCKED:
            blocked += 1
        elif outcome == TaskOutcome.CANCELLED:
            cancelled += 1
        else:
            missing += 1
    return successful, failed, blocked, cancelled, missing, outcomes


# ── Metrics computation ──────────────────────────────────────────────


def compute_completion_percentage(outcomes: List[TaskOutcome]) -> float:
    """Compute completion_percentage in 0.0 - 1.0.

    Pure.

    Per-task score:
    * SUCCESSFUL: 1.0
    * FAILED: 0.5 (partial work may be salvageable)
    * BLOCKED / CANCELLED / MISSING: 0.0

    Returns 0.0 if outcomes is empty.
    """
    if not outcomes:
        return 0.0
    total = 0.0
    for o in outcomes:
        if o == TaskOutcome.SUCCESSFUL:
            total += 1.0
        elif o == TaskOutcome.FAILED:
            total += 0.5
        # BLOCKED / CANCELLED / MISSING contribute 0.0
    return total / len(outcomes)


def build_evaluation_report(
    objective_id: str,
    task_ids: Tuple[str, ...],
    worker_runs: Tuple[dict, ...],
    *,
    worker_dispatch_fingerprint: str,
    policy_fingerprint: str,
    approval_fingerprint: str,
    plan_fingerprint: str,
    goal_fingerprint: str,
    objective_fingerprint: str,
    execution_fingerprint: str,
    aborted: bool = False,
    created_by: str = "executive_v2_phase6",
) -> EvaluationReport:
    """Build a complete EvaluationReport from raw Phase 5 data.

    Pure: no I/O, no state_meta writes. The caller is responsible
    for persistence.

    Steps:
    1. Classify each task into a TaskOutcome.
    2. Aggregate counts.
    3. Compute per-task scores + completion_percentage.
    4. Compute worker_success_rate, evidence_score, confidence_score.
    5. Determine the SuccessStatus.
    6. Determine retry_recommended + manual_intervention_required.
    7. Build the summary string.
    """
    from datetime import datetime, timezone

    successful, failed, blocked, cancelled, missing, outcomes = aggregate_task_outcomes(
        task_ids, worker_runs, aborted=aborted
    )
    total_tasks = len(task_ids)
    per_task_completion_sum = sum(
        1.0 if o == TaskOutcome.SUCCESSFUL else (0.5 if o == TaskOutcome.FAILED else 0.0)
        for o in outcomes
    )
    coverage = (
        (successful + failed + blocked + cancelled) / total_tasks
        if total_tasks > 0
        else 0.0
    )
    ran = successful + failed
    worker_success_rate = (
        (successful / ran) if ran > 0 else 0.0
    )
    mean_score = (
        sum(1.0 if o == TaskOutcome.SUCCESSFUL else 0.0 for o in outcomes)
        / total_tasks
        if total_tasks > 0
        else 0.0
    )
    evidence_score = coverage
    confidence_score = (
        (coverage * 0.5) + (worker_success_rate * 0.5)
        if total_tasks > 0
        else 0.0
    )
    completion_percentage = compute_completion_percentage(outcomes)

    # Status determination (see success_metrics_design.md §2).
    if aborted:
        status = SuccessStatus.ABORTED
    elif total_tasks == 0:
        status = SuccessStatus.BLOCKED
    elif cancelled > 0:
        status = SuccessStatus.ABORTED
    elif blocked > 0:
        status = SuccessStatus.BLOCKED
    elif completion_percentage >= 1.0:
        status = SuccessStatus.SUCCESS
    elif completion_percentage >= 0.5:
        status = SuccessStatus.PARTIAL_SUCCESS
    else:
        status = SuccessStatus.FAILED

    # retry_recommended + retry_reason + manual_intervention_required
    # (see success_metrics_design.md §3).
    timed_out_count = sum(
        1
        for wr in worker_runs
        if isinstance(wr, dict) and wr.get("timed_out")
    )
    transient_failures = failed - timed_out_count
    if status == SuccessStatus.SUCCESS:
        retry_recommended = False
        retry_reason = ""
        manual_intervention_required = False
    elif status == SuccessStatus.PARTIAL_SUCCESS:
        retry_recommended = True
        retry_reason = "partial completion; consider retry of failed tasks"
        manual_intervention_required = True
    elif status == SuccessStatus.FAILED:
        if transient_failures == 0:
            retry_recommended = False
            retry_reason = "all failures are permanent (timeouts)"
            manual_intervention_required = True
        else:
            retry_recommended = True
            retry_reason = f"{transient_failures} transient failure(s); retry recommended"
            manual_intervention_required = True
    elif status == SuccessStatus.BLOCKED:
        retry_recommended = False
        retry_reason = "blocked on missing handler; manual intervention required"
        manual_intervention_required = True
    else:  # ABORTED
        retry_recommended = False
        retry_reason = "aborted; operator decision required"
        manual_intervention_required = True

    # remaining_tasks: tasks that did NOT reach successful.
    remaining_tasks = tuple(
        task_id
        for task_id, o in zip(task_ids, outcomes)
        if o != TaskOutcome.SUCCESSFUL
    )

    # Summary
    summary = _render_summary(
        status=status,
        total_tasks=total_tasks,
        successful_tasks=successful,
        failed_tasks=failed,
        blocked_tasks=blocked,
        cancelled_tasks=cancelled,
        completion_percentage=completion_percentage,
    )

    # Metrics breakdown
    metrics = SuccessMetricBreakdown(
        successful_tasks=successful,
        failed_tasks=failed,
        blocked_tasks=blocked,
        cancelled_tasks=cancelled,
        missing_tasks=missing,
        total_tasks=total_tasks,
        per_task_completion_sum=per_task_completion_sum,
        coverage=coverage,
        worker_success_rate=worker_success_rate,
        mean_score=mean_score,
        evidence_score=evidence_score,
        confidence_score=confidence_score,
        completion_percentage=completion_percentage,
    )

    return EvaluationReport(
        objective_id=objective_id,
        execution_fingerprint=execution_fingerprint,
        worker_dispatch_fingerprint=worker_dispatch_fingerprint,
        policy_fingerprint=policy_fingerprint,
        approval_fingerprint=approval_fingerprint,
        plan_fingerprint=plan_fingerprint,
        goal_fingerprint=goal_fingerprint,
        objective_fingerprint=objective_fingerprint,
        status=status,
        completion_percentage=completion_percentage,
        successful_tasks=successful,
        failed_tasks=failed,
        blocked_tasks=blocked,
        cancelled_tasks=cancelled,
        worker_success_rate=worker_success_rate,
        evidence_score=evidence_score,
        confidence_score=confidence_score,
        retry_recommended=retry_recommended,
        retry_reason=retry_reason,
        manual_intervention_required=manual_intervention_required,
        remaining_tasks=remaining_tasks,
        summary=summary,
        metrics=metrics,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by=created_by,
    )


def _render_summary(
    *,
    status: SuccessStatus,
    total_tasks: int,
    successful_tasks: int,
    failed_tasks: int,
    blocked_tasks: int,
    cancelled_tasks: int,
    completion_percentage: float,
) -> str:
    """Render a short human-readable summary string."""
    if total_tasks == 0:
        return f"Objective has no tasks; status={status.value}."
    return (
        f"Status: {status.value}. "
        f"Total: {total_tasks}. "
        f"OK: {successful_tasks}. "
        f"Failed: {failed_tasks}. "
        f"Blocked: {blocked_tasks}. "
        f"Cancelled: {cancelled_tasks}. "
        f"Completion: {completion_percentage * 100:.1f}%."
    )


# ── SuccessReport (slim) builder ──────────────────────────────────────


def build_success_report(evaluation: EvaluationReport) -> SuccessReport:
    """Build a slim SuccessReport from a full EvaluationReport.

    Pure: no I/O.
    """
    return SuccessReport(
        objective_id=evaluation.objective_id,
        status=evaluation.status,
        completion_percentage=evaluation.completion_percentage,
        successful_tasks=evaluation.successful_tasks,
        failed_tasks=evaluation.failed_tasks,
        blocked_tasks=evaluation.blocked_tasks,
        cancelled_tasks=evaluation.cancelled_tasks,
        summary=evaluation.summary,
        created_at=evaluation.created_at,
        created_by=evaluation.created_by,
    )


__all__ = [
    "evaluate_worker_run",
    "aggregate_task_outcomes",
    "compute_completion_percentage",
    "build_evaluation_report",
    "build_success_report",
]
