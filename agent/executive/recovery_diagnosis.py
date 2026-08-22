"""Phase 7 Objective Recovery — diagnosis module.

This module is the **only** place that converts Phase 6's
persisted ``EvaluationReport`` + Phase 5's ``WorkerDispatchResult``
+ Phase 4B's ``KanbanApplyResult`` into a per-task
classification and an aggregated ``RecoveryDiagnosis``.

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
* ``SuccessEvaluatorEngine.evaluate`` (re-evaluation)
* ``WorkerDispatchEngine.apply`` (re-call)
* ``KanbanApplyEngine.apply`` (re-call)
* ``Planner.plan`` (re-call)
* ``evaluate_approval_gates`` (re-validation)
* Any LLM call (anthropic, openai, auxiliary_client, urllib, requests, httpx)
* Any subprocess / os.system / os.popen
* Any DB DDL (CREATE TABLE / ALTER TABLE / CREATE INDEX)
* gbrain / obsidian / notebooklm
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from .types import (
    EvaluationReport,
    ObjectiveState,
    RecoveryDiagnosis,
    RecoveryStatus,
    SuccessStatus,
    TaskOutcome,
    compute_recovery_diagnosis_fingerprint,
)

log = logging.getLogger("executive.recovery_diagnosis")


# ── Per-task outcome classification (Phase 7 view) ──────────────────


def classify_worker_run(worker_run: Any) -> str:
    """Classify a single ``WorkerRunResult.to_dict()`` for recovery purposes.

    Pure: no I/O, no LLM, no provider.

    Returns one of: "successful", "failed", "blocked", "cancelled", "missing".

    This is a thin wrapper over the same logic in Phase 6's
    ``success_metrics.evaluate_worker_run``; we re-derive here to
    avoid coupling Phase 7 to Phase 6 internals.

    The criteria:
    * ``action_executed`` indicates a NO_HANDLER_FOR or BLOCK → BLOCKED
    * ``aborted`` flag is True → CANCELLED
    * ``action_executed`` indicates RUN_WORKER AND clean exitcode → SUCCESSFUL
    * ``action_executed`` indicates RUN_WORKER but dirty → FAILED
    * Otherwise → MISSING
    """
    if not isinstance(worker_run, dict):
        return "missing"

    if worker_run.get("aborted") is True:
        return "cancelled"

    action = worker_run.get("action_executed") or ""
    if action.startswith("NO_HANDLER_FOR_") or action == "BLOCK":
        return "blocked"

    if action == "RUN_WORKER" or not action:
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
            return "successful"
        return "failed"

    return "missing"


# ── Aggregation ──────────────────────────────────────────────────────


def aggregate_task_outcomes(
    task_ids: Tuple[str, ...],
    worker_runs: Tuple[dict, ...],
    *,
    aborted: bool = False,
) -> Tuple[
    List[str],  # failed_task_ids
    List[str],  # blocked_task_ids
    List[str],  # cancelled_task_ids
    List[str],  # missing_task_ids
    int,        # transient_failures
    int,        # permanent_failures
    List[str],  # blocked_reasons
    bool,       # aborted_flag
]:
    """Aggregate per-task outcomes for recovery diagnosis.

    Pure: no I/O.

    Returns:
    * failed_task_ids: list of task IDs that failed (transient or permanent).
    * blocked_task_ids: list of task IDs that were blocked.
    * cancelled_task_ids: list of task IDs that were cancelled.
    * missing_task_ids: list of task IDs that have no corresponding run.
    * transient_failures: count of failed tasks with no timeout (retriable).
    * permanent_failures: count of failed tasks with timeout (NOT retriable).
    * blocked_reasons: list of action_executed strings for blocked tasks.
    * aborted_flag: True if any task was aborted.
    """
    failed_task_ids: List[str] = []
    blocked_task_ids: List[str] = []
    cancelled_task_ids: List[str] = []
    missing_task_ids: List[str] = []
    transient_failures = 0
    permanent_failures = 0
    blocked_reasons: List[str] = []
    aborted_flag = aborted

    for i, task_id in enumerate(task_ids):
        if aborted:
            cancelled_task_ids.append(task_id)
            aborted_flag = True
            continue
        if i >= len(worker_runs):
            missing_task_ids.append(task_id)
            continue
        run = worker_runs[i]
        action = (run.get("action_executed") or "") if isinstance(run, dict) else ""

        if run.get("aborted") is True if isinstance(run, dict) else False:
            cancelled_task_ids.append(task_id)
            aborted_flag = True
            continue

        if action.startswith("NO_HANDLER_FOR_") or action == "BLOCK":
            blocked_task_ids.append(task_id)
            blocked_reasons.append(action)
            continue

        if action == "RUN_WORKER" or not action:
            exitcode = run.get("exitcode") if isinstance(run, dict) else None
            error_type = run.get("error_type") if isinstance(run, dict) else None
            timed_out = bool(run.get("timed_out", False)) if isinstance(run, dict) else False
            killed = bool(run.get("killed", False)) if isinstance(run, dict) else False

            if exitcode == 0 and error_type is None and not timed_out and not killed:
                continue  # SUCCESSFUL — no classification needed
            failed_task_ids.append(task_id)
            if timed_out:
                permanent_failures += 1
            else:
                transient_failures += 1
            continue

    return (
        failed_task_ids,
        blocked_task_ids,
        cancelled_task_ids,
        missing_task_ids,
        transient_failures,
        permanent_failures,
        blocked_reasons,
        aborted_flag,
    )


# ── Recovery status classification ──────────────────────────────────


def _classify_recovery_status(
    *,
    evaluation_status: SuccessStatus,
    completion_percentage: float,
    transient_failures: int,
    permanent_failures: int,
    failed_task_count: int,
    blocked_task_count: int,
    cancelled_task_count: int,
    missing_task_count: int,
    total_tasks: int,
    blocked_reasons: List[str],
    manual_intervention_required: bool,
    aborted_flag: bool,
    risk_level: Optional[str] = None,
) -> RecoveryStatus:
    """Classify the EvaluationReport into a RecoveryStatus.

    Pure: no I/O, no LLM, no provider.

    Decision tree (priority order):
    1. SUCCESS → NO_ACTION_NEEDED.
    2. ABORTED → ABORT_RECOMMENDED (or NEEDS_HUMAN if manual intervention).
    3. BLOCKED → NEEDS_HUMAN.
    4. PARTIAL_SUCCESS with completion >= 0.8 → ACCEPT_PARTIAL.
    5. PARTIAL_SUCCESS with completion < 0.8 and failed tasks exist
       and blocked == 0 → RECOVERABLE.
    6. FAILED with transient failures > 0 and permanent == 0 → RECOVERABLE.
    7. FAILED with only permanent failures and missing > 0 → REPLAN_RECOMMENDED.
    8. FAILED with only permanent failures and missing == 0 → ABORT_RECOMMENDED.
    9. NOT_RECOVERABLE (e.g. risk_level R6) → NOT_RECOVERABLE.
    10. Default → ABORT_RECOMMENDED.
    """
    if evaluation_status == SuccessStatus.SUCCESS:
        return RecoveryStatus.NO_ACTION_NEEDED

    if evaluation_status == SuccessStatus.ABORTED:
        if manual_intervention_required and cancelled_task_count == total_tasks > 0:
            return RecoveryStatus.NEEDS_HUMAN
        return RecoveryStatus.ABORT_RECOMMENDED

    if evaluation_status == SuccessStatus.BLOCKED:
        return RecoveryStatus.NEEDS_HUMAN

    if evaluation_status == SuccessStatus.PARTIAL_SUCCESS:
        if completion_percentage >= 0.8 and not manual_intervention_required:
            return RecoveryStatus.ACCEPT_PARTIAL
        if failed_task_count > 0 and blocked_task_count == 0:
            return RecoveryStatus.RECOVERABLE
        return RecoveryStatus.NEEDS_HUMAN

    if evaluation_status == SuccessStatus.FAILED:
        if transient_failures > 0 and permanent_failures == 0:
            return RecoveryStatus.RECOVERABLE
        if permanent_failures > 0 and missing_task_count > 0:
            return RecoveryStatus.REPLAN_RECOMMENDED
        if permanent_failures > 0 and missing_task_count == 0:
            return RecoveryStatus.ABORT_RECOMMENDED
        # No failed tasks but FAILED status — treat as ABORT (defensive).
        return RecoveryStatus.ABORT_RECOMMENDED

    # Fallback.
    return RecoveryStatus.ABORT_RECOMMENDED


def _render_diagnosis_rationale(
    *,
    evaluation_status: SuccessStatus,
    recovery_status: RecoveryStatus,
    failed_task_count: int,
    blocked_task_count: int,
    cancelled_task_count: int,
    missing_task_count: int,
    transient_failures: int,
    permanent_failures: int,
    blocked_reasons: List[str],
    aborted_flag: bool,
) -> str:
    """Render a short human-readable rationale for the diagnosis."""
    if recovery_status == RecoveryStatus.NO_ACTION_NEEDED:
        return f"Status: {evaluation_status.value}. All tasks succeeded; no recovery needed."
    parts = [
        f"Status: {evaluation_status.value}.",
        f"Recovery: {recovery_status.value}.",
        f"Failed: {failed_task_count}.",
        f"Blocked: {blocked_task_count}.",
        f"Cancelled: {cancelled_task_count}.",
        f"Missing: {missing_task_count}.",
    ]
    if transient_failures > 0:
        parts.append(f"Transient: {transient_failures}.")
    if permanent_failures > 0:
        parts.append(f"Permanent: {permanent_failures}.")
    if blocked_reasons:
        parts.append(f"Block reasons: {','.join(blocked_reasons[:3])}.")
    if aborted_flag:
        parts.append("Aborted.")
    return " ".join(parts)


def build_recovery_diagnosis(
    objective_id: str,
    *,
    evaluation: EvaluationReport,
    worker_dispatch_fingerprint: str,
    policy_fingerprint: str,
    approval_fingerprint: str,
    plan_fingerprint: str,
    goal_fingerprint: str,
    objective_fingerprint: str,
    task_ids: Tuple[str, ...],
    worker_runs: Tuple[dict, ...],
    aborted: bool = False,
    risk_level: Optional[str] = None,
) -> RecoveryDiagnosis:
    """Build a complete RecoveryDiagnosis from Phase 5/6 data.

    Pure: no I/O, no state_meta writes. The caller is responsible
    for persistence.

    Steps:
    1. Classify each task into a per-task outcome.
    2. Aggregate counts.
    3. Determine the RecoveryStatus.
    4. Render rationale.
    5. Build the RecoveryDiagnosis.
    """
    (
        failed_task_ids,
        blocked_task_ids,
        cancelled_task_ids,
        missing_task_ids,
        transient_failures,
        permanent_failures,
        blocked_reasons,
        aborted_flag,
    ) = aggregate_task_outcomes(task_ids, worker_runs, aborted=aborted)

    recovery_status = _classify_recovery_status(
        evaluation_status=evaluation.status,
        completion_percentage=evaluation.completion_percentage,
        transient_failures=transient_failures,
        permanent_failures=permanent_failures,
        failed_task_count=len(failed_task_ids),
        blocked_task_count=len(blocked_task_ids),
        cancelled_task_count=len(cancelled_task_ids),
        missing_task_count=len(missing_task_ids),
        total_tasks=len(task_ids),
        blocked_reasons=blocked_reasons,
        manual_intervention_required=evaluation.manual_intervention_required,
        aborted_flag=aborted_flag,
        risk_level=risk_level,
    )

    rationale = _render_diagnosis_rationale(
        evaluation_status=evaluation.status,
        recovery_status=recovery_status,
        failed_task_count=len(failed_task_ids),
        blocked_task_count=len(blocked_task_ids),
        cancelled_task_count=len(cancelled_task_ids),
        missing_task_count=len(missing_task_ids),
        transient_failures=transient_failures,
        permanent_failures=permanent_failures,
        blocked_reasons=blocked_reasons,
        aborted_flag=aborted_flag,
    )

    return RecoveryDiagnosis(
        objective_id=objective_id,
        evaluation_fingerprint=evaluation.execution_fingerprint,
        worker_dispatch_fingerprint=worker_dispatch_fingerprint,
        policy_fingerprint=policy_fingerprint,
        approval_fingerprint=approval_fingerprint,
        plan_fingerprint=plan_fingerprint,
        goal_fingerprint=goal_fingerprint,
        objective_fingerprint=objective_fingerprint,
        evaluation_status=evaluation.status,
        recovery_status=recovery_status,
        failed_task_ids=tuple(failed_task_ids),
        blocked_task_ids=tuple(blocked_task_ids),
        cancelled_task_ids=tuple(cancelled_task_ids),
        missing_task_ids=tuple(missing_task_ids),
        transient_failures=transient_failures,
        permanent_failures=permanent_failures,
        blocked_reasons=tuple(blocked_reasons),
        aborted_flag=aborted_flag,
        manual_intervention_required=evaluation.manual_intervention_required,
        rationale=rationale,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by="executive_v2_phase7",
    )


__all__ = [
    "classify_worker_run",
    "aggregate_task_outcomes",
    "build_recovery_diagnosis",
]
