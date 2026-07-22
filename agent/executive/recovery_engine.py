"""Phase 7 Objective Recovery — engine facade.

Phase 7 consumes Phase 1+6 persisted state and produces a
deterministic ``RecoveryPlanPreview``. It does NOT spawn
workers, does NOT call Orchestrator / Dispatcher / BatchRunner,
does NOT call GoalManager, does NOT execute LLMs, does NOT
make provider API calls, and does NOT modify Runtime.

The engine has four modes:

* ``dry_run`` / ``preview`` — pure compute; produces a
  ``RecoveryPlanPreview`` with no state_meta writes.
* ``evaluate`` — re-reads all Phase 1+6 artifacts, builds the
  diagnosis and plan, and persists them to state_meta.
* ``persist`` — operator-supplied plan; persist to state_meta.
* ``rollback`` — best-effort, idempotent state_meta cleanup.

Forbidden APIs (PROHIBITED in this module):

* ``hermes_cli.kanban.kanban_command`` / ``_cmd_create`` / ``_cmd_swarm``
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
* Any LLM call (anthropic / openai / auxiliary_client / urllib / requests / httpx)
* Any subprocess / os.system / os.popen
* Any DB DDL (CREATE TABLE / ALTER TABLE / CREATE INDEX)
* gbrain / obsidian / notebooklm
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Tuple

from .types import (
    ApprovalRequest,
    EvaluationReport,
    ObjectivePlan,
    ObjectiveStateData,
    PolicyDecision,
    RecoveryAction,
    RecoveryDiagnosis,
    RecoveryPlanPreview,
    RecoveryStatus,
    RiskLevel,
    SuccessStatus,
    WorkerDispatchResult,
    compute_contract_fingerprint,
    compute_recovery_diagnosis_fingerprint,
    objective_recovery_diagnosis_key,
    objective_recovery_plan_key,
)
from .state_storage import (
    ObjectiveStateStorage,
    StateStorageError,
)
from .recovery_diagnosis import build_recovery_diagnosis

log = logging.getLogger("executive.recovery_engine")

SCHEMA_VERSION = "phase7.v1"


# ── Errors (custom for Phase 7) ──────────────────────────────────────


class RecoveryError(RuntimeError):
    """Base error for Phase 7."""


class RecoveryMappingError(RecoveryError):
    """Raised when Phase 1-6 state is missing or invalid."""


# ── Action matrix ──────────────────────────────────────────────────


def _classify_recovery_action(
    *,
    diagnosis: RecoveryDiagnosis,
    policy_decision: PolicyDecision,
) -> RecoveryAction:
    """Map a RecoveryDiagnosis + PolicyDecision to a RecoveryAction.

    Pure: no I/O, no LLM, no provider.

    Decision tree (priority order):
    1. NO_ACTION_NEEDED → NOOP.
    2. NOT_RECOVERABLE → ABORT_OBJECTIVE.
    3. ABORT_RECOMMENDED → ABORT_OBJECTIVE.
    4. ACCEPT_PARTIAL → ACCEPT_PARTIAL_SUCCESS.
    5. REPLAN_RECOMMENDED → REPLAN_OBJECTIVE.
    6. RECOVERABLE with blocked > 0 → RETRY_BLOCKED_TASKS (or REQUEST_WORKER).
    7. RECOVERABLE with failed > 0 → RETRY_FAILED_TASKS.
    8. NEEDS_HUMAN with BLOCK reason → REQUEST_APPROVAL.
    9. NEEDS_HUMAN with NO_HANDLER_FOR reason → REQUEST_WORKER.
    10. NEEDS_HUMAN default → ABORT_OBJECTIVE.
    11. Default → ABORT_OBJECTIVE.
    """
    if diagnosis.recovery_status == RecoveryStatus.NO_ACTION_NEEDED:
        return RecoveryAction.NOOP

    if diagnosis.recovery_status == RecoveryStatus.NOT_RECOVERABLE:
        return RecoveryAction.ABORT_OBJECTIVE

    if diagnosis.recovery_status == RecoveryStatus.ABORT_RECOMMENDED:
        return RecoveryAction.ABORT_OBJECTIVE

    if diagnosis.recovery_status == RecoveryStatus.ACCEPT_PARTIAL:
        return RecoveryAction.ACCEPT_PARTIAL_SUCCESS

    if diagnosis.recovery_status == RecoveryStatus.REPLAN_RECOMMENDED:
        return RecoveryAction.REPLAN_OBJECTIVE

    if diagnosis.recovery_status == RecoveryStatus.RECOVERABLE:
        if len(diagnosis.blocked_task_ids) > 0:
            # If blocked reason is NO_HANDLER_FOR_*, request a worker.
            if any(r.startswith("NO_HANDLER_FOR_") for r in diagnosis.blocked_reasons):
                return RecoveryAction.REQUEST_WORKER
            return RecoveryAction.RETRY_BLOCKED_TASKS
        if len(diagnosis.failed_task_ids) > 0:
            return RecoveryAction.RETRY_FAILED_TASKS
        # Defensive default.
        return RecoveryAction.NOOP

    if diagnosis.recovery_status == RecoveryStatus.NEEDS_HUMAN:
        if any(r == "BLOCK" for r in diagnosis.blocked_reasons):
            return RecoveryAction.REQUEST_APPROVAL
        if any(r.startswith("NO_HANDLER_FOR_") for r in diagnosis.blocked_reasons):
            return RecoveryAction.REQUEST_WORKER
        # If all tasks were cancelled (and there were tasks), recommend abort.
        total_problem = (
            len(diagnosis.failed_task_ids)
            + len(diagnosis.blocked_task_ids)
            + len(diagnosis.missing_task_ids)
            + len(diagnosis.cancelled_task_ids)
        )
        if diagnosis.aborted_flag and total_problem > 0 and len(diagnosis.cancelled_task_ids) == total_problem:
            return RecoveryAction.ABORT_OBJECTIVE
        # If there are failed or missing tasks (not just blocked), retry them.
        if len(diagnosis.failed_task_ids) > 0 or len(diagnosis.missing_task_ids) > 0:
            return RecoveryAction.RETRY_FAILED_TASKS
        return RecoveryAction.REQUEST_APPROVAL

    return RecoveryAction.ABORT_OBJECTIVE


def _requires_human_approval(
    recommended_action: RecoveryAction,
    policy_decision: PolicyDecision,
) -> bool:
    """Return True if the recommended action requires human approval.

    Pure: no I/O.

    Approval is required when:
    * The action is one of the worker-spawning actions.
    * The risk level is >= R5 (Worker_spawn).
    """
    if recommended_action in (
        RecoveryAction.RETRY_FAILED_TASKS,
        RecoveryAction.RETRY_BLOCKED_TASKS,
        RecoveryAction.REPLAN_OBJECTIVE,
    ):
        if policy_decision.risk_level >= RiskLevel.R5:
            return True
    return False


def _estimate_wasted_cycles(diagnosis: RecoveryDiagnosis) -> int:
    """Estimate the wasted cycles of the dispatch.

    Pure: no I/O.
    """
    return (
        diagnosis.transient_failures
        + diagnosis.permanent_failures
        + len(diagnosis.cancelled_task_ids)
        + len(diagnosis.missing_task_ids)
    )


def _render_summary(
    *,
    recovery_status: RecoveryStatus,
    recommended_action: RecoveryAction,
    failed_task_count: int,
    blocked_task_count: int,
    cancelled_task_count: int,
    missing_task_count: int,
    wasted_cycles: int,
    human_approval_required: bool,
) -> str:
    """Render a short human-readable summary."""
    return (
        f"Status: {recovery_status.value}. "
        f"Action: {recommended_action.value}. "
        f"Failed: {failed_task_count}. "
        f"Blocked: {blocked_task_count}. "
        f"Cancelled: {cancelled_task_count}. "
        f"Missing: {missing_task_count}. "
        f"Wasted cycles: {wasted_cycles}. "
        + ("Human approval required." if human_approval_required else "")
    )


def _render_next_step(
    *,
    recommended_action: RecoveryAction,
    recommended_retry_task_ids: Tuple[str, ...],
    human_approval_required: bool,
    risk_level: Optional[RiskLevel],
) -> str:
    """Render a longer next-step recommendation string."""
    base = ""
    if recommended_action == RecoveryAction.NOOP:
        base = "No action required. The objective completed successfully."
    elif recommended_action == RecoveryAction.ABORT_OBJECTIVE:
        base = "Mark the objective as aborted. Do not retry."
    elif recommended_action == RecoveryAction.REPLAN_OBJECTIVE:
        base = "Re-run Phase 3 (Planner) with a fresh plan."
    elif recommended_action == RecoveryAction.ACCEPT_PARTIAL_SUCCESS:
        base = "Accept the partial success. The objective is complete."
    elif recommended_action == RecoveryAction.RETRY_FAILED_TASKS:
        base = f"Re-run Phase 5 (Worker Dispatch) for the failed task_ids: {list(recommended_retry_task_ids)}."
    elif recommended_action == RecoveryAction.RETRY_BLOCKED_TASKS:
        base = f"Re-run Phase 5 (Worker Dispatch) for the blocked task_ids: {list(recommended_retry_task_ids)}."
    elif recommended_action == RecoveryAction.REQUEST_WORKER:
        base = f"Add a worker for the missing handler (blocked task_ids: {list(recommended_retry_task_ids)})."
    elif recommended_action == RecoveryAction.REQUEST_APPROVAL:
        base = f"Request approval from the operator to retry the blocked action (task_ids: {list(recommended_retry_task_ids)})."
    else:
        base = f"Action: {recommended_action.value}."

    if human_approval_required and risk_level is not None:
        base += f" Human approval required (risk level {risk_level.name})."
    return base


def _render_replan_rationale(diagnosis: RecoveryDiagnosis) -> str:
    """Render the rationale for REPLAN_OBJECTIVE."""
    if diagnosis.permanent_failures == 0 and len(diagnosis.missing_task_ids) == 0:
        return ""
    return (
        f"Plan has {diagnosis.permanent_failures} permanent failures and "
        f"{len(diagnosis.missing_task_ids)} missing task(s). "
        f"Recommend re-running Phase 3 (Planner) with a fresh plan."
    )


def _render_human_approval_rationale(
    recommended_action: RecoveryAction,
    risk_level: Optional[RiskLevel],
) -> str:
    """Render the rationale for human approval."""
    if risk_level is None:
        return ""
    return (
        f"Risk level {risk_level.name} requires "
        f"human approval for {recommended_action.value}."
    )


def build_recovery_plan(
    diagnosis: RecoveryDiagnosis,
    policy_decision: PolicyDecision,
) -> RecoveryPlanPreview:
    """Build a RecoveryPlanPreview from a RecoveryDiagnosis + PolicyDecision.

    Pure: no I/O, no state_meta writes. The caller is responsible
    for persistence.
    """
    recommended_action = _classify_recovery_action(
        diagnosis=diagnosis,
        policy_decision=policy_decision,
    )
    human_approval_required = _requires_human_approval(
        recommended_action=recommended_action,
        policy_decision=policy_decision,
    )
    wasted_cycles = _estimate_wasted_cycles(diagnosis)

    # Determine recommended_retry_task_ids based on the action.
    if recommended_action in (RecoveryAction.RETRY_FAILED_TASKS,):
        recommended_retry_task_ids = diagnosis.failed_task_ids
    elif recommended_action in (RecoveryAction.RETRY_BLOCKED_TASKS,):
        recommended_retry_task_ids = diagnosis.blocked_task_ids
    elif recommended_action in (RecoveryAction.REQUEST_WORKER, RecoveryAction.REQUEST_APPROVAL):
        recommended_retry_task_ids = diagnosis.blocked_task_ids
    else:
        recommended_retry_task_ids = ()

    recommended_replan_rationale = _render_replan_rationale(diagnosis)
    human_approval_rationale = _render_human_approval_rationale(
        recommended_action=recommended_action,
        risk_level=policy_decision.risk_level,
    )
    summary = _render_summary(
        recovery_status=diagnosis.recovery_status,
        recommended_action=recommended_action,
        failed_task_count=len(diagnosis.failed_task_ids),
        blocked_task_count=len(diagnosis.blocked_task_ids),
        cancelled_task_count=len(diagnosis.cancelled_task_ids),
        missing_task_count=len(diagnosis.missing_task_ids),
        wasted_cycles=wasted_cycles,
        human_approval_required=human_approval_required,
    )
    next_step_recommendation = _render_next_step(
        recommended_action=recommended_action,
        recommended_retry_task_ids=recommended_retry_task_ids,
        human_approval_required=human_approval_required,
        risk_level=policy_decision.risk_level,
    )

    diagnosis_fingerprint = compute_recovery_diagnosis_fingerprint(
        objective_id=diagnosis.objective_id,
        evaluation_fingerprint=diagnosis.evaluation_fingerprint,
        worker_dispatch_fingerprint=diagnosis.worker_dispatch_fingerprint,
        policy_fingerprint=diagnosis.policy_fingerprint,
        approval_fingerprint=diagnosis.approval_fingerprint,
        plan_fingerprint=diagnosis.plan_fingerprint,
        goal_fingerprint=diagnosis.goal_fingerprint,
        objective_fingerprint=diagnosis.objective_fingerprint,
        recovery_status=diagnosis.recovery_status.value if hasattr(diagnosis.recovery_status, "value") else str(diagnosis.recovery_status),
        failed_task_ids=diagnosis.failed_task_ids,
        blocked_task_ids=diagnosis.blocked_task_ids,
        cancelled_task_ids=diagnosis.cancelled_task_ids,
        missing_task_ids=diagnosis.missing_task_ids,
        transient_failures=diagnosis.transient_failures,
        permanent_failures=diagnosis.permanent_failures,
    )

    return RecoveryPlanPreview(
        objective_id=diagnosis.objective_id,
        diagnosis_fingerprint=diagnosis_fingerprint,
        recommended_action=recommended_action,
        recommended_retry_task_ids=recommended_retry_task_ids,
        recommended_replan_rationale=recommended_replan_rationale,
        human_approval_required=human_approval_required,
        human_approval_rationale=human_approval_rationale,
        rollback_safe=True,
        estimated_wasted_cycles=wasted_cycles,
        summary=summary,
        next_step_recommendation=next_step_recommendation,
        created_at=diagnosis.created_at,
        created_by=diagnosis.created_by,
    )


# ── Engine ────────────────────────────────────────────────────────


class ObjectiveRecoveryEngine:
    """High-level facade for Phase 7 Objective Recovery.

    Constructor takes an optional ``state_storage``. All side effects
    are gated on the read of Phase 6's ``EvaluationReport``.

    The engine is read-only with respect to Phase 1-6 state. It only
    writes 2 new state_meta keys (objective_recovery_diagnosis and
    objective_recovery_plan).
    """

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(
        self,
        *,
        state_storage: Optional[ObjectiveStateStorage] = None,
    ) -> None:
        self._storage = state_storage or ObjectiveStateStorage()

    # ── pure compute ──────────────────────────────────────────────

    def dry_run(
        self,
        objective_id: str,
        *,
        aborted: bool = False,
    ) -> RecoveryPlanPreview:
        """Pure compute: build RecoveryPlanPreview. NO state_meta writes.

        Reads Phase 1-6 state via ``self._storage``; does NOT mutate it.
        """
        return self._build_plan(objective_id, aborted=aborted)

    def preview(
        self,
        objective_id: str,
        *,
        aborted: bool = False,
    ) -> RecoveryPlanPreview:
        """Same as dry_run. Read-only. NO state_meta writes."""
        return self._build_plan(objective_id, aborted=aborted)

    def evaluate(
        self,
        objective_id: str,
        *,
        aborted: bool = False,
    ) -> RecoveryPlanPreview:
        """Re-read Phase 1-6 artifacts + build + persist RecoveryPlanPreview.

        Side effects:
        * state_meta[objective_recovery_diagnosis:<oid>]
        * state_meta[objective_recovery_plan:<oid>]

        Raises:
        * ``RecoveryMappingError`` if Phase 6's EvaluationReport is
          missing (the objective was never evaluated).
        """
        plan, diagnosis = self._build_plan_and_diagnosis(objective_id, aborted=aborted)
        # Persist the actual diagnosis (not a synthesized one).
        try:
            self._storage.set_objective_recovery_diagnosis(diagnosis)
        except StateStorageError as e:
            log.error("set_objective_recovery_diagnosis failed: %s", e)
            raise
        # Persist the plan.
        try:
            self._storage.set_objective_recovery_plan(
                plan.objective_id, plan
            )
        except StateStorageError as e:
            log.error("set_objective_recovery_plan failed: %s", e)
            raise
        return plan

    def persist(self, plan: RecoveryPlanPreview) -> None:
        """Operator-supplied plan. Persist to state_meta.

        Persists BOTH the plan and a synthesized diagnosis.
        """
        # Build a minimal diagnosis from the plan for the diagnosis key.
        # (The plan is the public artifact; the diagnosis is internal.)
        # Note: in the canonical entry path (evaluate), the diagnosis
        # is built before the plan. Here we synthesize one for the
        # operator-supplied case.
        synthetic_diagnosis = RecoveryDiagnosis(
            objective_id=plan.objective_id,
            evaluation_fingerprint="",
            worker_dispatch_fingerprint="",
            policy_fingerprint="",
            approval_fingerprint="",
            plan_fingerprint="",
            goal_fingerprint="",
            objective_fingerprint="",
            evaluation_status=SuccessStatus.FAILED,
            recovery_status=RecoveryStatus.NEEDS_HUMAN,
            failed_task_ids=(),
            blocked_task_ids=(),
            cancelled_task_ids=(),
            missing_task_ids=(),
            transient_failures=0,
            permanent_failures=0,
            blocked_reasons=(),
            aborted_flag=False,
            manual_intervention_required=plan.human_approval_required,
            rationale="Operator-supplied plan; see plan.summary.",
            created_at=plan.created_at,
            created_by=plan.created_by,
        )
        try:
            self._storage.set_objective_recovery_diagnosis(synthetic_diagnosis)
        except StateStorageError as e:
            log.error("set_objective_recovery_diagnosis failed: %s", e)
            raise
        try:
            self._storage.set_objective_recovery_plan(
                plan.objective_id, plan
            )
        except StateStorageError as e:
            log.error("set_objective_recovery_plan failed: %s", e)
            raise

    def rollback(
        self,
        objective_id: str,
    ) -> bool:
        """Best-effort, idempotent state_meta cleanup.

        Returns True if at least one of the 2 state_meta keys was
        deleted. Does NOT touch Phase 1-6 state. Does NOT touch
        kanban DB. Does NOT kill running workers.
        """
        deleted1 = self._storage.delete_objective_recovery_diagnosis(objective_id)
        deleted2 = self._storage.delete_objective_recovery_plan(objective_id)
        return deleted1 or deleted2

    # ── helpers ────────────────────────────────────────────────────

    def _build_plan(
        self,
        objective_id: str,
        *,
        aborted: bool = False,
    ) -> RecoveryPlanPreview:
        """Internal: build the RecoveryPlanPreview by reading Phase 1-6 state."""
        plan, _ = self._build_plan_and_diagnosis(objective_id, aborted=aborted)
        return plan

    def _build_plan_and_diagnosis(
        self,
        objective_id: str,
        *,
        aborted: bool = False,
    ) -> Tuple[RecoveryPlanPreview, RecoveryDiagnosis]:
        # 1. Load Phase 6 (EvaluationReport).
        evaluation = self._storage.get_objective_evaluation(objective_id)
        if evaluation is None:
            raise RecoveryMappingError(
                f"Phase 6 EvaluationReport missing for objective_id={objective_id!r}. "
                f"The objective was never evaluated (or was rolled back)."
            )

        # 2. Load Phase 5 (WorkerDispatchResult).
        worker_dispatch = self._storage.get_objective_worker_dispatch(objective_id)
        if worker_dispatch is None:
            raise RecoveryMappingError(
                f"Phase 5 WorkerDispatchResult missing for objective_id={objective_id!r}. "
                f"Cannot recommend recovery for an objective that was never dispatched."
            )

        # 3. Load Phase 4B (KanbanApplyResult) for the canonical task_ids.
        apply_record = self._storage.get_objective_kanban_apply(objective_id)
        if apply_record is None:
            raise RecoveryMappingError(
                f"Phase 4B KanbanApplyResult missing for objective_id={objective_id!r}. "
                f"Cannot recommend recovery for an objective that was never applied."
            )

        # 4. Load Phase 1+2+3+4A for the fingerprints.
        decision = self._storage.get_objective_policy_decision(objective_id)
        request = self._storage.get_objective_approval_request(objective_id)
        plan = self._storage.get_objective_plan(objective_id)
        goal_link = self._storage.get_objective_goal_link(objective_id)
        objective_state = self._storage.load(objective_id)

        # Use empty defaults if any of the optional artifacts are missing.
        decision_fp = str(getattr(decision, "decision_fingerprint", "") or "") if decision else ""
        request_fp = str(getattr(request, "request_fingerprint", "") or "") if request else ""
        plan_fp = str(getattr(plan, "plan_fingerprint", "") or "") if plan else ""
        goal_fp = str(getattr(goal_link, "goal_fingerprint", "") or "") if goal_link else ""
        objective_fp = (
            str(getattr(objective_state, "execution_fingerprint", "") or "")
            if objective_state is not None
            else ""
        )
        if not objective_fp:
            objective_fp = compute_contract_fingerprint(objective_id, "phase7")

        # 5. Build the diagnosis.
        task_ids = tuple(apply_record.task_ids or ())
        worker_runs = tuple(worker_dispatch.worker_runs or ())
        risk_level_value = getattr(decision, "risk_level", None) if decision else None

        diagnosis = build_recovery_diagnosis(
            objective_id,
            evaluation=evaluation,
            worker_dispatch_fingerprint=str(
                getattr(worker_dispatch, "dispatch_fingerprint", "") or ""
            ),
            policy_fingerprint=decision_fp,
            approval_fingerprint=request_fp,
            plan_fingerprint=plan_fp,
            goal_fingerprint=goal_fp,
            objective_fingerprint=objective_fp,
            task_ids=task_ids,
            worker_runs=worker_runs,
            aborted=aborted,
            risk_level=str(risk_level_value) if risk_level_value else None,
        )

        # 6. Build the plan.
        # For the policy decision, use a default if missing.
        if decision is None:
            from .types import PolicyDecision
            decision = PolicyDecision(
                objective_id=objective_id,
                risk_level=RiskLevel.R3,
                allowed_actions=(),
                forbidden_actions=(),
                approval_required=False,
                warnings=(),
                approval_requirements=(),
                risk_score=0.5,
                risk_components={},
                created_at="1970-01-01T00:00:00Z",
                decision_fingerprint="",
            )
        plan = build_recovery_plan(diagnosis=diagnosis, policy_decision=decision)
        return plan, diagnosis


# ── Module-level wrappers ──────────────────────────────────────────


def recovery_dry_run(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    aborted: bool = False,
) -> RecoveryPlanPreview:
    """Module-level wrapper: pure compute dry-run. NO state_meta writes."""
    eng = ObjectiveRecoveryEngine(state_storage=storage)
    return eng.dry_run(objective_id, aborted=aborted)


def recovery_preview(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    aborted: bool = False,
) -> RecoveryPlanPreview:
    """Module-level wrapper: same as dry_run. NO state_meta writes."""
    eng = ObjectiveRecoveryEngine(state_storage=storage)
    return eng.preview(objective_id, aborted=aborted)


def recovery_evaluate(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    aborted: bool = False,
) -> RecoveryPlanPreview:
    """Module-level wrapper: re-read Phase 1-6 + persist RecoveryPlanPreview."""
    eng = ObjectiveRecoveryEngine(state_storage=storage)
    return eng.evaluate(objective_id, aborted=aborted)


def recovery_persist(
    plan: RecoveryPlanPreview,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> None:
    """Module-level wrapper: operator-supplied plan. Persist to state_meta."""
    eng = ObjectiveRecoveryEngine(state_storage=storage)
    eng.persist(plan)


def recovery_rollback(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> bool:
    """Module-level wrapper: best-effort, idempotent rollback."""
    eng = ObjectiveRecoveryEngine(state_storage=storage)
    return eng.rollback(objective_id)


__all__ = [
    "ObjectiveRecoveryEngine",
    "RecoveryError",
    "RecoveryMappingError",
    "recovery_dry_run",
    "recovery_preview",
    "recovery_evaluate",
    "recovery_persist",
    "recovery_rollback",
    "SCHEMA_VERSION",
]
