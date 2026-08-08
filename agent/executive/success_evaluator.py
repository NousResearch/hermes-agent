"""Phase 6 Success Evaluator — engine facade.

Phase 6 consumes Phase 1+5 persisted state (WorkerDispatchResult,
KanbanApplyResult, PolicyDecision, ApprovalRequest, ObjectivePlan,
GoalLinkage, ExecutionContract, ObjectiveStateData) and produces a
deterministic EvaluationReport.

It does NOT spawn workers, does NOT call Orchestrator / Dispatcher /
BatchRunner, does NOT call GoalManager, does NOT execute LLMs, does
NOT make provider API calls, does NOT modify Runtime.

The engine has four modes:

* ``dry_run`` — pure compute; produces an EvaluationReport with
  no state_meta writes.
* ``evaluate`` — re-reads all Phase 1+5 artifacts, builds the
  EvaluationReport, and persists it to state_meta.
* ``persist`` — operator-supplied report; persist to state_meta.
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
* Any LLM call (anthropic / openai / auxiliary_client / urllib / requests / httpx)
* Any subprocess / os.system / os.popen
* Any DB DDL (CREATE TABLE / ALTER TABLE / CREATE INDEX)
* gbrain / obsidian / notebooklm
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from .types import (
    ApprovalRequest,
    EvaluationReport,
    KanbanApplyResult,
    PolicyDecision,
    SuccessReport,
    WorkerDispatchResult,
    compute_contract_fingerprint,
    compute_evaluation_fingerprint,
    objective_evaluation_key,
    objective_success_report_key,
)
from .state_storage import (
    ObjectiveStateStorage,
    StateStorageError,
)
from .success_metrics import (
    build_evaluation_report,
    build_success_report,
)

log = logging.getLogger("executive.success_evaluator")

SCHEMA_VERSION = "phase6.v1"


# ── Errors (custom for Phase 6) ──────────────────────────────────────


class SuccessEvaluatorError(RuntimeError):
    """Base error for Phase 6."""


class SuccessEvaluatorMappingError(SuccessEvaluatorError):
    """Raised when Phase 1+5 state is missing or invalid."""


# ── Engine ────────────────────────────────────────────────────────


class SuccessEvaluatorEngine:
    """High-level facade for Phase 6 Success Evaluator.

    Constructor takes an optional ``state_storage``. All side effects
    are gated on the read of Phase 5's WorkerDispatchResult.

    The engine is read-only with respect to Phase 1+5 state. It only
    writes 2 new state_meta keys (objective_evaluation and
    objective_success_report).
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
    ) -> EvaluationReport:
        """Pure compute: build EvaluationReport. NO state_meta writes.

        Reads Phase 1+5 state via ``self._storage``; does NOT mutate it.
        """
        return self._build_report(objective_id, aborted=aborted)

    def evaluate(
        self,
        objective_id: str,
        *,
        aborted: bool = False,
    ) -> EvaluationReport:
        """Re-read Phase 1+5 artifacts + build + persist EvaluationReport.

        Side effects:
        * state_meta[objective_evaluation:<oid>]
        * state_meta[objective_success_report:<oid>]

        Raises:
        * ``SuccessEvaluatorMappingError`` if Phase 5's
          WorkerDispatchResult is missing (the objective was never
          dispatched).
        """
        report = self._build_report(objective_id, aborted=aborted)
        self.persist(report)
        return report

    def persist(self, report: EvaluationReport) -> None:
        """Operator-supplied report. Persist to state_meta."""
        try:
            self._storage.set_objective_evaluation(report)
        except StateStorageError as e:
            log.error("set_objective_evaluation failed: %s", e)
            raise
        # Slim report.
        slim = build_success_report(report)
        try:
            self._storage.set_objective_success_report(
                report.objective_id, slim
            )
        except StateStorageError as e:
            log.error("set_objective_success_report failed: %s", e)
            raise

    def rollback(
        self,
        objective_id: str,
    ) -> bool:
        """Best-effort, idempotent state_meta cleanup.

        Returns True if at least one of the 2 state_meta keys was
        deleted. Does NOT touch Phase 1+5 state. Does NOT touch
        kanban DB. Does NOT kill running workers.
        """
        deleted1 = self._storage.delete_objective_evaluation(objective_id)
        deleted2 = self._storage.delete_objective_success_report(objective_id)
        return deleted1 or deleted2

    # ── helpers ────────────────────────────────────────────────────

    def _build_report(
        self,
        objective_id: str,
        *,
        aborted: bool = False,
    ) -> EvaluationReport:
        """Internal: build the EvaluationReport by reading Phase 1+5 state."""
        # 1. Load Phase 5 (WorkerDispatchResult).
        worker_dispatch = self._storage.get_objective_worker_dispatch(objective_id)
        if worker_dispatch is None:
            raise SuccessEvaluatorMappingError(
                f"Phase 5 WorkerDispatchResult missing for objective_id={objective_id!r}. "
                f"The objective was never dispatched (or was rolled back)."
            )

        # 2. Load Phase 4B (KanbanApplyResult) for the canonical task_ids.
        apply_record = self._storage.get_objective_kanban_apply(objective_id)
        if apply_record is None:
            raise SuccessEvaluatorMappingError(
                f"Phase 4B KanbanApplyResult missing for objective_id={objective_id!r}. "
                f"Cannot evaluate an objective that was never applied."
            )

        # 3. Load Phase 1+2+3+4A for the fingerprints (best-effort; missing
        #    artifacts yield empty-string fingerprints, which is fine).
        decision = self._storage.get_objective_policy_decision(objective_id)
        request = self._storage.get_objective_approval_request(objective_id)
        plan = self._storage.get_objective_plan(objective_id)
        goal_link = self._storage.get_objective_goal_link(objective_id)
        objective_state = self._storage.load(objective_id)

        decision_fp = str(getattr(decision, "decision_fingerprint", "") or "")
        request_fp = str(getattr(request, "request_fingerprint", "") or "")
        plan_fp = str(getattr(plan, "plan_fingerprint", "") or "")
        goal_fp = str(getattr(goal_link, "goal_fingerprint", "") or "")
        objective_fp = (
            str(getattr(objective_state, "execution_fingerprint", "") or "")
            if objective_state is not None
            else ""
        )
        if not objective_fp:
            # Fallback: derive from objective_id.
            objective_fp = compute_contract_fingerprint(
                objective_id, "phase6"
            )

        # 4. Build the report.
        task_ids = tuple(apply_record.task_ids or ())
        worker_runs = tuple(worker_dispatch.worker_runs or ())
        report = build_evaluation_report(
            objective_id,
            task_ids,
            worker_runs,
            worker_dispatch_fingerprint=str(
                getattr(worker_dispatch, "dispatch_fingerprint", "") or ""
            ),
            policy_fingerprint=decision_fp,
            approval_fingerprint=request_fp,
            plan_fingerprint=plan_fp,
            goal_fingerprint=goal_fp,
            objective_fingerprint=objective_fp,
            execution_fingerprint=objective_fp,
            aborted=aborted,
        )
        return report


# ── Module-level wrappers ──────────────────────────────────────────


def success_evaluator_dry_run(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    aborted: bool = False,
) -> EvaluationReport:
    """Module-level wrapper: pure compute dry-run. NO state_meta writes."""
    eng = SuccessEvaluatorEngine(state_storage=storage)
    return eng.dry_run(objective_id, aborted=aborted)


def success_evaluator_evaluate(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    aborted: bool = False,
) -> EvaluationReport:
    """Module-level wrapper: re-read Phase 1+5 + persist EvaluationReport."""
    eng = SuccessEvaluatorEngine(state_storage=storage)
    return eng.evaluate(objective_id, aborted=aborted)


def success_evaluator_persist(
    report: EvaluationReport,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> None:
    """Module-level wrapper: operator-supplied report. Persist to state_meta."""
    eng = SuccessEvaluatorEngine(state_storage=storage)
    eng.persist(report)


def success_evaluator_rollback(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> bool:
    """Module-level wrapper: best-effort, idempotent rollback."""
    eng = SuccessEvaluatorEngine(state_storage=storage)
    return eng.rollback(objective_id)


__all__ = [
    "SuccessEvaluatorEngine",
    "SuccessEvaluatorError",
    "SuccessEvaluatorMappingError",
    "success_evaluator_dry_run",
    "success_evaluator_evaluate",
    "success_evaluator_persist",
    "success_evaluator_rollback",
    "SCHEMA_VERSION",
]
