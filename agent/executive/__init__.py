"""Executive v2 Objective Engine — Phase 5 Worker Dispatch.

Phase 1 (Foundation): standalone Objective Engine that normalizes,
classifies, runs P0/P1 capability discovery, generates an
ExecutionContract.v1, and persists Objective state.

Phase 2 (GoalManager Bridge): maps objectives to goals with
session linkage, conflict detection, and goal->objective direction.

Phase 3 (Planner/Orchestrator Bridge): produces a deterministic
OrchestratorPlanPreview from an ObjectivePlan.

Phase 4A (Policy/Approval Gates): RiskClassification (R0-R6), 8-layer
ApprovalGateEvaluator, dry-run/persist/rollback for PolicyDecision
and ApprovalRequest.

Phase 4B (Kanban Apply): builds KanbanApplyPreview, applies via
the existing ``kb.create_task`` API, persists KanbanApplyResult to
state_meta, and rolls back via ``kb.archive_task`` (or
``kb.delete_task`` if ``hard_delete=True``).
Phase 5 (Worker Dispatch): consumes a Phase 4B KanbanApplyResult
and a Phase 4A ApprovalRequest, re-validates the 8 approval
gates (incl. Layer 6 Worker_spawn R5), and dispatches the kanban
tasks to real workers via the existing
`agent/orchestrator/{Dispatcher, BatchRunner, run_worker_subprocess,
make_handlers, KanbanAdapter}`
infrastructure. It does NOT spawn workers directly and does NOT
duplicate the dispatcher, scheduler, or worker_runner.

Phase 6 (Success Evaluator): consumes Phase 1+5 persisted state and
produces a deterministic EvaluationReport. It does NOT spawn
workers, does NOT call Orchestrator / Dispatcher / BatchRunner,
does NOT execute LLMs, does NOT make provider API calls, and does
NOT modify Runtime.

Phase 7 (Objective Recovery): consumes Phase 1-6 persisted state
and produces a deterministic RecoveryDiagnosis + RecoveryPlanPreview.
It does NOT spawn workers, does NOT create Kanban, does NOT call
Orchestrator/Dispatcher/BatchRunner/GoalManager/Planner/WorkerDispatch/KanbanApply,
does NOT revalidate approval gates, and does NOT modify Runtime.

This package re-exports the public API surface for each phase.
Tests import from the submodules directly; this ``__init__`` is
intentionally minimal to avoid a circular import path.
"""

# Re-export submodules so they can be accessed as
# ``agent.executive.worker_dispatch`` etc.
from . import (
    approval_gates,
    goalmanager_bridge,
    kanban_apply,
    kanban_mapping,
    orchestrator_preview,
    planner,
    policy,
    recovery_diagnosis,
    recovery_engine,
    risk,
    worker_dispatch,
    worker_mapping,
    success_metrics,
    success_evaluator,
)

# Re-export the public errors (reused across phases).
from .goalmanager_bridge import (
    BridgeApprovalError,
    BridgeError,
    BridgeLinkageConflictError,
    BridgeMappingError,
)
from .kanban_apply import KanbanLinkageConflictError

# Re-export the engine classes.
from .worker_dispatch import (
    WorkerDispatchEngine,
    worker_dispatch_apply,
    worker_dispatch_dry_run,
    worker_dispatch_rollback,
)
from .success_evaluator import (
    SuccessEvaluatorEngine,
    SuccessEvaluatorError,
    SuccessEvaluatorMappingError,
    success_evaluator_dry_run,
    success_evaluator_evaluate,
    success_evaluator_persist,
    success_evaluator_rollback,
)
from .recovery_engine import (
    ObjectiveRecoveryEngine,
    RecoveryError,
    RecoveryMappingError,
    recovery_dry_run,
    recovery_preview,
    recovery_evaluate,
    recovery_persist,
    recovery_rollback,
)
from .types import (
    SuccessStatus,
    TaskOutcome,
    SuccessMetricBreakdown,
    EvaluationReport,
    SuccessReport,
    # Phase 7
    RecoveryStatus,
    RecoveryAction,
    RecoveryDiagnosis,
    RecoveryPlanPreview,
)

__all__ = [
    # Submodules
    "approval_gates",
    "goalmanager_bridge",
    "kanban_apply",
    "kanban_mapping",
    "orchestrator_preview",
    "planner",
    "policy",
    "risk",
    "worker_dispatch",
    "worker_mapping",
    # Errors
    "BridgeApprovalError",
    "BridgeError",
    "BridgeLinkageConflictError",
    "BridgeMappingError",
    "KanbanLinkageConflictError",
    # Phase 5 engine
    "WorkerDispatchEngine",
    "worker_dispatch_dry_run",
    "worker_dispatch_apply",
    "worker_dispatch_rollback",
    # Phase 6
    "SuccessEvaluatorEngine",
    "SuccessEvaluatorError",
    "SuccessEvaluatorMappingError",
    "SuccessStatus",
    "TaskOutcome",
    "EvaluationReport",
    "SuccessMetricBreakdown",
    "SuccessReport",
    "success_evaluator_dry_run",
    "success_evaluator_evaluate",
    "success_evaluator_persist",
    "success_evaluator_rollback",
    # Phase 7
    "ObjectiveRecoveryEngine",
    "RecoveryError",
    "RecoveryMappingError",
    "RecoveryStatus",
    "RecoveryAction",
    "RecoveryDiagnosis",
    "RecoveryPlanPreview",
    "recovery_dry_run",
    "recovery_preview",
    "recovery_evaluate",
    "recovery_persist",
    "recovery_rollback",
]
