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

from __future__ import annotations

from importlib import import_module as _import_module


# Keep package import hermetic: Executive v2 runtime modules are
# resolved only when their public attributes are actually requested.
_LAZY_ATTRS = {
    'approval_gates': ('.approval_gates', None),
    'goalmanager_bridge': ('.goalmanager_bridge', None),
    'kanban_apply': ('.kanban_apply', None),
    'kanban_mapping': ('.kanban_mapping', None),
    'orchestrator_preview': ('.orchestrator_preview', None),
    'planner': ('.planner', None),
    'policy': ('.policy', None),
    'recovery_diagnosis': ('.recovery_diagnosis', None),
    'recovery_engine': ('.recovery_engine', None),
    'risk': ('.risk', None),
    'worker_dispatch': ('.worker_dispatch', None),
    'worker_mapping': ('.worker_mapping', None),
    'success_metrics': ('.success_metrics', None),
    'success_evaluator': ('.success_evaluator', None),
    'types': ('.types', None),
    'services': ('.services', None),
    'BridgeApprovalError': ('.goalmanager_bridge', 'BridgeApprovalError'),
    'BridgeError': ('.goalmanager_bridge', 'BridgeError'),
    'BridgeLinkageConflictError': ('.goalmanager_bridge', 'BridgeLinkageConflictError'),
    'BridgeMappingError': ('.goalmanager_bridge', 'BridgeMappingError'),
    'KanbanLinkageConflictError': ('.kanban_apply', 'KanbanLinkageConflictError'),
    'WorkerDispatchEngine': ('.worker_dispatch', 'WorkerDispatchEngine'),
    'worker_dispatch_apply': ('.worker_dispatch', 'worker_dispatch_apply'),
    'worker_dispatch_dry_run': ('.worker_dispatch', 'worker_dispatch_dry_run'),
    'worker_dispatch_rollback': ('.worker_dispatch', 'worker_dispatch_rollback'),
    'SuccessEvaluatorEngine': ('.success_evaluator', 'SuccessEvaluatorEngine'),
    'SuccessEvaluatorError': ('.success_evaluator', 'SuccessEvaluatorError'),
    'SuccessEvaluatorMappingError': ('.success_evaluator', 'SuccessEvaluatorMappingError'),
    'success_evaluator_dry_run': ('.success_evaluator', 'success_evaluator_dry_run'),
    'success_evaluator_evaluate': ('.success_evaluator', 'success_evaluator_evaluate'),
    'success_evaluator_persist': ('.success_evaluator', 'success_evaluator_persist'),
    'success_evaluator_rollback': ('.success_evaluator', 'success_evaluator_rollback'),
    'ObjectiveRecoveryEngine': ('.recovery_engine', 'ObjectiveRecoveryEngine'),
    'RecoveryError': ('.recovery_engine', 'RecoveryError'),
    'RecoveryMappingError': ('.recovery_engine', 'RecoveryMappingError'),
    'recovery_dry_run': ('.recovery_engine', 'recovery_dry_run'),
    'recovery_preview': ('.recovery_engine', 'recovery_preview'),
    'recovery_evaluate': ('.recovery_engine', 'recovery_evaluate'),
    'recovery_persist': ('.recovery_engine', 'recovery_persist'),
    'recovery_rollback': ('.recovery_engine', 'recovery_rollback'),
    'SuccessStatus': ('.types', 'SuccessStatus'),
    'TaskOutcome': ('.types', 'TaskOutcome'),
    'SuccessMetricBreakdown': ('.types', 'SuccessMetricBreakdown'),
    'EvaluationReport': ('.types', 'EvaluationReport'),
    'SuccessReport': ('.types', 'SuccessReport'),
    'RecoveryStatus': ('.types', 'RecoveryStatus'),
    'RecoveryAction': ('.types', 'RecoveryAction'),
    'RecoveryDiagnosis': ('.types', 'RecoveryDiagnosis'),
    'RecoveryPlanPreview': ('.types', 'RecoveryPlanPreview'),
    'EvidencePackDegradeReason': ('.services', 'EvidencePackDegradeReason'),
    'EvidencePackEngineFactory': ('.services', 'EvidencePackEngineFactory'),
    'EvidencePackStatus': ('.services', 'EvidencePackStatus'),
    'EvidencePackStorageUnavailable': ('.services', 'EvidencePackStorageUnavailable'),
    'ObjectiveServices': ('.services', 'ObjectiveServices'),
    'build_objective_services': ('.services', 'build_objective_services'),
}

__all__ = [
    'approval_gates',
    'goalmanager_bridge',
    'kanban_apply',
    'kanban_mapping',
    'orchestrator_preview',
    'planner',
    'policy',
    'risk',
    'worker_dispatch',
    'worker_mapping',
    'BridgeApprovalError',
    'BridgeError',
    'BridgeLinkageConflictError',
    'BridgeMappingError',
    'KanbanLinkageConflictError',
    'WorkerDispatchEngine',
    'worker_dispatch_dry_run',
    'worker_dispatch_apply',
    'worker_dispatch_rollback',
    'SuccessEvaluatorEngine',
    'SuccessEvaluatorError',
    'SuccessEvaluatorMappingError',
    'SuccessStatus',
    'TaskOutcome',
    'EvaluationReport',
    'SuccessMetricBreakdown',
    'SuccessReport',
    'success_evaluator_dry_run',
    'success_evaluator_evaluate',
    'success_evaluator_persist',
    'success_evaluator_rollback',
    'ObjectiveRecoveryEngine',
    'RecoveryError',
    'RecoveryMappingError',
    'RecoveryStatus',
    'RecoveryAction',
    'RecoveryDiagnosis',
    'RecoveryPlanPreview',
    'recovery_dry_run',
    'recovery_preview',
    'recovery_evaluate',
    'recovery_persist',
    'recovery_rollback',
    'EvidencePackDegradeReason',
    'EvidencePackEngineFactory',
    'EvidencePackStatus',
    'EvidencePackStorageUnavailable',
    'ObjectiveServices',
    'build_objective_services',
]


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute = target
    module = _import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(__all__))
