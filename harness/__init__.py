"""Dynamic agent harness: execution governance around the Hermes agent.

The agent reasons; the harness controls execution — tasks, features,
budgets, verification, recovery, persistence, and completion evidence.
"""

from .adapter import chat_step
from .budget import BudgetGovernor, BudgetUsage
from .knowledge import (
    KnowledgeCandidate,
    extract,
    is_durable,
    knowledge_id,
    resolve_conflict,
)
from .loop import AgentStep, HarnessRunner, StepResult
from .recovery import FailureClass, Strategy, classify_failure, decide, progress_made
from .state import (
    TERMINAL_OUTCOMES,
    TERMINAL_STATUSES,
    Checkpoint,
    ExecutionBudget,
    ExecutionState,
    FeatureLock,
    FeatureState,
    FeatureStatus,
    KnowledgeItem,
    KnowledgeType,
    Outcome,
    RiskLevel,
    ScopeReason,
    ScopeRejected,
    StepStatus,
    Task,
    TaskStatus,
    TaskType,
    ToolObservation,
    VerificationCheck,
    VerificationResult,
)
from .store import HarnessStore
from .verify import (
    CheckStrength,
    command_check,
    completion_allowed,
    file_contains_check,
    pytest_check,
    run_all,
    verify,
)

__all__ = [
    "AgentStep",
    "BudgetGovernor",
    "BudgetUsage",
    "CheckStrength",
    "Checkpoint",
    "ExecutionBudget",
    "ExecutionState",
    "FailureClass",
    "FeatureLock",
    "FeatureState",
    "FeatureStatus",
    "HarnessRunner",
    "HarnessStore",
    "KnowledgeCandidate",
    "KnowledgeItem",
    "KnowledgeType",
    "Outcome",
    "RiskLevel",
    "ScopeReason",
    "ScopeRejected",
    "StepResult",
    "StepStatus",
    "Strategy",
    "Task",
    "TaskStatus",
    "TaskType",
    "ToolObservation",
    "VerificationCheck",
    "VerificationResult",
    "chat_step",
    "classify_failure",
    "command_check",
    "completion_allowed",
    "decide",
    "extract",
    "file_contains_check",
    "is_durable",
    "knowledge_id",
    "progress_made",
    "pytest_check",
    "resolve_conflict",
    "run_all",
    "verify",
    "TERMINAL_OUTCOMES",
    "TERMINAL_STATUSES",
]
