"""Core workflow orchestration primitives for Hermes.

This package owns workflow policy, storage, DAG normalization, gates, artifacts,
and Kanban materialization. WebUI and gateway layers should serialize from these
modules instead of inferring workflow state from Kanban prose.
"""

from .dag import DagValidationResult, normalize_dag, validate_dag
from .errors import WorkflowError, WorkflowValidationError
from .policy import DEFAULT_POLICY, PolicyLoadResult, WorkflowPolicy, load_policy
from .worktrees import allocate_worktrees
from .store import (
    WorkflowArtifact,
    WorkflowEvent,
    WorkflowRecord,
    add_artifact,
    add_event,
    connect,
    create_workflow,
    get_workflow,
    list_artifacts,
    list_events,
    list_workflows,
    save_dag,
)

__all__ = [
    "DEFAULT_POLICY",
    "DagValidationResult",
    "PolicyLoadResult",
    "WorkflowArtifact",
    "WorkflowError",
    "WorkflowEvent",
    "WorkflowPolicy",
    "WorkflowRecord",
    "WorkflowValidationError",
    "add_artifact",
    "add_event",
    "allocate_worktrees",
    "connect",
    "create_workflow",
    "get_workflow",
    "list_artifacts",
    "list_events",
    "list_workflows",
    "load_policy",
    "normalize_dag",
    "save_dag",
    "validate_dag",
]
