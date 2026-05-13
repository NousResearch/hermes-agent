"""Core workflow orchestration primitives for Hermes.

This package owns workflow policy, storage, DAG normalization, gates, artifacts,
and Kanban materialization. WebUI and gateway layers should serialize from these
modules instead of inferring workflow state from Kanban prose.
"""

from .api import (
    get_workflow_artifacts,
    get_workflow_dag,
    get_workflow_events,
    get_workflow_node,
    list_workflow_summaries,
)
from .dag import DagValidationResult, normalize_dag, validate_dag
from .errors import WorkflowError, WorkflowValidationError
from .materialize import MaterializationResult, MaterializedTask, materialize_workflow
from .policy import DEFAULT_POLICY, PolicyLoadResult, WorkflowPolicy, load_policy
from .worktrees import allocate_worktrees
from .store import (
    WorkflowArtifact,
    WorkflowEvent,
    WorkflowGate,
    WorkflowRecord,
    add_artifact,
    add_event,
    add_gate,
    connect,
    create_workflow,
    get_workflow,
    list_artifacts,
    list_events,
    list_gates,
    list_workflows,
    resolve_gate,
    save_dag,
)

__all__ = [
    "DEFAULT_POLICY",
    "DagValidationResult",
    "MaterializationResult",
    "MaterializedTask",
    "PolicyLoadResult",
    "WorkflowArtifact",
    "WorkflowError",
    "WorkflowEvent",
    "WorkflowGate",
    "WorkflowPolicy",
    "WorkflowRecord",
    "WorkflowValidationError",
    "add_artifact",
    "add_event",
    "add_gate",
    "allocate_worktrees",
    "connect",
    "create_workflow",
    "get_workflow",
    "get_workflow_artifacts",
    "get_workflow_dag",
    "get_workflow_events",
    "get_workflow_node",
    "list_artifacts",
    "list_events",
    "list_gates",
    "list_workflow_summaries",
    "list_workflows",
    "load_policy",
    "materialize_workflow",
    "normalize_dag",
    "resolve_gate",
    "save_dag",
    "validate_dag",
]
