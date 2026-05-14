"""Core workflow orchestration primitives for Hermes.

This package owns workflow policy, storage, DAG normalization, gates, artifacts,
and Kanban materialization. WebUI and gateway layers should serialize from these
modules instead of inferring workflow state from Kanban prose.
"""

from .api import (
    get_inbox_item_detail,
    get_workflow_artifacts,
    get_workflow_dag,
    get_workflow_events,
    get_workflow_node,
    list_inbox_item_summaries,
    list_workflow_summaries,
    materialize_workflow_to_kanban,
    update_inbox_item_triage,
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
    WorkflowInboxItem,
    WorkflowRecord,
    add_artifact,
    add_event,
    add_gate,
    connect,
    create_inbox_item,
    create_workflow,
    get_inbox_item,
    get_workflow,
    list_artifacts,
    list_events,
    list_gates,
    list_inbox_items,
    list_workflows,
    resolve_gate,
    save_dag,
    update_inbox_item,
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
    "WorkflowInboxItem",
    "WorkflowPolicy",
    "WorkflowRecord",
    "WorkflowValidationError",
    "add_artifact",
    "add_event",
    "add_gate",
    "allocate_worktrees",
    "connect",
    "create_inbox_item",
    "create_workflow",
    "get_inbox_item",
    "get_inbox_item_detail",
    "get_workflow",
    "get_workflow_artifacts",
    "get_workflow_dag",
    "get_workflow_events",
    "get_workflow_node",
    "list_artifacts",
    "list_events",
    "list_gates",
    "list_inbox_item_summaries",
    "list_inbox_items",
    "list_workflow_summaries",
    "list_workflows",
    "load_policy",
    "materialize_workflow",
    "materialize_workflow_to_kanban",
    "normalize_dag",
    "resolve_gate",
    "save_dag",
    "update_inbox_item",
    "update_inbox_item_triage",
    "validate_dag",
]
