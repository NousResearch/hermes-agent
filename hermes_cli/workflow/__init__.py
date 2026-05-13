"""Core workflow orchestration primitives for Hermes.

This package owns workflow policy, storage, DAG normalization, gates, artifacts,
and Kanban materialization. WebUI and gateway layers should serialize from these
modules instead of inferring workflow state from Kanban prose.
"""

from .dag import DagValidationResult, normalize_dag, validate_dag
from .errors import WorkflowError, WorkflowValidationError
from .policy import DEFAULT_POLICY, PolicyLoadResult, WorkflowPolicy, load_policy
from .store import (
    WorkflowEvent,
    WorkflowRecord,
    add_event,
    connect,
    create_workflow,
    get_workflow,
    list_events,
    list_workflows,
)

__all__ = [
    "DEFAULT_POLICY",
    "DagValidationResult",
    "PolicyLoadResult",
    "WorkflowError",
    "WorkflowEvent",
    "WorkflowPolicy",
    "WorkflowRecord",
    "WorkflowValidationError",
    "add_event",
    "connect",
    "create_workflow",
    "get_workflow",
    "list_events",
    "list_workflows",
    "load_policy",
    "normalize_dag",
    "validate_dag",
]
