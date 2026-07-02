"""Hermes Agent runtime API foundation.

Provides the models and run manager that power the /v1/runs contract.
Isolated from the existing api_server.py adapter so both the current
aiohttp API server and future WebUI adapters can share a single
runtime run manager.
"""

from gateway.runtime.models import (
    RuntimeEvent,
    RuntimeStatus,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_AWAITING_APPROVAL,
    RUN_STATUS_AWAITING_CLARIFY,
    RUN_STATUS_PAUSED,
    RUN_STATUS_CANCELLING,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_EXPIRED,
    TERMINAL_STATUSES,
    EVENT_RUN_STARTED,
    EVENT_RUN_STATUS,
    EVENT_TOKEN_DELTA,
    EVENT_REASONING_DELTA,
    EVENT_REASONING_DONE,
    EVENT_PROGRESS,
    EVENT_TOOL_STARTED,
    EVENT_TOOL_UPDATED,
    EVENT_TOOL_DONE,
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_RESOLVED,
    EVENT_CLARIFY_REQUESTED,
    EVENT_CLARIFY_RESOLVED,
    EVENT_TITLE_UPDATED,
    EVENT_USAGE_UPDATED,
    EVENT_USAGE_FINAL,
    EVENT_ERROR,
    EVENT_DONE,
    TERMINAL_EVENT_TYPES,
    redact_secrets,
)

from gateway.runtime.run_manager import RunManager
from gateway.runtime.routes import register_runtime_routes
from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.executor import (
    RuntimeExecutor,
    AgentFactory,
    FakeAgentFactory,
    SessionKeyFactory,
)
from gateway.runtime.agent_factory import DefaultAgentFactory

__all__ = [
    "RuntimeEvent",
    "RuntimeStatus",
    "RunManager",
    "RuntimeControlBridge",
    "RUN_STATUS_QUEUED",
    "RUN_STATUS_RUNNING",
    "RUN_STATUS_AWAITING_APPROVAL",
    "RUN_STATUS_AWAITING_CLARIFY",
    "RUN_STATUS_PAUSED",
    "RUN_STATUS_CANCELLING",
    "RUN_STATUS_CANCELLED",
    "RUN_STATUS_FAILED",
    "RUN_STATUS_COMPLETED",
    "RUN_STATUS_EXPIRED",
    "TERMINAL_STATUSES",
    "EVENT_RUN_STARTED",
    "EVENT_RUN_STATUS",
    "EVENT_TOKEN_DELTA",
    "EVENT_REASONING_DELTA",
    "EVENT_REASONING_DONE",
    "EVENT_PROGRESS",
    "EVENT_TOOL_STARTED",
    "EVENT_TOOL_UPDATED",
    "EVENT_TOOL_DONE",
    "EVENT_APPROVAL_REQUESTED",
    "EVENT_APPROVAL_RESOLVED",
    "EVENT_CLARIFY_REQUESTED",
    "EVENT_CLARIFY_RESOLVED",
    "EVENT_TITLE_UPDATED",
    "EVENT_USAGE_UPDATED",
    "EVENT_USAGE_FINAL",
    "EVENT_ERROR",
    "EVENT_DONE",
    "TERMINAL_EVENT_TYPES",
    "redact_secrets",
    "register_runtime_routes",
    "RuntimeControlBridge",
    "RuntimeExecutor",
    "AgentFactory",
    "FakeAgentFactory",
    "SessionKeyFactory",
    "DefaultAgentFactory",
]
