"""Runtime state foundations for Hermes Agent.

This package is intentionally additive. It defines serializable state primitives
that future runtime, cron, and self-evolution features can opt into without
changing the current Gateway or agent loop behavior.
"""

from .state import (
    AgentRuntimeState,
    ModelRuntimeState,
    RuntimeMutation,
    RuntimeMutationRisk,
    RuntimeMutationType,
    RuntimeSessionState,
    ToolRuntimeState,
)
from .state_store import RuntimeStateCheckpoint, RuntimeStateStore

__all__ = [
    "AgentRuntimeState",
    "ModelRuntimeState",
    "RuntimeMutation",
    "RuntimeMutationRisk",
    "RuntimeMutationType",
    "RuntimeSessionState",
    "RuntimeStateCheckpoint",
    "RuntimeStateStore",
    "ToolRuntimeState",
]
