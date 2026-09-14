"""Runtime adapters — the only place NOVA knows what runtime it is running on.

Everything above this package speaks NOVA vocabulary (agents, identity, tenants). An
adapter translates that into one runtime's concepts and is the sole holder of that
knowledge. Swapping runtimes is therefore a matter of writing one adapter, not of
rewriting the platform.
"""

from nova.runtime.base import (
    TASK_STATES,
    AgentRuntime,
    ExtractedDocument,
    MaterializedAgent,
    MaterializeResult,
    RuntimeCapabilities,
    ModelUsage,
    RuntimeHealth,
    TaskView,
    UsageSummary,
)
from nova.runtime.registry import available_runtimes, get_runtime, register_runtime

__all__ = [
    "TASK_STATES",
    "AgentRuntime",
    "ExtractedDocument",
    "MaterializedAgent",
    "MaterializeResult",
    "ModelUsage",
    "RuntimeCapabilities",
    "RuntimeHealth",
    "TaskView",
    "UsageSummary",
    "available_runtimes",
    "get_runtime",
    "register_runtime",
]
