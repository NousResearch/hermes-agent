"""Built-in middlewares for Hermes agent_bus.

Middlewares are NOT auto-registered on import. Call `register_defaults()`
once during application startup to wire them in. Tests can build their own
subset via the individual classes.
"""

from __future__ import annotations

from agent_bus.middleware import register
from agent_bus.middlewares.dangling_tool_call import DanglingToolCallMiddleware
from agent_bus.middlewares.loop_detection import LoopDetectionMiddleware

__all__ = [
    "DanglingToolCallMiddleware",
    "LoopDetectionMiddleware",
    "register_defaults",
]


def register_defaults() -> None:
    """Register all built-in middlewares at their canonical `order` slots.

    Order slots match DeerFlow's 18-stage chain to make future additions
    line up naturally:
    - 40: DanglingToolCallMiddleware (before_model)
    - 170: LoopDetectionMiddleware (after_model)
    """
    register(order=40, env_var="HERMES_MW_DANGLING_TOOL", critical=False)(DanglingToolCallMiddleware)
    register(order=170, env_var="HERMES_MW_LOOP_DETECT", critical=False)(LoopDetectionMiddleware)
