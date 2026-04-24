"""Built-in middlewares for Hermes agent_bus.

Middlewares are NOT auto-registered on import. Call `register_defaults()`
once during application startup to wire them in. Tests can build their own
subset via the individual classes.

Order slots (align with DeerFlow's 18-stage chain):
  10 — TracingMiddleware           (all hooks) — S10
  20 — ThreadDataMiddleware        (before_model/tool) — S7 substrate
  40 — DanglingToolCallMiddleware  (before_model)
  60 — GuardrailMiddleware         (before_tool)
  90 — SummarizationMiddleware     (before_model)
 100 — TodoListMiddleware          (after_model)
 130 — MemoryExtractionMiddleware  (after_model, on_session_end)
 170 — LoopDetectionMiddleware     (after_model)
"""

from __future__ import annotations

from agent_bus.middleware import register
from agent_bus.middlewares.dangling_tool_call import DanglingToolCallMiddleware
from agent_bus.middlewares.guardrail import GuardrailMiddleware
from agent_bus.middlewares.loop_detection import LoopDetectionMiddleware
from agent_bus.middlewares.memory_extraction import MemoryExtractionMiddleware
from agent_bus.middlewares.summarization import SummarizationMiddleware
from agent_bus.middlewares.thread_data import ThreadDataMiddleware
from agent_bus.middlewares.todo_list import TodoListMiddleware
from agent_bus.middlewares.tracing import TracingMiddleware

__all__ = [
    "DanglingToolCallMiddleware",
    "GuardrailMiddleware",
    "LoopDetectionMiddleware",
    "MemoryExtractionMiddleware",
    "SummarizationMiddleware",
    "ThreadDataMiddleware",
    "TodoListMiddleware",
    "TracingMiddleware",
    "register_defaults",
]


def register_defaults() -> None:
    register(order=10, env_var="HERMES_MW_TRACING", critical=False)(TracingMiddleware)
    register(order=20, env_var="HERMES_MW_THREAD_DATA", critical=False)(ThreadDataMiddleware)
    register(order=40, env_var="HERMES_MW_DANGLING_TOOL", critical=False)(DanglingToolCallMiddleware)
    register(order=60, env_var="HERMES_MW_GUARDRAIL", critical=False)(GuardrailMiddleware)
    register(order=90, env_var="HERMES_MW_SUMMARIZATION", critical=False)(SummarizationMiddleware)
    register(order=100, env_var="HERMES_MW_TODO_LIST", critical=False)(TodoListMiddleware)
    register(order=130, env_var="HERMES_MW_MEMORY_EXTRACT", critical=False)(MemoryExtractionMiddleware)
    register(order=170, env_var="HERMES_MW_LOOP_DETECT", critical=False)(LoopDetectionMiddleware)
