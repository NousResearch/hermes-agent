"""Immutable execution attribution exposed to native plugins.

The current value identifies the agent turn whose execution context is running. It is
metadata only: it grants no authority and does not prove that the turn is still live.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class AgentContext:
    session_id: str
    task_id: str
    turn_id: str
    platform: str
    source: str
    parent_session_id: str | None


_CURRENT_AGENT_CONTEXT: ContextVar[AgentContext | None] = ContextVar(
    "hermes_plugin_agent_context",
    default=None,
)


def get_agent_context() -> AgentContext | None:
    """Return the bound turn snapshot, or ``None`` outside an agent turn."""
    return _CURRENT_AGENT_CONTEXT.get()


@contextmanager
def bind_agent_context(context: AgentContext) -> Iterator[None]:
    """Bind one turn snapshot and restore any enclosing binding on exit."""
    token = _CURRENT_AGENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_AGENT_CONTEXT.reset(token)
