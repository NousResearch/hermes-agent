from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

__all__ = [
    "CARD",
    "LINK",
    "SIDE",
    "SIDE_AGENT_TOOL_DROPS",
    "agent_connection_surface",
    "connection_surface",
    "scoped_connection_surface",
    "side_agent_tool_drops",
]

CARD = "card"
SIDE = "side"
LINK = "link"

_SURFACE: ContextVar[str] = ContextVar("hermes_connection_surface", default=LINK)


def connection_surface() -> str:
    return _SURFACE.get()


def agent_connection_surface(agent: Any) -> str:
    if getattr(agent, "connection_callback", None) is not None:
        return CARD
    return SIDE if getattr(agent, "side_agent", False) else LINK


SIDE_AGENT_TOOL_DROPS = frozenset({"manage_connections"})


def side_agent_tool_drops(agent: Any) -> frozenset:
    return SIDE_AGENT_TOOL_DROPS if getattr(agent, "side_agent", False) else frozenset()


@contextmanager
def scoped_connection_surface(surface: str) -> Iterator[None]:
    token = _SURFACE.set(surface)
    try:
        yield
    finally:
        _SURFACE.reset(token)
