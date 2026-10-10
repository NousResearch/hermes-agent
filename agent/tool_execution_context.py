"""Trusted, request-local identity for tool consumers, without exposing an agent."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, Mapping
import weakref

_CURRENT: ContextVar[dict[str, Any] | None] = ContextVar("hermes_tool_execution_context", default=None)


def current_tool_execution_context() -> Mapping[str, Any]:
    """Return an immutable snapshot; an unbound caller has no supervisor authority.

    Identity comes from runtime agent objects, never tool arguments. This is an
    in-process plugin contract, not a sandbox against arbitrary Python code.
    """
    value = _CURRENT.get()
    return MappingProxyType(dict(value) if value is not None else {
        "session_id": "", "root_session_id": "", "parent_session_id": "",
        "delegate_depth": None, "lineage_valid": False, "profile_home": "",
        "task_id": "", "tool_call_id": "", "turn_id": "", "api_request_id": "",
    })


def _lineage(agent: Any) -> tuple[str, str, int | None, bool]:
    depth = getattr(agent, "_delegate_depth", None)
    if type(depth) is not int or not 0 <= depth <= 128:
        return "", "", None, False
    current, parent_id, seen = agent, "", set()
    for expected_depth in range(depth, -1, -1):
        sid = getattr(current, "session_id", None)
        if not isinstance(sid, str) or not sid or sid in seen:
            return "", "", depth, False
        seen.add(sid)
        actual_depth = getattr(current, "_delegate_depth", None)
        if type(actual_depth) is not int or actual_depth != expected_depth:
            return "", "", depth, False
        ref = getattr(current, "_delegate_parent_ref", None)
        if expected_depth == 0:
            return (sid, parent_id, depth, True) if ref is None else ("", "", depth, False)
        if not isinstance(ref, weakref.ReferenceType):
            return "", "", depth, False
        current = ref()
        if current is None:
            return "", "", depth, False
        if expected_depth == depth:
            parent_id = getattr(current, "session_id", "")
    return "", "", depth, False


@contextmanager
def bind_tool_execution_context(agent: Any, *, task_id: str = "", tool_call_id: str = ""):
    """Bind on the thread doing dispatch, including sequential and worker paths."""
    from hermes_constants import get_hermes_home

    root, parent, depth, valid = _lineage(agent)
    value = {
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "root_session_id": root, "parent_session_id": parent,
        "delegate_depth": depth, "lineage_valid": valid,
        "profile_home": str(get_hermes_home()),
        "task_id": task_id or "", "tool_call_id": tool_call_id or "",
        "turn_id": str(getattr(agent, "_current_turn_id", "") or ""),
        "api_request_id": str(getattr(agent, "_current_api_request_id", "") or ""),
    }
    token = _CURRENT.set(value)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def dispatch_in_tool_context(agent: Any, callback, *, task_id: str = "", tool_call_id: str = ""):
    """Run an actual dispatch under its own runtime identity, then restore the parent."""
    with bind_tool_execution_context(agent, task_id=task_id, tool_call_id=tool_call_id):
        return callback()
