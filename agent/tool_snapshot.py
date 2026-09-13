"""Bind advertised tool contracts through provider requests and handler start."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import json
import threading
from typing import Any


class ToolSnapshotChangedError(RuntimeError):
    """No handler started for a response whose advertised contract changed."""


class ToolSnapshotRefreshError(RuntimeError):
    """A stale contract could not be replaced by a coherent snapshot."""


@dataclass
class ToolExecutionSnapshot:
    agent: Any
    epoch: int
    started: int = 0
    stale: bool = False
    effects_unknown: bool = False
    lock: Any = field(default_factory=threading.Lock)

    @property
    def effects_started(self):
        with self.lock:
            return self.started > 0 or self.effects_unknown


_REQUEST: ContextVar[tuple[Any, int] | None] = ContextVar("tool_request_snapshot", default=None)
_EXECUTION: ContextVar[ToolExecutionSnapshot | None] = ContextVar("tool_execution_snapshot", default=None)
_ROUTE: ContextVar[tuple[str, str, Any] | None] = ContextVar("tool_execution_route", default=None)


@contextmanager
def bind_tool_request_snapshot(tools, epoch):
    token = _REQUEST.set((tools, epoch))
    try:
        yield
    finally:
        _REQUEST.reset(token)


def capture_tool_request_snapshot(agent):
    from tools.mcp_tool_agent import capture_agent_tool_request_snapshot
    return _REQUEST.get() or capture_agent_tool_request_snapshot(agent)


class ToolSnapshotAPIKwargs(dict):
    """Local epoch metadata is an attribute, never a provider wire parameter."""
    __slots__ = ("_hermes_tool_snapshot_epoch",)


def bind_tool_snapshot_epoch(kwargs, epoch):
    bound = ToolSnapshotAPIKwargs(kwargs)
    bound._hermes_tool_snapshot_epoch = epoch
    return bound


def expected_tool_snapshot_epoch(message):
    epoch = getattr(message, "_hermes_tool_snapshot_epoch", None)
    return epoch if isinstance(epoch, int) else None


def require_current_tool_snapshot(agent, message):
    from tools.mcp_tool_agent import agent_tool_snapshot_epoch_is_current
    epoch = expected_tool_snapshot_epoch(message)
    if epoch is not None and not agent_tool_snapshot_epoch_is_current(agent, epoch):
        raise ToolSnapshotChangedError("tool snapshot changed while the model request was in flight")


@contextmanager
def bind_tool_execution_snapshot(agent, message):
    """Workers inherit this request-local state through the existing context propagation."""
    epoch = expected_tool_snapshot_epoch(message)
    state = ToolExecutionSnapshot(agent, epoch) if epoch is not None else None
    token = _EXECUTION.set(state)
    try:
        require_current_tool_snapshot(agent, message)
        yield state
        if state is not None and state.stale:
            if not state.effects_started:
                raise ToolSnapshotChangedError("all tool calls stopped before handler start")
            refresh_tool_snapshot_after_stale(agent, message)
    finally:
        _EXECUTION.reset(token)


def captured_tool_route(function_name):
    route = _ROUTE.get()
    return route[1:] if route is not None and route[0] == function_name else None


@contextmanager
def bind_tool_execution_route(agent, function_name):
    """Validate and retain the route after authorization, before checkpoints or handler effects."""
    state = _EXECUTION.get()
    if state is None or state.agent is not agent:
        yield
        return
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    from tools.mcp_tool_agent import (
        agent_tool_snapshot_epoch_is_current, capture_agent_tool_execution_route,
    )
    route = capture_agent_tool_execution_route(agent, state.epoch, function_name)
    from tools.tool_gateway.names import is_connector_name
    inline = function_name in INLINE_TOOL_EXECUTORS
    dynamic_owned = memory_provider_owns_tool(agent, function_name) or function_name in (
        getattr(agent, "_context_engine_tool_names", None) or set()
    )
    routes = getattr(agent, "_tool_registry_routes", None)
    if (
        route is None and not dynamic_owned and isinstance(routes, dict)
        and function_name not in routes
        and function_name in (getattr(agent, "valid_tool_names", None) or set())
    ):
        # Trusted direct callers may extend tools/valid_tool_names without the
        # optional route map. Capture that legacy route once; an advertised
        # route that was replaced or removed never takes this fallback.
        from tools.registry import registry
        entry = registry.get_entry(function_name)
        if entry is not None:
            route = ("registry", entry)
    if not agent_tool_snapshot_epoch_is_current(agent, state.epoch) or (
        route is None and (dynamic_owned or not inline) and not is_connector_name(function_name)
    ):
        state.stale = True
        raise ToolSnapshotChangedError("tool snapshot changed before advertised handler start")
    token = _ROUTE.set((function_name, *route) if route is not None else None)
    try:
        with state.lock:
            state.started += 1
        try:
            yield
        except ToolSnapshotChangedError:
            # This exception certifies no handler effect, including an inner
            # dispatcher that detects staleness after the outer authorization.
            state.stale = True
            with state.lock:
                state.started -= 1
            raise
    finally:
        _ROUTE.reset(token)


def refresh_tool_snapshot_after_stale(agent, message):
    from tools.mcp_tool_agent import agent_tool_snapshot_epoch_is_current, refresh_agent_mcp_tools
    epoch = expected_tool_snapshot_epoch(message)
    if epoch is None or not agent_tool_snapshot_epoch_is_current(agent, epoch):
        return
    try:
        refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True)
    except Exception as exc:
        raise ToolSnapshotRefreshError("current tool snapshot could not be refreshed safely") from exc


def memory_provider_owns_tool(agent, function_name):
    names = getattr(agent, "_memory_provider_tool_names", None)
    if isinstance(names, set):
        return function_name in names
    manager = getattr(agent, "_memory_manager", None)
    has_tool = getattr(manager, "has_tool", None)
    return bool(callable(has_tool) and has_tool(function_name))


def snapshot_bound_tool_batch(execute):
    @wraps(execute)
    def bound(agent, message, *args, **kwargs):
        active = _EXECUTION.get()
        if active is not None and active.agent is agent:
            return execute(agent, message, *args, **kwargs)
        with bind_tool_execution_snapshot(agent, message):
            return execute(agent, message, *args, **kwargs)
    return bound


def stale_tool_result(function_name):
    return json.dumps({"error": f"Tool snapshot changed; '{function_name}' was not started",
                       "error_type": "tool_snapshot_changed"})


def execute_dynamic_tool(function_name, fallback, args, **kwargs):
    route = captured_tool_route(function_name)
    handler = route[1] if route is not None else fallback
    return handler(function_name, args, **kwargs)


def mark_tool_effects_unknown(agent):
    """An abandoned worker cannot certify that the whole batch had no effects."""
    state = _EXECUTION.get()
    if state is not None and state.agent is agent:
        with state.lock:
            state.effects_unknown = True


@contextmanager
def direct_tool_execution(agent, epoch, function_name):
    """Direct dispatcher calls carry explicit metadata; agent calls inherit it."""
    if agent is None or epoch is None:
        yield
        return
    from types import SimpleNamespace
    message = SimpleNamespace(_hermes_tool_snapshot_epoch=epoch)
    with bind_tool_execution_snapshot(agent, message):
        with bind_tool_execution_route(agent, function_name):
            yield
