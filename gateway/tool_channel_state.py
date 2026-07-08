"""In-process state for API-server split-runtime tool requests.

The API server owns the model/agent loop. A local executor client can attach to
``/v1/runs/{run_id}/events`` and receive ``tool.request`` events for a small
allowlisted tool surface. This module is the synchronous block/resume registry
between the tool worker thread and the HTTP/SSE handlers.
"""

from __future__ import annotations

import contextvars
import threading
import time
from typing import Any, Callable, Dict, Optional

from tools.execution_target import LOCAL_ROUTABLE_TOOLSETS, normalize_routed_toolsets


class _ToolRequestEntry:
    """One pending local-executor tool request."""

    __slots__ = ("event", "request", "result", "created_at")

    def __init__(self, request: dict[str, Any]):
        self.event = threading.Event()
        self.request = request
        self.result: Optional[str] = None
        self.created_at = time.time()


_lock = threading.Lock()
_tool_queues: dict[str, list[_ToolRequestEntry]] = {}
_tool_notify: dict[str, Callable[[dict[str, Any]], None]] = {}
_seen_results: dict[str, set[str]] = {}
_attached_clients: dict[str, str] = {}

_split_runtime_config: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "split_runtime_config",
    default=None,
)


def set_current_split_runtime(config: Optional[dict[str, Any]]) -> contextvars.Token:
    """Bind split-runtime config to the current agent/thread context."""

    if not config or not config.get("enabled"):
        normalized = None
    else:
        normalized = dict(config)
        normalized["routed_toolsets"] = normalize_routed_toolsets(
            normalized.get("routed_toolsets", LOCAL_ROUTABLE_TOOLSETS)
        )
        try:
            timeout = float(normalized.get("request_timeout_seconds", 300.0))
        except (TypeError, ValueError):
            timeout = 300.0
        normalized["request_timeout_seconds"] = max(0.1, timeout)
    return _split_runtime_config.set(normalized)


def reset_current_split_runtime(token: contextvars.Token) -> None:
    """Restore the previous split-runtime context."""

    _split_runtime_config.reset(token)


def get_current_split_runtime() -> Optional[dict[str, Any]]:
    """Return active split-runtime config for this context, if enabled."""

    return _split_runtime_config.get()


def register_tool_notify(session_key: str, cb: Callable[[dict[str, Any]], None], client_token: str) -> bool:
    """Attach one local executor client for a run/session.

    Returns ``False`` if another executor is already attached. A run can have at
    most one local executor to avoid split-brain writes/results.
    """

    if not session_key:
        return False
    token = client_token or ""
    with _lock:
        existing = _attached_clients.get(session_key)
        if existing is not None:
            return False
        _attached_clients[session_key] = token
        _tool_notify[session_key] = cb
        _seen_results.setdefault(session_key, set())
        return True


def unregister_tool_notify(session_key: str, client_token: Optional[str] = None) -> bool:
    """Detach the local executor and release all blocked tool workers.

    When ``client_token`` is provided, only the matching attached executor can
    detach. This prevents a stale SSE finally block from unregistering a newer
    executor after reconnect.
    """

    with _lock:
        existing = _attached_clients.get(session_key)
        if client_token is not None and existing not in {None, ""} and existing != client_token:
            return False
        _tool_notify.pop(session_key, None)
        _attached_clients.pop(session_key, None)
        _seen_results.pop(session_key, None)
        entries = _tool_queues.pop(session_key, [])
    for entry in entries:
        entry.event.set()
    return True


def tool_result_authorized(session_key: str, client_token: Optional[str]) -> bool:
    """Return True when a tool-result POST may answer this run's executor.

    Empty executor tokens mean the bearer key is the whole trust boundary.
    Non-empty tokens add a per-attachment guard for clients that want one.
    """

    with _lock:
        existing = _attached_clients.get(session_key)
    return not existing or existing == (client_token or "")


def has_attached_client(session_key: str) -> bool:
    """Return True when a local executor is attached for ``session_key``."""

    with _lock:
        return session_key in _attached_clients and session_key in _tool_notify


def submit_tool_request(session_key: str, request: dict[str, Any]) -> Optional[_ToolRequestEntry]:
    """Queue a tool request and notify the attached executor.

    Returns the pending entry, or ``None`` when no executor is attached or the
    notify callback failed synchronously.
    """

    if not session_key:
        return None
    entry = _ToolRequestEntry(dict(request or {}))
    with _lock:
        cb = _tool_notify.get(session_key)
        if cb is None or session_key not in _attached_clients:
            return None
        _tool_queues.setdefault(session_key, []).append(entry)
    try:
        cb(entry.request)
    except Exception:
        cancel_tool_request(session_key, str(entry.request.get("tool_call_id") or ""))
        return None
    return entry


def cancel_tool_request(session_key: str, tool_call_id: str) -> bool:
    """Remove a still-pending request after timeout or notify failure."""

    if not session_key or not tool_call_id:
        return False
    with _lock:
        queue = _tool_queues.get(session_key)
        if not queue:
            return False
        for index, entry in enumerate(list(queue)):
            if str(entry.request.get("tool_call_id") or "") == tool_call_id:
                queue.pop(index)
                if not queue:
                    _tool_queues.pop(session_key, None)
                entry.event.set()
                return True
    return False


def resolve_tool_result(session_key: str, tool_call_id: str, result: Any) -> str:
    """Resolve one pending local-executor request.

    Returns ``resolved``, ``duplicate``, or ``unknown``.
    """

    if not session_key or not tool_call_id:
        return "unknown"
    with _lock:
        seen = _seen_results.setdefault(session_key, set())
        if tool_call_id in seen:
            return "duplicate"
        queue = _tool_queues.get(session_key)
        if not queue:
            return "unknown"
        target: Optional[_ToolRequestEntry] = None
        for index, entry in enumerate(list(queue)):
            if str(entry.request.get("tool_call_id") or "") == tool_call_id:
                target = entry
                queue.pop(index)
                break
        if target is None:
            return "unknown"
        if not queue:
            _tool_queues.pop(session_key, None)
        seen.add(tool_call_id)

    if isinstance(result, str):
        target.result = result
    else:
        import json

        target.result = json.dumps(result, ensure_ascii=False)
    target.event.set()
    return "resolved"


def clear_tool_channel_state(session_key: Optional[str] = None) -> None:
    """Test/helper cleanup for one session or all split-runtime state."""

    with _lock:
        if session_key is None:
            entries = [entry for queue in _tool_queues.values() for entry in queue]
            _tool_queues.clear()
            _tool_notify.clear()
            _seen_results.clear()
            _attached_clients.clear()
        else:
            entries = _tool_queues.pop(session_key, [])
            _tool_notify.pop(session_key, None)
            _seen_results.pop(session_key, None)
            _attached_clients.pop(session_key, None)
    for entry in entries:
        entry.event.set()
