"""In-process state for API-server split-runtime tool requests.

The API server owns the model/agent loop. A local executor client can attach to
``/v1/runs/{run_id}/events`` and receive ``tool.request`` events for a small
allowlisted tool surface. This module is the synchronous block/resume registry
between the tool worker thread and the HTTP/SSE handlers.
"""

from __future__ import annotations

import contextvars
import hmac
import json
import math
import threading
import time
import uuid
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
_attached_condition = threading.Condition(_lock)
_tool_queues: dict[str, list[_ToolRequestEntry]] = {}
_tool_notify: dict[str, Callable[[dict[str, Any]], None]] = {}
_seen_results: dict[str, set[str]] = {}
_attached_clients: dict[str, str] = {}
_closed_channels: set[str] = set()

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
        if not math.isfinite(timeout):
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
    with _attached_condition:
        if session_key in _closed_channels:
            return False
        existing = _attached_clients.get(session_key)
        if existing is not None:
            return False
        _attached_clients[session_key] = token
        _tool_notify[session_key] = cb
        _seen_results.setdefault(session_key, set())
        _attached_condition.notify_all()
        return True


def _constant_time_token_equal(left: str, right: str) -> bool:
    """Compare arbitrary Unicode executor tokens without timing leaks."""

    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def unregister_tool_notify(session_key: str, client_token: Optional[str] = None) -> bool:
    """Detach the local executor and release all blocked tool workers.

    When ``client_token`` is provided, only the matching attached executor can
    detach. This prevents a stale SSE finally block from unregistering a newer
    executor after reconnect.
    """

    with _attached_condition:
        existing = _attached_clients.get(session_key)
        if (
            client_token is not None
            and existing not in {None, ""}
            and not _constant_time_token_equal(existing or "", client_token)
        ):
            return False
        _tool_notify.pop(session_key, None)
        _attached_clients.pop(session_key, None)
        entries = _tool_queues.pop(session_key, [])
        _attached_condition.notify_all()
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
    return not existing or _constant_time_token_equal(existing, client_token or "")


def has_attached_client(session_key: str) -> bool:
    """Return True when a local executor is attached for ``session_key``."""

    with _lock:
        return session_key in _attached_clients and session_key in _tool_notify


def pending_tool_request_count(session_key: str) -> int:
    """Return the number of unresolved requests for an attached executor."""

    with _lock:
        return len(_tool_queues.get(session_key, ()))


def is_tool_request_pending(session_key: str, request_id: str) -> bool:
    """Return whether ``request_id`` is still pending for this run channel."""

    with _lock:
        return any(
            str(entry.request.get("request_id") or "") == request_id
            for entry in _tool_queues.get(session_key, ())
        )


def wait_for_attached_client(session_key: str, timeout: float) -> bool:
    """Wait for a local executor attachment without polling.

    This closes the HTTP ordering gap where a run can call a routed tool before
    the client receives its ``run_id`` and attaches the executor SSE stream.
    """

    if not session_key:
        return False
    try:
        wait_timeout = float(timeout)
    except (TypeError, ValueError):
        wait_timeout = 0.0
    if not math.isfinite(wait_timeout):
        wait_timeout = 0.0
    with _attached_condition:
        _attached_condition.wait_for(
            lambda: (
                session_key in _closed_channels
                or (session_key in _attached_clients and session_key in _tool_notify)
            ),
            timeout=max(0.0, wait_timeout),
        )
        return (
            session_key not in _closed_channels
            and session_key in _attached_clients
            and session_key in _tool_notify
        )


def submit_tool_request(session_key: str, request: dict[str, Any]) -> Optional[_ToolRequestEntry]:
    """Queue a tool request and notify the attached executor.

    Returns the pending entry, or ``None`` when no executor is attached or the
    notify callback failed synchronously. The broker always mints a unique wire
    ``request_id``; model-provided ``tool_call_id`` remains metadata only.
    """

    if not session_key:
        return None
    request_payload = dict(request or {})
    request_payload["request_id"] = f"toolreq_{uuid.uuid4().hex}"
    entry = _ToolRequestEntry(request_payload)
    with _lock:
        if session_key in _closed_channels:
            return None
        cb = _tool_notify.get(session_key)
        if cb is None or session_key not in _attached_clients:
            return None
        _tool_queues.setdefault(session_key, []).append(entry)
    try:
        cb(entry.request)
    except Exception:
        cancel_tool_request(session_key, str(entry.request["request_id"]))
        return None
    return entry


def cancel_tool_request(session_key: str, request_id: str) -> bool:
    """Remove a still-pending request after timeout or notify failure."""

    if not session_key or not request_id:
        return False
    with _lock:
        queue = _tool_queues.get(session_key)
        if not queue:
            return False
        for index, entry in enumerate(list(queue)):
            if str(entry.request.get("request_id") or "") == request_id:
                queue.pop(index)
                if not queue:
                    _tool_queues.pop(session_key, None)
                entry.event.set()
                return True
    return False


def resolve_tool_result(session_key: str, request_id: str, result: Any) -> str:
    """Resolve one pending local-executor request.

    Returns ``resolved``, ``duplicate``, or ``unknown``.
    """

    if not session_key or not request_id:
        return "unknown"
    with _lock:
        seen = _seen_results.setdefault(session_key, set())
        if request_id in seen:
            return "duplicate"
        queue = _tool_queues.get(session_key)
        if not queue:
            return "unknown"
        target: Optional[_ToolRequestEntry] = None
        for index, entry in enumerate(list(queue)):
            if str(entry.request.get("request_id") or "") == request_id:
                target = entry
                queue.pop(index)
                break
        if target is None:
            return "unknown"
        if not queue:
            _tool_queues.pop(session_key, None)
        seen.add(request_id)
        if isinstance(result, str):
            target.result = result
        else:
            target.result = json.dumps(result, ensure_ascii=False)
        target.event.set()
    return "resolved"


def clear_tool_channel_state(session_key: Optional[str] = None) -> None:
    """Test/helper cleanup for one session or all split-runtime state."""

    with _attached_condition:
        if session_key is None:
            entries = [entry for queue in _tool_queues.values() for entry in queue]
            _tool_queues.clear()
            _tool_notify.clear()
            _seen_results.clear()
            _attached_clients.clear()
            _closed_channels.clear()
        else:
            entries = _tool_queues.pop(session_key, [])
            _tool_notify.pop(session_key, None)
            _seen_results.pop(session_key, None)
            _attached_clients.pop(session_key, None)
            _closed_channels.discard(session_key)
        _attached_condition.notify_all()
    for entry in entries:
        entry.event.set()


def close_tool_channel(session_key: str) -> None:
    """Terminally close a run channel and wake every blocked tool worker."""

    if not session_key:
        return
    with _attached_condition:
        _closed_channels.add(session_key)
        _tool_notify.pop(session_key, None)
        _attached_clients.pop(session_key, None)
        _seen_results.pop(session_key, None)
        entries = _tool_queues.pop(session_key, [])
        _attached_condition.notify_all()
    for entry in entries:
        entry.event.set()
