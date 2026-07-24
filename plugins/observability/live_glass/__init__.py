"""live_glass — plugin-first event bus for computer-use live glass.

The plugin is intentionally transport-agnostic. It observes existing
PluginManager hooks and publishes three event types for downstream dashboard or
gateway renderers:

* ``frame`` — a screenshot already returned by ``computer_use``.
* ``log`` — tool/action lifecycle metadata.
* ``approval_request`` — an observer-only approval prompt notification.

It does not capture the desktop on its own and it does not approve or deny
requests. Consumers subscribe to the in-process event bus or read the bounded
history and decide how to render events for their surface.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

EVENT_TYPES = frozenset({"frame", "log", "approval_request"})
_DEFAULT_MAX_EVENTS = 500
_DATA_IMAGE_RE = re.compile(r"^data:(image/[^;,]+);base64,")

EventCallback = Callable[[dict[str, Any]], None]

_LOCK = threading.RLock()
_SEQUENCE = 0
_EVENTS: deque[dict[str, Any]] = deque(maxlen=_DEFAULT_MAX_EVENTS)
_SUBSCRIBERS: dict[str, tuple[EventCallback, Optional[frozenset[str]]]] = {}


def _max_events() -> int:
    raw = os.environ.get("HERMES_LIVE_GLASS_MAX_EVENTS", "").strip()
    if not raw:
        return _DEFAULT_MAX_EVENTS
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid HERMES_LIVE_GLASS_MAX_EVENTS=%r; using %d", raw, _DEFAULT_MAX_EVENTS)
        return _DEFAULT_MAX_EVENTS
    return max(1, min(value, 10_000))


def _ensure_capacity() -> None:
    global _EVENTS
    maxlen = _max_events()
    if _EVENTS.maxlen == maxlen:
        return
    _EVENTS = deque(_EVENTS, maxlen=maxlen)


def _jsonable(value: Any) -> Any:
    """Return a deep-copied JSON-like representation for event payloads."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _context_from_kwargs(kwargs: dict[str, Any]) -> dict[str, str]:
    return {
        "session_id": str(kwargs.get("session_id") or kwargs.get("session_key") or ""),
        "task_id": str(kwargs.get("task_id") or ""),
        "tool_call_id": str(kwargs.get("tool_call_id") or ""),
        "turn_id": str(kwargs.get("turn_id") or ""),
        "api_request_id": str(kwargs.get("api_request_id") or ""),
    }


def publish(event_type: str, payload: dict[str, Any], **context: Any) -> dict[str, Any]:
    """Publish a live-glass event and return the stored event dict.

    ``event_type`` must be one of ``frame``, ``log``, or
    ``approval_request``. Subscriber exceptions are caught and logged so event
    rendering cannot break the agent path being observed.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown live-glass event type: {event_type}")

    global _SEQUENCE
    callbacks: list[EventCallback] = []
    with _LOCK:
        _ensure_capacity()
        _SEQUENCE += 1
        safe_context = _context_from_kwargs(context)
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "sequence": _SEQUENCE,
            "timestamp": time.time(),
            "payload": _jsonable(payload),
            **safe_context,
        }
        _EVENTS.append(event)
        event_for_callbacks = copy.deepcopy(event)
        callbacks = [
            callback
            for callback, event_types in _SUBSCRIBERS.values()
            if event_types is None or event_type in event_types
        ]

    for callback in callbacks:
        try:
            callback(copy.deepcopy(event_for_callbacks))
        except Exception:
            logger.debug("live-glass subscriber failed", exc_info=True)
    return event


def subscribe(
    callback: EventCallback,
    *,
    event_types: Optional[Iterable[str]] = None,
    replay: bool = False,
) -> Callable[[], None]:
    """Subscribe to live-glass events and return an unsubscribe function."""
    filter_set: Optional[frozenset[str]] = None
    if event_types is not None:
        filter_set = frozenset(event_types)
        unknown = filter_set - EVENT_TYPES
        if unknown:
            raise ValueError(f"unknown live-glass event type(s): {', '.join(sorted(unknown))}")

    token = str(uuid.uuid4())
    replay_events: list[dict[str, Any]] = []
    with _LOCK:
        _SUBSCRIBERS[token] = (callback, filter_set)
        if replay:
            replay_events = [
                copy.deepcopy(event)
                for event in _EVENTS
                if filter_set is None or event["type"] in filter_set
            ]

    for event in replay_events:
        try:
            callback(event)
        except Exception:
            logger.debug("live-glass replay subscriber failed", exc_info=True)

    def _unsubscribe() -> None:
        with _LOCK:
            _SUBSCRIBERS.pop(token, None)

    return _unsubscribe


def get_events(
    *,
    event_type: str | None = None,
    since_sequence: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return bounded event history, optionally filtered by type/sequence."""
    if event_type is not None and event_type not in EVENT_TYPES:
        raise ValueError(f"unknown live-glass event type: {event_type}")
    with _LOCK:
        events = [
            copy.deepcopy(event)
            for event in _EVENTS
            if (event_type is None or event["type"] == event_type)
            and (since_sequence is None or event["sequence"] > since_sequence)
        ]
    if limit is not None:
        if limit <= 0:
            return []
        events = events[-limit:]
    return events


def reset_event_bus_for_tests() -> None:
    """Clear in-memory bus state. Intended for tests only."""
    global _SEQUENCE, _EVENTS
    with _LOCK:
        _SEQUENCE = 0
        _EVENTS = deque(maxlen=_max_events())
        _SUBSCRIBERS.clear()


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _extract_image_url(result: Any) -> tuple[str, str] | tuple[None, None]:
    data = _maybe_json(result)
    if not isinstance(data, dict):
        return None, None
    for part in data.get("content") or []:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        image = part.get("image_url")
        if not isinstance(image, dict):
            continue
        url = image.get("url")
        if not isinstance(url, str):
            continue
        match = _DATA_IMAGE_RE.match(url)
        if match:
            return url, match.group(1)
    return None, None


def _extract_frame_payload(result: Any) -> dict[str, Any] | None:
    data = _maybe_json(result)
    if not isinstance(data, dict):
        return None
    image_url, mime_type = _extract_image_url(data)
    if not image_url or not mime_type:
        return None
    raw_meta = data.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    return {
        "image_url": image_url,
        "mime_type": mime_type,
        "mode": meta.get("mode"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "summary": str(data.get("text_summary") or ""),
        "source": "computer_use",
    }


def on_post_tool_call(**kwargs: Any) -> None:
    """Observe completed tool calls and publish log/frame events."""
    tool_name = str(kwargs.get("tool_name") or "")
    payload = {
        "tool_name": tool_name,
        "args": _jsonable(kwargs.get("args") or {}),
        "status": str(kwargs.get("status") or ""),
        "duration_ms": int(kwargs.get("duration_ms") or 0),
        "error_type": str(kwargs.get("error_type") or ""),
        "error_message": str(kwargs.get("error_message") or ""),
        "source": "post_tool_call",
    }
    publish("log", payload, **kwargs)

    if tool_name != "computer_use":
        return
    frame_payload = _extract_frame_payload(kwargs.get("result"))
    if frame_payload is None:
        return
    publish("frame", frame_payload, **kwargs)


def on_pre_approval_request(**kwargs: Any) -> None:
    """Observe an approval prompt without becoming an approval authority."""
    payload = {
        "command": str(kwargs.get("command") or ""),
        "description": str(kwargs.get("description") or ""),
        "pattern_key": str(kwargs.get("pattern_key") or ""),
        "pattern_keys": _jsonable(kwargs.get("pattern_keys") or []),
        "surface": str(kwargs.get("surface") or ""),
        "source": "approval_hook",
    }
    publish("approval_request", payload, **kwargs)


def on_post_approval_response(**kwargs: Any) -> None:
    """Log approval responses as lifecycle events; decisions remain external."""
    payload = {
        "tool_name": "approval",
        "args": {
            "command": str(kwargs.get("command") or ""),
            "surface": str(kwargs.get("surface") or ""),
            "choice": str(kwargs.get("choice") or ""),
        },
        "status": "ok" if str(kwargs.get("choice") or "") != "timeout" else "error",
        "duration_ms": 0,
        "error_type": "timeout" if str(kwargs.get("choice") or "") == "timeout" else "",
        "error_message": "" if str(kwargs.get("choice") or "") != "timeout" else "approval timed out",
        "source": "post_approval_response",
    }
    publish("log", payload, **kwargs)


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_approval_request", on_pre_approval_request)
    ctx.register_hook("post_approval_response", on_post_approval_response)

    # Bridge computer_use approval into PluginManager hooks so the
    # live-glass event bus (and any other observer) sees approval events
    # from computer_use actions.
    try:
        from plugins.observability.live_glass.approval_bridge import (
            register_approval_bridge,
        )
        register_approval_bridge(ctx)
    except Exception:
        logger.debug("live-glass: approval bridge setup failed", exc_info=True)
