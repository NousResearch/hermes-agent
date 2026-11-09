"""Emit a privacy-minimal audit event for serialized OpenAI-compatible requests."""

from __future__ import annotations

import json
import sys
import threading
from typing import Any
from urllib.parse import urlsplit

EVENT_PREFIX = "HERMES_OUTBOUND_REASONING "
_EVENT_LOCK = threading.Lock()

_REASONING_PATHS = (
    ("reasoning", "effort"),
    ("reasoning_effort",),
    ("output_config", "effort"),
    ("thinking", "effort"),
    ("thinking", "type"),
    ("thinking", "budget_tokens"),
    ("thinkingConfig", "thinkingLevel"),
    ("thinking_level",),
)


def _field(body: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = body
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


_MISSING = object()


def _request_details(request: Any) -> tuple[str | None, list[dict[str, Any]], str]:
    try:
        body = json.loads(request.content)
    except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, RuntimeError):
        return None, [], "unavailable"
    if not isinstance(body, dict):
        return None, [], "unavailable"

    model = body.get("model") if isinstance(body.get("model"), str) else None
    fields: list[dict[str, Any]] = []
    unsupported = False
    for path in _REASONING_PATHS:
        value = _field(body, path)
        if value is _MISSING:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            fields.append({"path": ".".join(path), "value": value})
        else:
            unsupported = True
    if fields:
        status = "explicit"
    elif unsupported:
        status = "unavailable"
    else:
        status = "not_present"
    return model, fields, status


def audit_outbound_reasoning_request(request: Any) -> None:
    """Log only model, endpoint host/path, and recognized reasoning fields for POST requests."""
    if getattr(request, "method", "").upper() != "POST":
        return

    model, fields, status = _request_details(request)
    try:
        url = urlsplit(str(request.url))
        host = url.hostname
        path = url.path
    except (AttributeError, TypeError, ValueError):
        host = None
        path = None

    event = {
        "model": model,
        "host": host,
        "path": path,
        "reasoning_fields": fields,
        "reasoning_status": status,
        "stage": "before_transport_send",
    }
    try:
        line = EVENT_PREFIX + json.dumps(event, separators=(",", ":"), ensure_ascii=True)
        with _EVENT_LOCK:
            print(line, file=sys.stderr, flush=True)
    except (OSError, TypeError, ValueError):
        # Auditing must never break the provider request.
        return


async def audit_outbound_reasoning_request_async(request: Any) -> None:
    """Async HTTPX event-hook adapter; request content is already buffered by HTTPX."""
    audit_outbound_reasoning_request(request)


__all__ = [
    "EVENT_PREFIX",
    "audit_outbound_reasoning_request",
    "audit_outbound_reasoning_request_async",
]
