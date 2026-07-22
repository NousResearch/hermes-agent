"""Minimal telemetry helpers for request/trace IDs and lifecycle events."""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_TRACE_ID: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_SESSION_ID: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


def _now_local() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _now_utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def generate_request_id(session_id: Optional[str] = None) -> str:
    unique = uuid.uuid4().hex[:8]
    return f"req_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{unique}"


def generate_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def unavailable_request_id() -> str:
    return "unavailable"


def unavailable_trace_id() -> str:
    return "unavailable"


class RequestContext:
    @staticmethod
    def set(request_id: Optional[str], trace_id: Optional[str], session_id: Optional[str]) -> None:
        _REQUEST_ID.set(request_id)
        _TRACE_ID.set(trace_id)
        _SESSION_ID.set(session_id)

    @staticmethod
    def clear() -> None:
        _REQUEST_ID.set(None)
        _TRACE_ID.set(None)
        _SESSION_ID.set(None)

    @staticmethod
    def get_request_id() -> Optional[str]:
        return _REQUEST_ID.get(None)

    @staticmethod
    def get_trace_id() -> Optional[str]:
        return _TRACE_ID.get(None)

    @staticmethod
    def get_session_id() -> Optional[str]:
        return _SESSION_ID.get(None)


def emit_event(
    *,
    component: str,
    event: str,
    status: str,
    duration_ms: Optional[float] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_count: Optional[int] = None,
    retry_count: Optional[int] = None,
    response_length: Optional[int] = None,
    estimated_context_tokens: Optional[int] = None,
    estimated_response_tokens: Optional[int] = None,
    budget_remaining: Optional[int] = None,
    error_category: Optional[str] = None,
    error_code: Optional[str] = None,
    non_secret_message: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    payload = {
        "timestamp_utc": _now_utc(),
        "timestamp_local": _now_local(),
        "request_id": RequestContext.get_request_id() or "unavailable",
        "trace_id": RequestContext.get_trace_id() or "unavailable",
        "session_id": RequestContext.get_session_id() or "unavailable",
        "component": component,
        "event": event,
        "status": status,
        "duration_ms": duration_ms,
        "provider": provider,
        "model": model,
        "tool_name": tool_name,
        "tool_count": tool_count,
        "retry_count": retry_count,
        "response_length": response_length,
        "estimated_context_tokens": estimated_context_tokens,
        "estimated_response_tokens": estimated_response_tokens,
        "budget_remaining": budget_remaining,
        "error_category": error_category,
        "error_code": error_code,
        "non_secret_message": non_secret_message,
        "source": source,
    }
    logger.info("telemetry_event %s", payload)


def classify_send_error(result: object) -> str:
    if result is None:
        return "result_unavailable"
    success = getattr(result, "success", None)
    if success is True:
        return "none"
    if success is False:
        error = getattr(result, "error", None)
        if error:
            return "send_failed"
        return "send_failed_unknown"
    return "unknown_result_state"


def non_safe_send_message(result: object) -> str:
    if result is None:
        return "result_unavailable"
    success = getattr(result, "success", None)
    if success is False:
        msg = getattr(result, "message_id", None)
        txt = ""
        try:
            txt = str(getattr(result, "text", "") or "")
        except Exception:
            txt = ""
        txt = txt.replace("\n", " ")[:120]
        return f"failed message_id={msg or 'unavailable'} preview={txt or 'unavailable'}"
    return "ok"
