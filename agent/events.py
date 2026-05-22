"""Structured event sink helpers for agent frontends.

The default sink is intentionally a no-op. Frontends that need live run
events can provide an ``AgentEventSink`` implementation without changing
normal transcript or callback behavior.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Mapping

logger = logging.getLogger(__name__)

MAX_PREVIEW_CHARS = 1000
MAX_STRING_CHARS = 2000
MAX_COLLECTION_ITEMS = 50
MAX_EVENT_JSON_CHARS = 6000

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|refresh[_-]?token|auth(?:orization)?|"
    r"bearer|cookie|password|passwd|secret|private[_-]?key|token)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|"
    r"passwd|secret|authorization)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_OPENAI_STYLE_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


class AgentEventSink:
    """Base class for consumers of structured agent events."""

    def emit(self, event: dict[str, Any]) -> None:
        """Receive one structured event."""
        return None


class NullAgentEventSink(AgentEventSink):
    """Default event sink used when no frontend is listening."""

    def emit(self, event: dict[str, Any]) -> None:
        return None


def emit_run_status(agent: Any, status: str = "thinking", message: str = "Agent is thinking.") -> None:
    emit_agent_event(
        agent,
        {
            "type": "run.status",
            "status": status,
            "message": message,
            "updatedAt": utc_now_iso(),
        },
    )


def emit_run_done(agent: Any) -> None:
    emit_agent_event(agent, {"type": "run.done", "updatedAt": utc_now_iso()})


def emit_run_cancelled(agent: Any, message: str = "Agent run cancelled.") -> None:
    emit_agent_event(
        agent,
        {
            "type": "run.cancelled",
            "message": message,
            "updatedAt": utc_now_iso(),
        },
    )


def emit_run_error(agent: Any, error: Any) -> None:
    emit_agent_event(
        agent,
        {
            "type": "run.error",
            "message": safe_preview(error, limit=MAX_PREVIEW_CHARS) or "Agent run failed.",
            "debug": safe_debug(error),
            "updatedAt": utc_now_iso(),
        },
    )


def emit_assistant_message(agent: Any, content: Any) -> None:
    emit_agent_event(
        agent,
        {
            "type": "assistant.message",
            "content": redact_string(str(content or "")),
            "createdAt": utc_now_iso(),
        },
    )


def emit_tool_started(agent: Any, tool_call_id: str | None, name: str, args: Any) -> None:
    emit_agent_event(
        agent,
        {
            "type": "tool.call.started",
            "toolCallId": tool_call_id or "",
            "name": name,
            "args": sanitize_event_value(args),
            "category": categorize_tool(name),
            "createdAt": utc_now_iso(),
        },
    )


def emit_tool_completed(
    agent: Any,
    tool_call_id: str | None,
    name: str,
    args: Any,
    result: Any,
    duration_s: float | None = None,
) -> None:
    payload = _tool_result_payload(result)
    emit_agent_event(
        agent,
        {
            "type": "tool.call.completed",
            "toolCallId": tool_call_id or "",
            "name": name,
            "ok": True,
            "args": sanitize_event_value(args),
            "category": categorize_tool(name),
            "durationMs": duration_ms(duration_s),
            "completedAt": utc_now_iso(),
            **payload,
        },
    )


def emit_tool_failed(
    agent: Any,
    tool_call_id: str | None,
    name: str,
    args: Any,
    error: Any,
    duration_s: float | None = None,
) -> None:
    payload = _tool_result_payload(error)
    preview = payload.get("resultPreview") or safe_preview(error, limit=MAX_PREVIEW_CHARS)
    emit_agent_event(
        agent,
        {
            "type": "tool.call.failed",
            "toolCallId": tool_call_id or "",
            "name": name,
            "ok": False,
            "args": sanitize_event_value(args),
            "category": categorize_tool(name),
            "durationMs": duration_ms(duration_s),
            "error": preview or "Tool call failed.",
            "resultPreview": preview or "",
            "resultTruncated": payload.get("resultTruncated", False),
            "completedAt": utc_now_iso(),
        },
    )


def emit_agent_event(agent: Any, event: Mapping[str, Any]) -> None:
    sink = getattr(agent, "event_sink", None)
    if sink is None:
        return
    try:
        sink.emit(dict(event))
    except Exception:
        logger.debug("agent event sink failed for %s", event.get("type"), exc_info=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def duration_ms(duration_s: float | None) -> int:
    if duration_s is None:
        return 0
    try:
        return max(0, int(float(duration_s) * 1000))
    except (TypeError, ValueError):
        return 0


def categorize_tool(name: str) -> str:
    lowered = (name or "").lower()
    if lowered in {"terminal", "execute_code"}:
        return "command"
    if lowered in {"write_file", "read_file", "patch", "search_files", "list_files"}:
        return "file"
    if "browser" in lowered:
        return "browser"
    if "web" in lowered or lowered in {"crawl", "extract"}:
        return "web"
    if "memory" in lowered or lowered == "session_search":
        return "memory"
    if "delegate" in lowered:
        return "agent"
    return "tool"


def sanitize_event_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return truncate_text(redact_string(value), MAX_STRING_CHARS)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_COLLECTION_ITEMS:
                out["..."] = f"{len(value) - MAX_COLLECTION_ITEMS} more"
                break
            key_text = str(key)
            if _is_secret_key(key_text):
                out[key_text] = "[redacted]"
            else:
                out[key_text] = sanitize_event_value(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [sanitize_event_value(item, depth=depth + 1) for item in items[:MAX_COLLECTION_ITEMS]]
        if len(items) > MAX_COLLECTION_ITEMS:
            out.append(f"[{len(items) - MAX_COLLECTION_ITEMS} more]")
        return out
    return truncate_text(redact_string(str(value)), MAX_STRING_CHARS)


def safe_preview(value: Any, *, limit: int = MAX_PREVIEW_CHARS) -> str:
    sanitized = sanitize_event_value(value)
    if isinstance(sanitized, str):
        text = sanitized
    else:
        try:
            text = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(sanitized)
    return truncate_text(text, limit)


def safe_debug(error: Any) -> dict[str, Any]:
    if isinstance(error, BaseException):
        return {
            "type": type(error).__name__,
            "message": safe_preview(str(error), limit=MAX_PREVIEW_CHARS),
        }
    return {"value": sanitize_event_value(error)}


def redact_string(text: str) -> str:
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    redacted = _BEARER_RE.sub("Bearer [redacted]", redacted)
    redacted = _OPENAI_STYLE_TOKEN_RE.sub("[redacted-token]", redacted)
    return redacted


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _tool_result_payload(result: Any) -> dict[str, Any]:
    preview = safe_preview(result, limit=MAX_PREVIEW_CHARS)
    payload: dict[str, Any] = {
        "resultPreview": preview,
        "resultTruncated": len(preview) >= MAX_PREVIEW_CHARS and preview.endswith("..."),
    }
    structured = _safe_structured_result(result)
    if structured is not None:
        try:
            encoded = json.dumps(structured, ensure_ascii=False, sort_keys=True)
        except TypeError:
            encoded = ""
        if encoded and len(encoded) <= MAX_EVENT_JSON_CHARS:
            payload["resultJson"] = structured
    return payload


def _safe_structured_result(result: Any) -> Any | None:
    if isinstance(result, (Mapping, list, tuple)):
        return sanitize_event_value(result)
    if isinstance(result, str):
        stripped = result.strip()
        if not stripped or stripped[0] not in "[{":
            return None
        try:
            return sanitize_event_value(json.loads(stripped))
        except Exception:
            return None
    return None


def _is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY_RE.search(key))


__all__ = [
    "AgentEventSink",
    "NullAgentEventSink",
    "emit_agent_event",
    "emit_assistant_message",
    "emit_run_cancelled",
    "emit_run_done",
    "emit_run_error",
    "emit_run_status",
    "emit_tool_completed",
    "emit_tool_failed",
    "emit_tool_started",
    "safe_preview",
    "sanitize_event_value",
]
