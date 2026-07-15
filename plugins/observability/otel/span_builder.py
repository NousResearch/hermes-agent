"""Privacy-safe OpenTelemetry span lifecycle management for Hermes hooks."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")


def _text(value: Any) -> str:
    return str(value or "")


def _safe_label(value: Any, default: str = "unknown") -> str:
    text = _text(value).strip()
    return text if _SAFE_LABEL.fullmatch(text) else default


def _stable_digest(value: Any) -> str:
    text = _text(value)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16] if text else ""


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _duration_ms(kwargs: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float(kwargs.get(key))
        if value is not None:
            return value * 1000 if key == "api_duration" else value
    return None


def _ids(kwargs: dict[str, Any]) -> dict[str, str]:
    attributes = {}
    for key in (
        "session_id",
        "task_id",
        "turn_id",
        "api_request_id",
        "tool_call_id",
        "parent_session_id",
        "parent_turn_id",
        "parent_subagent_id",
        "child_session_id",
        "child_subagent_id",
    ):
        value = _text(kwargs.get(key))
        if value:
            attributes[f"hermes.{key}"] = value
    schema = _text(kwargs.get("telemetry_schema_version"))
    if schema:
        attributes["hermes.telemetry_schema_version"] = schema
    return attributes


@dataclass
class _SpanEntry:
    span: Any
    started_ns: int
    session_id: str = ""


class SpanBuilder:
    """Build and correlate Hermes spans without exporting sensitive payloads."""

    def __init__(self, tracer: Any) -> None:
        self.tracer = tracer
        self._lock = threading.RLock()
        self._sessions: dict[str, _SpanEntry] = {}
        self._turns: dict[str, _SpanEntry] = {}
        self._requests: dict[str, _SpanEntry] = {}
        self._tools: dict[str, _SpanEntry] = {}
        self._subagents: dict[str, _SpanEntry] = {}
        self._approvals: dict[str, list[_SpanEntry]] = defaultdict(list)

    @staticmethod
    def _now_ns() -> int:
        return time.time_ns()

    def _parent_context(self, span: Any) -> Any:
        if span is None:
            return None
        from opentelemetry import trace

        return trace.set_span_in_context(span)

    def _start(
        self,
        name: str,
        attributes: dict[str, Any],
        *,
        parent: _SpanEntry | None = None,
        started_at: Any = None,
    ) -> _SpanEntry:
        start_ns = self._now_ns()
        started = _float(started_at)
        if started is not None:
            start_ns = int(started * 1_000_000_000)
        span = self.tracer.start_span(
            name,
            context=self._parent_context(parent.span if parent else None),
            attributes={key: value for key, value in attributes.items() if value is not None},
            start_time=start_ns,
        )
        return _SpanEntry(
            span=span,
            started_ns=start_ns,
            session_id=_text(attributes.get("hermes.session_id")),
        )

    def _end(
        self,
        entry: _SpanEntry | None,
        attributes: dict[str, Any] | None = None,
        *,
        error: bool = False,
    ) -> None:
        if entry is None:
            return
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    entry.span.set_attribute(key, value)
        if error:
            from opentelemetry.trace import Status, StatusCode

            entry.span.set_status(Status(StatusCode.ERROR))
        entry.span.end()

    def _session(self, session_id: str) -> _SpanEntry | None:
        return self._sessions.get(session_id) if session_id else None

    def _turn(self, turn_id: str) -> _SpanEntry | None:
        return self._turns.get(turn_id) if turn_id else None

    def start_session(self, kwargs: dict[str, Any]) -> None:
        session_id = _text(kwargs.get("session_id"))
        if not session_id:
            return
        with self._lock:
            if session_id in self._sessions:
                return
            attributes = _ids(kwargs)
            attributes.update(
                {
                    "hermes.model": _safe_label(kwargs.get("model")),
                    "hermes.platform": _safe_label(kwargs.get("platform")),
                }
            )
            self._sessions[session_id] = self._start("hermes.session", attributes)

    def mark_session_end(self, kwargs: dict[str, Any]) -> None:
        session_id = _text(kwargs.get("session_id"))
        with self._lock:
            entry = self._sessions.pop(session_id, None)
            attributes: dict[str, Any] = {}
            for key in ("completed", "interrupted"):
                value = kwargs.get(key)
                if isinstance(value, bool):
                    attributes[f"hermes.session.{key}"] = value
            self._end(entry, attributes, error=bool(kwargs.get("interrupted")))

    def finalize_session(self, kwargs: dict[str, Any]) -> None:
        session_id = _text(kwargs.get("session_id") or kwargs.get("old_session_id"))
        with self._lock:
            self._end(self._sessions.pop(session_id, None), {"hermes.session.finalized": True})
            self._finish_session_children(session_id)

    def _finish_session_children(self, session_id: str) -> None:
        for mapping in (self._turns, self._requests, self._tools, self._subagents):
            stale = [
                key for key, entry in mapping.items() if entry.session_id == session_id
            ]
            for key in stale:
                self._end(mapping.pop(key, None), {"hermes.incomplete": True}, error=True)

    def start_turn(self, kwargs: dict[str, Any]) -> None:
        turn_id = _text(kwargs.get("turn_id"))
        if not turn_id:
            return
        with self._lock:
            if turn_id in self._turns:
                return
            session_id = _text(kwargs.get("session_id"))
            if session_id not in self._sessions:
                self.start_session(kwargs)
            attributes = _ids(kwargs)
            attributes.update(
                {
                    "hermes.model": _safe_label(kwargs.get("model")),
                    "hermes.platform": _safe_label(kwargs.get("platform")),
                    "hermes.turn.is_first": bool(kwargs.get("is_first_turn", False)),
                    "hermes.turn.user_message_length": len(_text(kwargs.get("user_message"))),
                    "hermes.turn.history_count": len(kwargs.get("conversation_history") or []),
                }
            )
            self._turns[turn_id] = self._start(
                "hermes.turn", attributes, parent=self._session(session_id)
            )

    def end_turn(self, kwargs: dict[str, Any]) -> None:
        turn_id = _text(kwargs.get("turn_id"))
        with self._lock:
            self._end(
                self._turns.pop(turn_id, None),
                {
                    "hermes.turn.assistant_response_length": len(
                        _text(kwargs.get("assistant_response"))
                    ),
                    "hermes.turn.history_count": len(kwargs.get("conversation_history") or []),
                },
            )

    def start_llm_request(self, kwargs: dict[str, Any]) -> None:
        request_id = _text(kwargs.get("api_request_id"))
        if not request_id:
            return
        with self._lock:
            attributes = _ids(kwargs)
            attributes.update(
                {
                    "gen_ai.request.model": _safe_label(kwargs.get("model")),
                    "gen_ai.provider.name": _safe_label(kwargs.get("provider")),
                    "hermes.api_mode": _safe_label(kwargs.get("api_mode")),
                    "hermes.api_call_count": _int(kwargs.get("api_call_count")),
                    "hermes.request.message_count": _int(kwargs.get("message_count")),
                    "hermes.request.tool_count": _int(kwargs.get("tool_count")),
                    "hermes.request.approx_input_tokens": _int(kwargs.get("approx_input_tokens")),
                    "hermes.request.char_count": _int(kwargs.get("request_char_count")),
                }
            )
            parent = self._turn(_text(kwargs.get("turn_id"))) or self._session(
                _text(kwargs.get("session_id"))
            )
            self._requests[request_id] = self._start(
                "hermes.llm_request",
                attributes,
                parent=parent,
                started_at=kwargs.get("started_at"),
            )

    def end_llm_request(self, kwargs: dict[str, Any], *, error: bool = False) -> None:
        request_id = _text(kwargs.get("api_request_id"))
        usage = kwargs.get("usage") if isinstance(kwargs.get("usage"), dict) else {}
        attributes = {
            "hermes.api.duration_ms": _duration_ms(kwargs, "api_duration"),
            "gen_ai.response.finish_reasons": _safe_label(kwargs.get("finish_reason"), ""),
            "gen_ai.response.model": _safe_label(kwargs.get("response_model"), ""),
            "gen_ai.usage.input_tokens": _int(usage.get("input_tokens")),
            "gen_ai.usage.output_tokens": _int(usage.get("output_tokens")),
            "hermes.usage.cache_read_tokens": _int(usage.get("cache_read_tokens")),
            "hermes.usage.cache_write_tokens": _int(usage.get("cache_write_tokens")),
            "hermes.response.content_length": _int(kwargs.get("assistant_content_chars")),
            "hermes.response.tool_call_count": _int(kwargs.get("assistant_tool_call_count")),
            "hermes.error.type": _safe_label(
                (kwargs.get("error") or {}).get("type") if isinstance(kwargs.get("error"), dict) else None,
                "unknown" if error else "",
            ),
            "hermes.error.reason": _safe_label(kwargs.get("reason"), ""),
            "hermes.error.status_code": _int(kwargs.get("status_code")),
            "hermes.error.retryable": kwargs.get("retryable") if isinstance(kwargs.get("retryable"), bool) else None,
        }
        with self._lock:
            self._end(self._requests.pop(request_id, None), attributes, error=error)

    def start_tool(self, kwargs: dict[str, Any]) -> None:
        call_id = _text(kwargs.get("tool_call_id"))
        if not call_id:
            return
        with self._lock:
            tool_name = _safe_label(kwargs.get("tool_name"))
            attributes = _ids(kwargs)
            attributes.update(
                {
                    "hermes.tool.name": tool_name,
                    "hermes.tool.arg_count": len(kwargs.get("args") or {})
                    if isinstance(kwargs.get("args"), dict)
                    else 0,
                }
            )
            parent = self._turn(_text(kwargs.get("turn_id"))) or self._session(
                _text(kwargs.get("session_id"))
            )
            self._tools[call_id] = self._start(
                f"hermes.tool.{tool_name}", attributes, parent=parent
            )

    def end_tool(self, kwargs: dict[str, Any]) -> None:
        call_id = _text(kwargs.get("tool_call_id"))
        status = _safe_label(kwargs.get("status"), "unknown")
        attributes = {
            "hermes.tool.status": status,
            "hermes.tool.duration_ms": _duration_ms(kwargs, "duration_ms"),
            "hermes.tool.result_length": len(_text(kwargs.get("result"))),
            "hermes.tool.error_type": _safe_label(kwargs.get("error_type"), ""),
        }
        with self._lock:
            self._end(self._tools.pop(call_id, None), attributes, error=status != "ok")

    def start_subagent(self, kwargs: dict[str, Any]) -> None:
        child_session_id = _text(kwargs.get("child_session_id"))
        child_id = _text(kwargs.get("child_subagent_id")) or child_session_id
        if not child_id:
            return
        with self._lock:
            attributes = _ids(kwargs)
            goal = _text(kwargs.get("child_goal"))
            attributes.update(
                {
                    "hermes.subagent.role": _safe_label(kwargs.get("child_role"), "custom"),
                    "hermes.subagent.goal_length": len(goal),
                    "hermes.subagent.goal_id": _stable_digest(goal),
                }
            )
            parent = self._turn(_text(kwargs.get("parent_turn_id"))) or self._session(
                _text(kwargs.get("parent_session_id"))
            )
            self._subagents[child_id] = self._start(
                "hermes.subagent", attributes, parent=parent
            )

    def end_subagent(self, kwargs: dict[str, Any]) -> None:
        child_id = _text(kwargs.get("child_subagent_id")) or _text(kwargs.get("child_session_id"))
        status = _safe_label(kwargs.get("child_status"), "unknown")
        with self._lock:
            self._end(
                self._subagents.pop(child_id, None),
                {
                    "hermes.subagent.status": status,
                    "hermes.subagent.duration_ms": _duration_ms(kwargs, "duration_ms"),
                    "hermes.subagent.summary_length": len(_text(kwargs.get("child_summary"))),
                },
                error=status not in {"ok", "completed", "success"},
            )

    def start_approval(self, kwargs: dict[str, Any]) -> None:
        key = _text(kwargs.get("session_key") or kwargs.get("session_id")) or "global"
        with self._lock:
            command = _text(kwargs.get("command"))
            attributes = _ids(kwargs)
            attributes.update(
                {
                    "hermes.approval.surface": _safe_label(kwargs.get("surface")),
                    "hermes.approval.command_length": len(command),
                    "hermes.approval.command_id": _stable_digest(command),
                    "hermes.approval.description_length": len(_text(kwargs.get("description"))),
                    "hermes.approval.pattern_count": len(kwargs.get("pattern_keys") or []),
                }
            )
            parent = self._session(_text(kwargs.get("session_id") or kwargs.get("session_key")))
            self._approvals[key].append(self._start("hermes.approval", attributes, parent=parent))

    def end_approval(self, kwargs: dict[str, Any]) -> None:
        key = _text(kwargs.get("session_key") or kwargs.get("session_id")) or "global"
        choice = _safe_label(kwargs.get("choice"), "unknown")
        with self._lock:
            entries = self._approvals.get(key, [])
            entry = entries.pop(0) if entries else None
            if not entries:
                self._approvals.pop(key, None)
            self._end(
                entry,
                {"hermes.approval.decision": choice},
                error=choice in {"deny", "timeout", "smart_deny"},
            )
