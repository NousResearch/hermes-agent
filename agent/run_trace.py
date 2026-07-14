"""Metadata-only run trace persistence.

Run traces are deliberately smaller and safer than full trajectories: they
capture turn-level execution metadata (IDs, model/provider, status, API call
count, and tool names) without raw prompts, raw tool arguments, raw tool
outputs, or assistant text.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Iterable

from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "hermes_run_trace_v1"
_DEFAULT_TRACE_PATH = "run_traces/run_traces.jsonl"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_ALLOWED_STATUS = {"running", "completed", "interrupted", "failed", "partial"}
_ALLOWED_EXIT_REASONS = {
    "unknown",
    "interrupted_by_user",
    "budget_exhausted",
    "ollama_runtime_context_too_small",
    "interrupted_during_api_call",
    "all_retries_exhausted_no_response",
    "guardrail_halt",
    "partial_stream_recovery",
    "fallback_prior_turn_content",
    "empty_response_exhausted",
    "text_response",
    "error_near_max_iterations",
    "max_iterations_reached",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_text(value: Any, *, max_len: int = 240) -> str:
    """Return redacted bounded metadata text.

    This helper is intentionally used only for metadata fields, never raw prompt
    text, tool arguments, or tool outputs.  ``force=True`` means trace safety is
    not weakened when a user disables general log redaction.
    """

    if value is None:
        return ""
    try:
        text = str(value)
    except Exception:
        text = "<unserializable>"
    text = redact_sensitive_text(text, force=True)
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _safe_tool_name(value: Any) -> str:
    return _safe_text(value, max_len=128)


def _safe_identifier(value: Any, *, max_len: int = 160) -> str:
    """Return a stable fingerprint without persisting identifier text.

    Identifier-shaped strings are not inherently safe: caller-provided labels
    can be valid slugs while still containing customer or task details. Hash all
    identifiers so generated IDs remain correlatable without creating an
    allowlist bypass for free text.
    """

    text = _safe_text(value, max_len=max_len)
    if not text:
        return ""
    digest = sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"sha256:{digest}"


def _safe_status(value: Any) -> str:
    status = _safe_text(value, max_len=48).strip().lower()
    return status if status in _ALLOWED_STATUS else "partial"


def _safe_exit_reason(value: Any) -> str:
    """Return a controlled exit-reason code, never raw exception text."""

    text = _safe_text(value, max_len=240).strip()
    if not text:
        return "unknown"
    code = text.split("(", 1)[0].strip().lower()
    code = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in code)
    return code if code in _ALLOWED_EXIT_REASONS else "other"


def _cfg_get(config: Any, *keys: str, default: Any = None) -> Any:
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        logger.debug("run_trace config load failed", exc_info=True)
        return {}


def trace_enabled(*, config: dict[str, Any] | None = None) -> bool:
    """Return whether run trace persistence is enabled.

    Disabled by default.  The first PR is observe-only and opt-in through
    ``observability.run_trace_enabled``.
    """

    cfg = _load_config() if config is None else config
    raw = _cfg_get(cfg, "observability", "run_trace_enabled", default=False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in _TRUE_VALUES
    if isinstance(raw, (int, float)):
        return bool(raw)
    return False


def _trace_path(*, config: dict[str, Any] | None = None):
    cfg = _load_config() if config is None else config
    rel = _cfg_get(cfg, "observability", "run_trace_path", default=_DEFAULT_TRACE_PATH)
    if not isinstance(rel, str) or not rel.strip():
        rel = _DEFAULT_TRACE_PATH
    rel = rel.strip()
    home = get_hermes_home()
    path = home / rel
    try:
        path.resolve().relative_to(home.resolve())
    except Exception:
        # Keep the writer inside HERMES_HOME even if config is malicious or
        # accidentally absolute.  This is a safety fallback, not enforcement UX.
        path = home / _DEFAULT_TRACE_PATH
    return path


@dataclass
class RunTraceToolCall:
    """Safe metadata about one requested tool call."""

    name: str
    tool_call_id: str = ""
    status: str = "requested"
    duration_ms: int | None = None
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": _safe_tool_name(self.name),
            "tool_call_id": _safe_identifier(self.tool_call_id, max_len=160),
            "status": _safe_text(self.status, max_len=48),
            "duration_ms": self.duration_ms if isinstance(self.duration_ms, int) else None,
            "error_type": _safe_text(self.error_type, max_len=120),
            "error_message": _safe_exit_reason(self.error_message) if self.error_message else "",
        }


@dataclass
class RunTrace:
    """Metadata-only record for one agent turn."""

    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    session_id: str = ""
    turn_id: str = ""
    task_id: str = ""
    model: str = ""
    provider: str = ""
    source: str = ""
    status: str = "running"
    exit_reason: str = ""
    api_call_count: int = 0
    started_at_ms: int = field(default_factory=_now_ms)
    ended_at_ms: int | None = None
    duration_ms: int | None = None
    tool_calls: list[RunTraceToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": _safe_identifier(self.run_id, max_len=160),
            "session_id": _safe_identifier(self.session_id, max_len=160),
            "turn_id": _safe_identifier(self.turn_id, max_len=160),
            "task_id": _safe_identifier(self.task_id, max_len=160),
            "model": _safe_text(self.model, max_len=160),
            "provider": _safe_text(self.provider, max_len=160),
            "source": _safe_text(self.source, max_len=80),
            "status": _safe_status(self.status),
            "exit_reason": _safe_exit_reason(self.exit_reason),
            "api_call_count": int(self.api_call_count or 0),
            "started_at_ms": int(self.started_at_ms or 0),
            "ended_at_ms": int(self.ended_at_ms or 0) if self.ended_at_ms is not None else None,
            "duration_ms": int(self.duration_ms) if isinstance(self.duration_ms, int) else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


def start_trace_for_turn(
    agent: Any,
    *,
    turn_id: str,
    task_id: str | None,
    config: dict[str, Any] | None = None,
) -> RunTrace | None:
    """Build a run trace for a turn when tracing is enabled."""

    if not trace_enabled(config=config):
        return None
    return RunTrace(
        session_id=_safe_text(getattr(agent, "session_id", ""), max_len=160),
        turn_id=_safe_text(turn_id, max_len=160),
        task_id=_safe_text(task_id or "", max_len=160),
        model=_safe_text(getattr(agent, "model", ""), max_len=160),
        provider=_safe_text(getattr(agent, "provider", ""), max_len=160),
        source=_safe_text(getattr(agent, "platform", "") or "cli", max_len=80),
    )


def record_tool_batch(
    trace: RunTrace | None,
    tool_calls: Iterable[Any] | None,
    *,
    status: str = "requested",
) -> None:
    """Append safe metadata for a batch of tool calls.

    Raw arguments are intentionally ignored.
    """

    if trace is None or not tool_calls:
        return
    for tc in tool_calls:
        function = getattr(tc, "function", None)
        trace.tool_calls.append(
            RunTraceToolCall(
                name=_safe_tool_name(getattr(function, "name", "")),
                tool_call_id=_safe_text(getattr(tc, "id", ""), max_len=160),
                status=status,
            )
        )


def append_run_trace(
    trace: RunTrace,
    *,
    config: dict[str, Any] | None = None,
) -> bool:
    """Append one run trace JSON line.

    Returns ``False`` instead of raising for all disabled/write-failure paths;
    tracing is observability only and must never fail an agent turn.
    """

    if trace is None or not trace_enabled(config=config):
        return False
    try:
        path = _trace_path(config=config)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return True
    except Exception:
        logger.debug("run_trace append failed", exc_info=True)
        return False


def finish_trace_for_turn(
    trace: RunTrace | None,
    *,
    status: str,
    exit_reason: str,
    api_call_count: int,
    config: dict[str, Any] | None = None,
) -> bool:
    """Mark a trace complete and append it, swallowing observability failures."""

    if trace is None:
        return False
    try:
        now = _now_ms()
        trace.status = status
        trace.exit_reason = exit_reason
        trace.api_call_count = api_call_count
        trace.ended_at_ms = now
        trace.duration_ms = max(0, now - int(trace.started_at_ms or now))
        return append_run_trace(trace, config=config)
    except Exception:
        logger.debug("run_trace finish failed", exc_info=True)
        return False


__all__ = [
    "RunTrace",
    "RunTraceToolCall",
    "append_run_trace",
    "finish_trace_for_turn",
    "record_tool_batch",
    "start_trace_for_turn",
    "trace_enabled",
]
