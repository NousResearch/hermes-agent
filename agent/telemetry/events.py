"""Typed local telemetry events.

These dataclasses are the rows written to the local JSONL log and the ``tel_*``
SQLite tables. They record the values observed for each run — model id, provider, tool
name, token counts, durations — and stay on the machine unless explicitly exported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

# ── local telemetry events (real values) ────────────────────────────────────


def _now_ns() -> int:
    return time.time_ns()


@dataclass(slots=True)
class RunEvent:
    """One top-level workflow execution (a trace root). A run spans one session."""
    run_id: str
    trace_id: str
    entrypoint: str
    session_id: Optional[str] = None
    platform: Optional[str] = None
    start_ns: int = field(default_factory=_now_ns)
    end_ns: Optional[int] = None
    end_reason: Optional[str] = None
    model_call_count: int = 0
    tool_call_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "run", **asdict(self)}


@dataclass(slots=True)
class ModelCallEvent:
    span_id: str
    run_id: str
    provider: Optional[str] = None        # raw provider, e.g. "anthropic"
    model: Optional[str] = None           # raw model id, e.g. "claude-opus-4"
    base_url: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "model_call", **asdict(self)}


@dataclass(slots=True)
class ToolCallEvent:
    span_id: str
    run_id: str
    tool_name: Optional[str] = None       # raw tool name, e.g. "web_search"
    duration_ms: Optional[int] = None
    result_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "tool_call", **asdict(self)}


@dataclass(slots=True)
class SpanEvent:
    """A timed span — the timing/lineage backbone of a trace.

    One row per run (the root, ``parent_span_id=None``) and one per model/tool call
    (``parent_span_id`` = the run's root span). Detail rows in ``tel_model_calls`` /
    ``tel_tool_calls`` share the ``span_id`` and are joined here for ordering and
    placement on a timeline.
    """
    span_id: str
    trace_id: str
    run_id: str
    name: str
    kind: str                              # "run" | "model" | "tool"
    start_ns: int
    end_ns: Optional[int] = None
    parent_span_id: Optional[str] = None
    status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "span", **asdict(self)}


@dataclass(slots=True)
class ErrorEvent:
    run_id: Optional[str]
    error_class: str
    subsystem: str
    recovery: Optional[str] = None
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "error", **asdict(self)}


@dataclass(slots=True)
class GatewayHealthEvent:
    """Content-free gateway health snapshot or lifecycle event."""

    name: str
    gateway_state: Optional[str] = None
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    exit_reason: Optional[str] = None
    restart_requested: Optional[bool] = None
    active_agents: int = 0
    gateway_busy: bool = False
    gateway_drainable: bool = False
    platform_count: int = 0
    fatal_platform_count: int = 0
    profile: Optional[str] = None
    install_id: Optional[str] = None
    version: Optional[str] = None
    supervision_mode: Optional[str] = None
    pid: Optional[int] = None
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "gateway_health", **asdict(self)}


@dataclass(slots=True)
class GatewayDiagnosticEvent:
    """Redacted gateway diagnostic event for customer-owned operations."""

    name: str
    subsystem: str
    error_class: str = "unknown"
    error_code: Optional[str] = None
    redacted_message: Optional[str] = None
    platform: Optional[str] = None
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    profile: Optional[str] = None
    version: Optional[str] = None
    severity: str = "warning"
    ts_ns: int = field(default_factory=_now_ns)

    def to_dict(self) -> Dict[str, Any]:
        return {"event": "gateway_diagnostic", **asdict(self)}


__all__ = [
    "RunEvent",
    "ModelCallEvent",
    "ToolCallEvent",
    "SpanEvent",
    "ErrorEvent",
    "GatewayHealthEvent",
    "GatewayDiagnosticEvent",
]
