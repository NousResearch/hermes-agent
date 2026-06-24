"""Trace / run / span id propagation via contextvars.

Telemetry events share IDs so a workflow can be reconstructed: one ``trace_id`` per
workflow, one ``run_id`` per top-level execution, ``span_id`` per timed operation, and
``parent_span_id`` for nesting. These live in contextvars so async tool calls and
spawned subagents inherit the lineage automatically.

Provides helpers to start/clear a run context and mint child span ids. The telemetry
plugin sets the run context on session start and reads it in each hook callback.
Nothing here writes to storage — it only carries ids.
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass
from typing import Optional

_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hermes_tel_trace_id", default=None
)
_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hermes_tel_run_id", default=None
)
_parent_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hermes_tel_parent_span_id", default=None
)


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(slots=True)
class RunContext:
    trace_id: str
    run_id: str


def start_run(trace_id: Optional[str] = None, run_id: Optional[str] = None) -> RunContext:
    """Begin a run context, minting ids when not supplied. Sets contextvars."""
    tid = trace_id or new_id()
    rid = run_id or new_id()
    _trace_id.set(tid)
    _run_id.set(rid)
    _parent_span_id.set(None)
    return RunContext(trace_id=tid, run_id=rid)


def current_trace_id() -> Optional[str]:
    return _trace_id.get()


def current_run_id() -> Optional[str]:
    return _run_id.get()


def current_parent_span_id() -> Optional[str]:
    return _parent_span_id.get()


def new_span_id() -> str:
    """Mint a span id (does not alter the parent pointer)."""
    return new_id()


def clear_run() -> None:
    _trace_id.set(None)
    _run_id.set(None)
    _parent_span_id.set(None)


__all__ = [
    "RunContext",
    "new_id",
    "start_run",
    "current_trace_id",
    "current_run_id",
    "current_parent_span_id",
    "new_span_id",
    "clear_run",
]
