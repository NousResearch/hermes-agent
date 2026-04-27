"""Lightweight tracing system for Hermes Agent.

Provides per-turn spans recorded in the SessionDB SQLite store — no external
dependencies, no OTel collector needed. Each span captures timing, tokens,
cost, and tool outcomes for a single agent turn or tool call.

Design goals:
- Zero config: spans write to the same state.db as sessions
- Replay: hermes replay --session-id <id> reconstructs a run from spans
- Observable: /spans slash command shows per-turn cost/latency breakdown
- Compatible: does not block or slow down the agent loop
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Span data model
# =========================================================================

@dataclass
class Span:
    """A single unit of work — one agent turn, one tool call, one LLM request."""

    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    session_id: Optional[str] = None
    name: str = ""  # "agent-turn", "tool.browser_click", "llm.anthropic"
    kind: str = "span"  # "span" | "event" | "error"

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # LLM-specific
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Tool-specific
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None  # JSON-serialized args (truncated)
    tool_output_size: int = 0
    tool_error: Optional[str] = None

    # General
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # "ok" | "error" | "cancelled"
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        return cls(**{k: v for k, v in data.items() if k in Span.__dataclass_fields__})


# =========================================================================
# Tracer — context-aware span management
# =========================================================================

class Tracer:
    """Manages a stack of active spans for the current agent turn.

    Usage:
        tracer = Tracer(session_id="abc123")

        # Agent turn (root span)
        with tracer.span("agent-turn", kind="span") as turn:
            turn.attributes["turn_number"] = 5

            # Tool call (child span)
            with tracer.span("tool.terminal", tool_name="terminal") as tool:
                tool.tool_input = '{"command": "ls"}'
                # ... execute ...
                tool.tool_output_size = 1024

            # LLM call (child span)
            with tracer.span("llm.anthropic", model="claude-sonnet-4") as llm:
                llm.input_tokens = 12000
                llm.output_tokens = 800
                llm.estimated_cost_usd = 0.084
    """

    def __init__(self, session_id: Optional[str] = None, db=None):
        self.session_id = session_id
        self.db = db  # SessionDB instance
        self._stack: List[Span] = []
        self._completed: List[Span] = []

    @contextmanager
    def span(self, name: str, **kwargs):
        """Context manager that creates, yields, and finalizes a span."""
        span = Span(
            name=name,
            session_id=self.session_id,
            parent_id=self._stack[-1].span_id if self._stack else None,
            **kwargs,
        )
        self._stack.append(span)
        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.error_message = str(e)
            raise
        finally:
            span.end_time = time.time()
            self._stack.pop()
            self._completed.append(span)
            self._persist(span)

    def event(self, name: str, **kwargs):
        """Record a point-in-time event (zero-duration span)."""
        now = time.time()
        span = Span(
            name=name,
            kind="event",
            session_id=self.session_id,
            parent_id=self._stack[-1].span_id if self._stack else None,
            start_time=now,
            end_time=now,
            **kwargs,
        )
        self._completed.append(span)
        self._persist(span)
        return span

    def record_error(self, name: str, error: Exception, **kwargs):
        """Record an error span."""
        now = time.time()
        span = Span(
            name=name,
            kind="error",
            session_id=self.session_id,
            parent_id=self._stack[-1].span_id if self._stack else None,
            start_time=now,
            end_time=now,
            status="error",
            error_message=str(error),
            **kwargs,
        )
        self._completed.append(span)
        self._persist(span)
        return span

    def _persist(self, span: Span):
        """Write span to SessionDB if available."""
        if self.db is None:
            return
        try:
            self.db.record_span(span.to_dict())
        except Exception:
            logger.debug("Failed to persist span %s", span.span_id, exc_info=True)

    def get_completed(self) -> List[Span]:
        return list(self._completed)

    def get_completed_dicts(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._completed]

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate stats from completed spans."""
        spans = self._completed
        if not spans:
            return {"total_spans": 0}

        llm_spans = [s for s in spans if s.name.startswith("llm.")]
        tool_spans = [s for s in spans if s.name.startswith("tool.")]
        turn_spans = [s for s in spans if s.name == "agent-turn"]

        total_cost = sum(s.estimated_cost_usd for s in llm_spans)
        total_input = sum(s.input_tokens for s in llm_spans)
        total_output = sum(s.output_tokens for s in llm_spans)
        total_tool_errors = sum(1 for s in tool_spans if s.status == "error")

        return {
            "total_spans": len(spans),
            "turns": len(turn_spans),
            "llm_calls": len(llm_spans),
            "tool_calls": len(tool_spans),
            "tool_errors": total_tool_errors,
            "total_cost_usd": round(total_cost, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_duration_s": round(sum(s.duration_s for s in spans), 2),
            "avg_turn_latency_ms": round(
                sum(s.duration_ms for s in turn_spans) / len(turn_spans), 1
            ) if turn_spans else 0,
        }


# =========================================================================
# Replay — reconstruct a session from stored spans
# =========================================================================

def replay_spans(spans_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Replay stored span data into a human-readable timeline.

    Returns a structured timeline suitable for CLI display or analysis.
    """
    spans = [Span.from_dict(s) for s in spans_data]
    spans.sort(key=lambda s: s.start_time)

    timeline = []
    for span in spans:
        entry = {
            "time": f"{span.start_time:.1f}",
            "span_id": span.span_id,
            "name": span.name,
            "kind": span.kind,
            "status": span.status,
            "duration_ms": round(span.duration_ms, 1),
        }

        if span.model:
            entry["model"] = span.model
        if span.input_tokens or span.output_tokens:
            entry["tokens"] = f"in={span.input_tokens} out={span.output_tokens}"
        if span.estimated_cost_usd > 0:
            entry["cost"] = f"${span.estimated_cost_usd:.4f}"
        if span.tool_name:
            entry["tool"] = span.tool_name
        if span.tool_error:
            entry["tool_error"] = span.tool_error
        if span.error_message:
            entry["error"] = span.error_message

        # Indentation based on depth
        entry["indent"] = 0
        parent_ids = set()
        s = span
        while s.parent_id:
            entry["indent"] += 1
            parent_ids.add(s.parent_id)
            parent = next((x for x in spans if x.span_id == s.parent_id), None)
            if parent is None:
                break
            s = parent

        timeline.append(entry)

    # Aggregate summary
    summary = {
        "total_spans": len(spans),
        "total_cost_usd": round(sum(s.estimated_cost_usd for s in spans), 4),
        "total_input_tokens": sum(s.input_tokens for s in spans),
        "total_output_tokens": sum(s.output_tokens for s in spans),
        "total_duration_s": round(
            (max(s.end_time or s.start_time for s in spans) -
             min(s.start_time for s in spans)), 2
        ) if spans else 0,
        "errors": [s for s in spans if s.status == "error"],
    }

    return {"timeline": timeline, "summary": summary}


def format_replay_timeline(replay_result: Dict[str, Any]) -> str:
    """Format replay output as a readable text timeline."""
    lines = []
    summary = replay_result.get("summary", {})

    lines.append("=" * 60)
    lines.append("Session Replay")
    lines.append("=" * 60)
    lines.append(f"  Spans:       {summary.get('total_spans', 0)}")
    lines.append(f"  Duration:    {summary.get('total_duration_s', 0):.1f}s")
    lines.append(f"  Cost:        ${summary.get('total_cost_usd', 0):.4f}")
    lines.append(f"  Tokens:      in={summary.get('total_input_tokens', 0)} "
                 f"out={summary.get('total_output_tokens', 0)}")
    errors = summary.get("errors", [])
    if errors:
        lines.append(f"  Errors:      {len(errors)}")
    lines.append("-" * 60)

    for entry in replay_result.get("timeline", []):
        indent = "  " * entry["indent"]
        status_icon = {
            "ok": "✓",
            "error": "✗",
            "cancelled": "○",
        }.get(entry.get("status", ""), "?")

        parts = [
            f"{indent}{status_icon}",
            f"[{entry['duration_ms']:.0f}ms]",
            entry["name"],
        ]

        if entry.get("model"):
            parts.append(f"({entry['model']})")
        if entry.get("tokens"):
            parts.append(entry["tokens"])
        if entry.get("cost"):
            parts.append(entry["cost"])
        if entry.get("tool_error"):
            parts.append(f"ERROR: {entry['tool_error']}")
        if entry.get("error"):
            parts.append(f"ERROR: {entry['error']}")

        lines.append(" ".join(parts))

    lines.append("=" * 60)
    return "\n".join(lines)
