"""OpenTelemetry Tracing — optional observability for agent execution.

Creates spans around run_conversation, tool calls, and LLM API calls.
Exports to SQLite (local) or OTLP (remote) when OpenTelemetry is available.
Degrades gracefully when otel is not installed.

Inspired by agno's tracing module with Trace/Span hierarchy.

Usage:
    from agent.tracing import get_tracer, trace_tool_call

    tracer = get_tracer()  # Returns real or no-op tracer
    with tracer.start_as_current_span("run_conversation"):
        ...
        with trace_tool_call(tracer, "web_search", args):
            result = tool.execute()

Config in cli-config.yaml:
    tracing:
      enabled: true
      exporter: sqlite  # or "otlp" or "console"
      sqlite_path: ~/.hermes/traces.db
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry — all functionality degrades to no-ops if missing
_HAS_OTEL = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    _HAS_OTEL = True
except ImportError:
    pass


# ============================================================================
# Lightweight trace data model (works without otel)
# ============================================================================

@dataclass
class TraceSpan:
    """A single span in a trace (lightweight, no otel dependency)."""
    span_id: str = ""
    trace_id: str = ""
    parent_span_id: str = ""
    name: str = ""
    kind: str = "internal"  # internal, tool, llm, agent
    status: str = "ok"  # ok, error
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    """A complete trace (collection of spans)."""
    trace_id: str = ""
    name: str = ""
    session_id: str = ""
    spans: List[TraceSpan] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_spans: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "session_id": self.session_id,
            "spans": [s.to_dict() for s in self.spans],
            "total_spans": len(self.spans),
            "error_count": sum(1 for s in self.spans if s.status == "error"),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else 0,
        }


# ============================================================================
# Lightweight tracer (no otel needed)
# ============================================================================

class _SpanContext:
    """Context manager for a lightweight span."""
    def __init__(self, collector: "TraceCollector", name: str, kind: str = "internal", **attrs):
        self._collector = collector
        self._name = name
        self._kind = kind
        self._attrs = attrs
        self._span: Optional[TraceSpan] = None
        self._start: float = 0.0

    def __enter__(self):
        import uuid
        self._start = time.time()
        self._span = TraceSpan(
            span_id=uuid.uuid4().hex[:16],
            trace_id=self._collector.trace_id,
            parent_span_id=self._collector.current_span_id,
            name=self._name,
            kind=self._kind,
            start_time=self._start,
            attributes=self._attrs,
        )
        self._collector._span_stack.append(self._span.span_id)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            self._span.end_time = time.time()
            self._span.duration_ms = (self._span.end_time - self._span.start_time) * 1000
            if exc_type:
                self._span.status = "error"
                self._span.error = str(exc_val) if exc_val else exc_type.__name__
            self._collector.spans.append(self._span)
            if self._collector._span_stack:
                self._collector._span_stack.pop()
        return False  # Don't suppress exceptions


class TraceCollector:
    """Collects spans for a single trace (one per run_conversation call)."""

    def __init__(self, session_id: str = "", name: str = "run_conversation"):
        import uuid
        self.trace_id = uuid.uuid4().hex[:32]
        self.session_id = session_id
        self.name = name
        self.spans: List[TraceSpan] = []
        self._span_stack: List[str] = []
        self.start_time = time.time()

    @property
    def current_span_id(self) -> str:
        return self._span_stack[-1] if self._span_stack else ""

    def span(self, name: str, kind: str = "internal", **attrs) -> _SpanContext:
        """Create a new span context manager."""
        return _SpanContext(self, name, kind, **attrs)

    def to_trace(self) -> Trace:
        """Finalize and return the Trace object."""
        return Trace(
            trace_id=self.trace_id,
            name=self.name,
            session_id=self.session_id,
            spans=self.spans,
            start_time=self.start_time,
            end_time=time.time(),
        )


# ============================================================================
# Persistence
# ============================================================================

def persist_trace(trace: Trace, session_db: Any = None, session_id: str = None) -> None:
    """Store a trace in session state for later analysis."""
    if not session_db or not session_id:
        return
    try:
        state = session_db.get_session_state(session_id)
        traces = state.get("_traces", [])
        # Keep only summary (not full spans) to avoid state bloat
        summary = {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "total_spans": len(trace.spans),
            "error_count": sum(1 for s in trace.spans if s.status == "error"),
            "duration_ms": (trace.end_time - trace.start_time) * 1000,
            "timestamp": trace.start_time,
            "span_names": [s.name for s in trace.spans[:20]],  # First 20 spans
        }
        traces.append(summary)
        if len(traces) > 20:
            traces = traces[-20:]
        session_db.update_session_state(session_id, {"_traces": traces})
    except Exception as e:
        logger.debug("Failed to persist trace: %s", e)


# ============================================================================
# Convenience helpers for agent integration
# ============================================================================

@contextmanager
def trace_tool_call(collector: Optional[TraceCollector], tool_name: str, args: dict = None):
    """Context manager for tracing a tool call."""
    if collector is None:
        yield None
        return
    with collector.span(f"tool:{tool_name}", kind="tool", tool_name=tool_name, args_keys=list((args or {}).keys())) as span:
        yield span


@contextmanager
def trace_llm_call(collector: Optional[TraceCollector], model: str = "", provider: str = ""):
    """Context manager for tracing an LLM API call."""
    if collector is None:
        yield None
        return
    with collector.span(f"llm:{model}", kind="llm", model=model, provider=provider) as span:
        yield span


def setup_otel_tracer(exporter: str = "console") -> Optional[Any]:
    """Set up OpenTelemetry tracer if available. Returns tracer or None."""
    if not _HAS_OTEL:
        return None
    try:
        provider = TracerProvider()
        if exporter == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        return trace.get_tracer("hermes-agent")
    except Exception as e:
        logger.debug("Failed to set up OTel tracer: %s", e)
        return None
