"""
Telemetry Backends for Hermes-Agent.

This module provides pluggable telemetry backends for routing events
to different observability destinations (console, file, OpenTelemetry).

It also provides the ``HermesTracer`` — a high-level OpenTelemetry wrapper
that manages:
  - W3C TraceContext trace_id generation and propagation
  - Named spans for agent loop, tool calls, and subagent execution
  - Metrics (counters + histograms) for token usage, latency, and errors
  - Configurable OTLP / Jaeger / Console exporters

Usage:
    from agent.hermes.telemetry import HermesTracer, ConsoleBackend, TelemetryPipeline

    tracer = HermesTracer(service_name="hermes-agent")
    tracer.start_session(session_id="abc", trace_id=None)   # generate new trace

    with tracer.span("tool_call", name="bash", attributes={"tool.name": "bash"}):
        # ... do work ...
        tracer.record_token_usage(input_tokens=100, output_tokens=50)

    tracer.end_session()

    # Fan-out via TelemetryPipeline
    pipeline = TelemetryPipeline()
    pipeline.add_backend(ConsoleBackend())
    pipeline.add_backend(FileBackend("/var/log/hermes/events.jsonl"))
    pipeline.emit(event)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# TelemetryBackend (original interface — preserved for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

class TelemetryBackend(ABC):
    """
    Abstract base class for telemetry backends.

    All backends must implement emit() which is called for each event.
    Exceptions are caught and logged — backends are fire-and-forget.
    """

    @abstractmethod
    def emit(self, event: "Event") -> None:
        """
        Emit an event to this backend.

        Args:
            event: The Event to emit (from agent.hermes.analytics)

        Note:
            Implementations must NOT raise exceptions.
            Any exceptions must be caught and logged internally.
        """
        pass

    def _event_to_dict(self, event: "Event") -> Dict[str, Any]:
        """Convert an Event to a serializable dictionary."""
        return {
            "type": event.type,
            "payload": event.payload,
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
            "session_id": event.session_id,
        }


class ConsoleBackend(TelemetryBackend):
    """
    Prints structured JSON events to stdout.

    Useful for debugging and development.
    """

    def __init__(self, prefix: str = "TELEMETRY"):
        """
        Initialize ConsoleBackend.

        Args:
            prefix: Prefix for each output line (default: "TELEMETRY")
        """
        self.prefix = prefix

    def emit(self, event: "Event") -> None:
        """Print event as JSON to stdout."""
        try:
            data = self._event_to_dict(event)
            print(f"[{self.prefix}] {json.dumps(data)}")
        except Exception as e:
            logger.warning(f"ConsoleBackend.emit failed: {e}")


class FileBackend(TelemetryBackend):
    """
    Appends JSON lines to a file.

    Thread-safe via threading.Lock. Uses append mode to support
    multiple processes writing to the same file.
    """

    def __init__(self, filepath: str, flush: bool = False):
        """
        Initialize FileBackend.

        Args:
            filepath: Path to the output JSONL file
            flush: If True, flush after each write (default: False)
        """
        self.filepath = Path(filepath)
        self.flush = flush
        self._lock = threading.Lock()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: "Event") -> None:
        """Append event as JSON line to file."""
        try:
            data = self._event_to_dict(event)
            line = json.dumps(data) + "\n"
            with self._lock:
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(line)
                    if self.flush:
                        f.flush()
        except Exception as e:
            logger.warning(f"FileBackend.emit failed: {e}")


class TelemetryPipeline:
    """
    Routes events to multiple backends (fan-out).

    All backends receive every event. Emits are fire-and-forget —
    exceptions in backends are caught and logged.
    """

    def __init__(self):
        self._backends: list[TelemetryBackend] = []
        self._lock = threading.Lock()

    def add_backend(self, backend: TelemetryBackend) -> None:
        with self._lock:
            self._backends.append(backend)

    def remove_backend(self, backend: TelemetryBackend) -> None:
        with self._lock:
            self._backends = [b for b in self._backends if b is not backend]

    def emit(self, event: "Event") -> None:
        with self._lock:
            backends = list(self._backends)
        for backend in backends:
            try:
                backend.emit(event)
            except Exception as e:
                logger.warning(f"TelemetryPipeline backend {backend.__class__.__name__} failed: {e}")

    def clear_backends(self) -> None:
        with self._lock:
            self._backends.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._backends)


# ──────────────────────────────────────────────────────────────────────────────
# OpenTelemetry helpers — lazy imports so the module works without OTel installed
# ──────────────────────────────────────────────────────────────────────────────

_OTEL_AVAILABLE: bool = False
try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry import trace as _otel_trace
    from opentelemetry.metrics import Counter, Histogram, ObservableGauge
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation import set_span_in_context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagate import set_global_textmap
    _OTEL_AVAILABLE = True
except ImportError:
    Span = None
    SpanKind = None
    Status = None
    StatusCode = None
    Counter = None
    Histogram = None
    ObservableGauge = None


# ──────────────────────────────────────────────────────────────────────────────
# Exporter registry
# ──────────────────────────────────────────────────────────────────────────────

class ExporterType(Enum):
    """Supported OpenTelemetry exporter types."""
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"
    JAEGER = "jaeger"
    CONSOLE = "console"          # SDK ConsoleSpanExporter
    LOGGING = "logging"          # Python logging bridge
    NOOP = "noop"


def _build_exporter(
    exporter_type: ExporterType,
    endpoint: str,
    insecure: bool = True,
) -> Optional[Any]:
    """
    Build an OTel exporter from an ExporterType and endpoint.

    Returns None if OTel is not installed or the exporter is not available.
    """
    if not _OTEL_AVAILABLE:
        return None

    try:
        if exporter_type == ExporterType.OTLP_GRPC:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as _grpc_exp
            return _grpc_exp(endpoint=endpoint, insecure=insecure)

        if exporter_type == ExporterType.OTLP_HTTP:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as _http_exp
            return _http_exp(endpoint=endpoint)

        if exporter_type == ExporterType.JAEGER:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter as _jg_exp
            return _jg_exp(endpoint=endpoint)

        if exporter_type == ExporterType.CONSOLE:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            return ConsoleSpanExporter()

        if exporter_type == ExporterType.LOGGING:
            # Return a placeholder; handled specially in setup()
            return "logging"

        return None
    except Exception as exc:
        logger.warning("Failed to build %s exporter at %s: %s", exporter_type.value, endpoint, exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# HermesTracer — high-level OpenTelemetry interface
# ──────────────────────────────────────────────────────────────────────────────

class SpanKind(Enum):
    """Kind of span, mirroring OpenTelemetry SpanKind."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SpanContext:
    """W3C TraceContext span context."""
    trace_id: str
    span_id: str
    trace_flags: str = "01"   # sampled

    def to_trace_parent(self) -> str:
        """Return the W3C traceparent header value."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"

    @classmethod
    def from_trace_parent(cls, traceparent: str) -> "SpanContext":
        """Parse a W3C traceparent header into a SpanContext."""
        parts = traceparent.split("-")
        if len(parts) != 4 or parts[0] != "00":
            raise ValueError(f"Invalid traceparent: {traceparent}")
        return cls(trace_id=parts[1], span_id=parts[2], trace_flags=parts[3])


class HermesTracer:
    """
    High-level OpenTelemetry wrapper for Hermes-Agent.

    Features:
      - W3C TraceContext trace_id generation and propagation via env / headers
      - Named spans for agent loop, tool calls, subagent execution
      - Metrics (counters + histograms) for token usage, latency, errors
      - Configurable exporters (OTLP gRPC/HTTP, Jaeger, Console)
      - Thread-safe — each thread gets its own active span context
      - Graceful no-op when OpenTelemetry packages are not installed

    The tracer is a singleton per service_name. Access it via ``HermesTracer.get()``.

    Usage::

        tracer = HermesTracer.get("hermes-agent")
        tracer.start_session(session_id="abc", trace_id=None)  # None = auto-generate

        with tracer.span("tool_call", name="bash", attributes={"tool.name": "bash"}):
            result = execute_bash("ls")
            tracer.record_token_usage(input_tokens=100, output_tokens=50)

        tracer.end_session()

    Environment variables (mirrors OTel SDK conventions):
      OTEL_SERVICE_NAME     — service name
      OTEL_EXPORTER_TYPE    — otlp_grpc | otlp_http | jaeger | console | noop
      OTEL_EXPORTER_ENDPOINT — collector endpoint
      HERMES_TRACE_ID       — optional incoming trace_id to continue
    """

    _instances: Dict[str, "HermesTracer"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(
        cls,
        service_name: str = "hermes-agent",
        endpoint: Optional[str] = None,
        exporter_type: Optional[ExporterType] = None,
    ) -> "HermesTracer":
        """
        Get (or create) a HermesTracer singleton for *service_name*.

        Configuration is read from env vars if not passed explicitly.
        Subsequent calls with the same service_name return the same instance.
        """
        with cls._instances_lock:
            if service_name not in cls._instances:
                cls._instances[service_name] = HermesTracer(
                    service_name=service_name,
                    endpoint=endpoint,
                    exporter_type=exporter_type,
                )
            return cls._instances[service_name]

    def __init__(
        self,
        service_name: str = "hermes-agent",
        endpoint: Optional[str] = None,
        exporter_type: Optional[ExporterType] = None,
    ):
        self.service_name = service_name
        self._lock = threading.Lock()

        # Current session context
        self._session_id: Optional[str] = None
        self._trace_id: Optional[str] = None
        self._root_span: Optional[Any] = None   # OTel Span or None
        self._spans: List[Any] = []

        # OTel primitives (None until setup())
        self._tracer: Optional[Any] = None
        self._meter: Optional[Any] = None
        self._exporter_type = exporter_type
        self._endpoint = endpoint

        # Metric instruments
        self._token_counter: Optional[Any] = None
        self._token_histogram: Optional[Any] = None
        self._latency_histogram: Optional[Any] = None
        self._error_counter: Optional[Any] = None
        self._span_counter: Optional[Any] = None

        # Per-thread active span stack
        self._thread_local = threading.local()

        if _OTEL_AVAILABLE:
            self._setup_otel()
        else:
            logger.debug("OpenTelemetry packages not installed — tracer runs in no-op mode")

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_otel(self) -> None:
        """Initialize OTel tracer and meter providers."""
        if not _OTEL_AVAILABLE:
            return

        # Read config from env
        svc_name = os.environ.get("OTEL_SERVICE_NAME", self.service_name)
        exp_type_str = os.environ.get("OTEL_EXPORTER_TYPE", "")
        exp_endpoint = os.environ.get(
            "OTEL_EXPORTER_ENDPOINT",
            self._endpoint or "http://localhost:4317",
        )

        if self._exporter_type is None:
            try:
                self._exporter_type = ExporterType(exp_type_str) if exp_type_str else ExporterType.OTLP_GRPC
            except ValueError:
                self._exporter_type = ExporterType.OTLP_GRPC

        if self._endpoint is None:
            self._endpoint = exp_endpoint

        # Build resource
        resource = Resource.create({
            "service.name": svc_name,
            "service.version": "0.8.0",
            "deployment.environment": os.environ.get("OTEL_ENV", "development"),
        })

        # Build tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Add span exporter
        if self._exporter_type != ExporterType.NOOP and self._endpoint:
            exporter = _build_exporter(self._exporter_type, self._endpoint)
            if exporter:
                tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            else:
                logger.warning(
                    "Could not build %s exporter at %s — tracing disabled",
                    self._exporter_type.value,
                    self._endpoint,
                )

        # Also expose spans via Python logging when in logging mode
        if self._exporter_type == ExporterType.LOGGING:
            try:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            except Exception as exc:
                logger.warning("ConsoleSpanExporter unavailable: %s", exc)

        _otel_trace.set_tracer_provider(tracer_provider)

        # Set W3C TraceContext propagator globally
        try:
            set_global_textmap(TraceContextTextMapPropagator())
        except Exception as exc:
            logger.debug("Could not set global propagator: %s", exc)

        self._tracer = _otel_trace.get_tracer(__name__, "0.8.0")

        # Build meter provider with optional metric exporter
        try:
            metric_reader = None
            if self._exporter_type not in (ExporterType.NOOP, None) and self._endpoint:
                from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
                metric_exporter = _build_metric_exporter(self._exporter_type, self._endpoint)
                if metric_exporter:
                    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30_000)

            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader] if metric_reader else [])
            _otel_metrics.set_meter_provider(meter_provider)
            self._meter = _otel_metrics.get_meter(__name__, "0.8.0")

            self._create_metric_instruments()
        except Exception as exc:
            logger.warning("Failed to set up metrics: %s", exc)

    def _create_metric_instruments(self) -> None:
        """Create OTel metric instruments (counters + histograms)."""
        if not self._meter:
            return

        try:
            self._token_counter = self._meter.create_counter(
                name="hermes.tokens",
                description="Token usage counter",
                unit="1",
            )
        except Exception:
            pass

        try:
            self._token_histogram = self._meter.create_histogram(
                name="hermes.tokens.histogram",
                description="Token usage histogram",
                unit="1",
            )
        except Exception:
            pass

        try:
            self._latency_histogram = self._meter.create_histogram(
                name="hermes.latency",
                description="Operation latency in seconds",
                unit="s",
            )
        except Exception:
            pass

        try:
            self._error_counter = self._meter.create_counter(
                name="hermes.errors",
                description="Error counter",
                unit="1",
            )
        except Exception:
            pass

        try:
            self._span_counter = self._meter.create_counter(
                name="hermes.spans",
                description="Span creation counter",
                unit="1",
            )
        except Exception:
            pass

    # ── Trace ID management ─────────────────────────────────────────────────

    @staticmethod
    def generate_trace_id() -> str:
        """
        Generate a 32-character hex trace_id (128-bit, W3C TraceContext compliant).

        Uses os.urandom for cryptographic quality.
        """
        return uuid.uuid4().hex

    @staticmethod
    def generate_span_id() -> str:
        """
        Generate a 16-character hex span_id (64-bit, W3C TraceContext compliant).
        """
        return uuid.uuid4().hex[:16]

    @classmethod
    def parse_traceparent(cls, traceparent: str) -> tuple[str, str]:
        """
        Extract (trace_id, span_id) from a W3C traceparent header.

        Returns (trace_id, span_id). Raises ValueError on parse failure.
        """
        ctx = SpanContext.from_trace_parent(traceparent)
        return ctx.trace_id, ctx.span_id

    def get_trace_id(self) -> Optional[str]:
        """Return the current trace_id, or None if no session is active."""
        return self._trace_id

    def get_traceparent(self) -> Optional[str]:
        """Return the W3C traceparent for the current session, or None."""
        if not self._trace_id:
            return None
        return f"00-{self._trace_id}-{self.generate_span_id()}-01"

    # ── Session lifecycle ───────────────────────────────────────────────────

    def start_session(
        self,
        session_id: str,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new tracing session.

        Creates the root span for this agent run. If *trace_id* is None,
        a new one is generated. If the HERMES_TRACE_ID env var is set,
        it takes precedence (incoming trace from gateway / upstream).

        Args:
            session_id: The agent session ID
            trace_id: Optional 32-char hex trace_id (W3C format). Auto-generated if None.
            attributes: Optional root span attributes (model, provider, etc.)

        Returns:
            The trace_id (generated or provided)
        """
        with self._lock:
            self._session_id = session_id

            # Priority: env var > provided > generated
            self._trace_id = os.environ.get("HERMES_TRACE_ID") or trace_id or self.generate_trace_id()

            attributes = dict(attributes) if attributes else {}
            attributes["session.id"] = session_id
            attributes["trace.id"] = self._trace_id

            if self._tracer:
                try:
                    kind = _otel_trace.SpanKind.INTERNAL
                    self._root_span = self._tracer.start_span(
                        name=f"session:{session_id}",
                        kind=kind,
                        attributes=attributes,
                    )
                    ctx = set_span_in_context(self._root_span)
                    self._thread_local.ctx = ctx
                except Exception as exc:
                    logger.warning("Failed to start root span: %s", exc)

        logger.debug(
            "OTel session started: session_id=%s trace_id=%s",
            session_id,
            self._trace_id,
        )
        return self._trace_id

    def end_session(self, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        End the current tracing session.

        Closes the root span and records any final attributes.
        """
        with self._lock:
            if self._root_span and _OTEL_AVAILABLE:
                try:
                    if attributes:
                        for k, v in attributes.items():
                            if isinstance(v, (str, int, float, bool)):
                                self._root_span.set_attribute(k, v)
                    self._root_span.end()
                except Exception as exc:
                    logger.warning("Failed to end root span: %s", exc)

            self._root_span = None
            self._session_id = None
            self._trace_id = None
            self._thread_local = threading.local()

    # ── Span management ──────────────────────────────────────────────────────

    def span(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True,
    ) -> "_OtelSpan":
        """
        Return a context-manager span for use with ``with``.

        Usage::

            with tracer.span("tool_call", name="bash", attributes={"tool.name": "bash"}):
                result = run_bash()

        Args:
            name: Span name (e.g. "tool_call", "subagent_execution", "agent_loop")
            kind: SpanKind — internal | server | client | producer | consumer
            attributes: Span attributes (will be merged with auto-added ones)
            record_exception: If True, exceptions in the block are recorded on the span

        Returns:
            An OpenTelemetry span context-manager (is a no-op if OTel unavailable)
        """
        return _OtelSpan(
            tracer=self,
            name=name,
            kind=kind,
            attributes=attributes,
            record_exception=record_exception,
        )

    def start_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Programmatically start a span (use with ``end_span()``).

        Prefer ``span()`` as a context manager. This is for manual span
        lifecycle control.

        Returns the raw OTel Span (or None if OTel unavailable).
        """
        if not self._tracer:
            return None

        attrs = dict(attributes) if attributes else {}
        if self._session_id:
            attrs["session.id"] = self._session_id
        if self._trace_id:
            attrs["trace.id"] = self._trace_id

        kind_map = {
            "internal": _otel_trace.SpanKind.INTERNAL if _OTEL_AVAILABLE else None,
            "server": _otel_trace.SpanKind.SERVER if _OTEL_AVAILABLE else None,
            "client": _otel_trace.SpanKind.CLIENT if _OTEL_AVAILABLE else None,
            "producer": _otel_trace.SpanKind.PRODUCER if _OTEL_AVAILABLE else None,
            "consumer": _otel_trace.SpanKind.CONSUMER if _OTEL_AVAILABLE else None,
        }

        span_kind = kind_map.get(kind, kind_map["internal"])

        try:
            span = self._tracer.start_span(name=name, kind=span_kind, attributes=attrs)
            self._spans.append(span)

            if self._span_counter:
                self._span_counter.add(1, {"span.name": name, "span.kind": kind})

            return span
        except Exception as exc:
            logger.warning("Failed to start span %s: %s", name, exc)
            return None

    def end_span(self, span: Any, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        End a span started with ``start_span()``.

        Args:
            span: The span returned by ``start_span()``
            attributes: Optional final attributes to set before ending
        """
        if not span:
            return
        try:
            if attributes:
                for k, v in attributes.items():
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(k, v)
            span.end()
            if span in self._spans:
                self._spans.remove(span)
        except Exception as exc:
            logger.warning("Failed to end span: %s", exc)

    def set_span_error(self, span: Any, exc: Exception, message: Optional[str] = None) -> None:
        """Record an exception on a span and increment the error counter."""
        if not span:
            return
        try:
            if _OTEL_AVAILABLE:
                span.set_status(Status(StatusCode.ERROR, str(message or exc)))
                span.record_exception(exc)
            if self._error_counter:
                self._error_counter.add(1, {"error.type": type(exc).__name__})
        except Exception:
            pass

    # ── Convenience spans ────────────────────────────────────────────────────

    def agent_loop_span(
        self,
        iteration: int,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "_OtelSpan":
        """
        Span covering a single agent loop iteration.

        Attributes auto-include ``iteration`` and ``session.id``.
        """
        attrs = dict(attributes) if attributes else {}
        attrs["iteration"] = iteration
        return self.span(
            name=f"agent_loop.iter_{iteration}",
            kind="internal",
            attributes=attrs,
        )

    def tool_call_span(
        self,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "_OtelSpan":
        """
        Span covering a single tool call execution.

        Attributes auto-include ``tool.name`` and optionally ``tool.call_id``.
        """
        attrs = dict(attributes) if attributes else {}
        attrs["tool.name"] = tool_name
        if tool_call_id:
            attrs["tool.call_id"] = tool_call_id
        return self.span(
            name=f"tool_call.{tool_name}",
            kind="client",   # tool execution is a "client" of the tool runtime
            attributes=attrs,
        )

    def subagent_span(
        self,
        subagent_name: str,
        depth: int,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "_OtelSpan":
        """
        Span covering subagent execution.

        Attributes auto-include ``subagent.name`` and ``delegate.depth``.
        """
        attrs = dict(attributes) if attributes else {}
        attrs["subagent.name"] = subagent_name
        attrs["delegate.depth"] = depth
        return self.span(
            name=f"subagent.{subagent_name}",
            kind="client",
            attributes=attrs,
        )

    def llm_call_span(
        self,
        model: str,
        provider: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "_OtelSpan":
        """
        Span covering an LLM API call.

        Attributes auto-include ``model`` and ``provider``.
        """
        attrs = dict(attributes) if attributes else {}
        attrs["model"] = model
        if provider:
            attrs["provider"] = provider
        return self.span(
            name=f"llm_call.{model}",
            kind="client",
            attributes=attrs,
        )

    # ── Metrics ─────────────────────────────────────────────────────────────

    def record_token_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record token usage as both a counter and a histogram observation.

        Args:
            input_tokens: Prompt tokens consumed
            output_tokens: Completion tokens produced
            cache_read_tokens: Cached tokens read
            cache_write_tokens: Cached tokens written
            reasoning_tokens: Reasoning/thinking tokens (if supported)
            attributes: Metric label attributes
        """
        attrs = dict(attributes) if attributes else {}
        if self._session_id:
            attrs["session.id"] = self._session_id

        total = input_tokens + output_tokens

        if self._token_counter:
            try:
                self._token_counter.add(total, {**attrs, "token.type": "total"})
                if input_tokens:
                    self._token_counter.add(input_tokens, {**attrs, "token.type": "input"})
                if output_tokens:
                    self._token_counter.add(output_tokens, {**attrs, "token.type": "output"})
                if cache_read_tokens:
                    self._token_counter.add(cache_read_tokens, {**attrs, "token.type": "cache_read"})
                if cache_write_tokens:
                    self._token_counter.add(cache_write_tokens, {**attrs, "token.type": "cache_write"})
                if reasoning_tokens:
                    self._token_counter.add(reasoning_tokens, {**attrs, "token.type": "reasoning"})
            except Exception as exc:
                logger.debug("token_counter.add failed: %s", exc)

        if self._token_histogram:
            try:
                if input_tokens:
                    self._token_histogram.record(input_tokens, {**attrs, "token.type": "input"})
                if output_tokens:
                    self._token_histogram.record(output_tokens, {**attrs, "token.type": "output"})
            except Exception as exc:
                logger.debug("token_histogram.record failed: %s", exc)

    def record_latency(
        self,
        duration_seconds: float,
        operation: str,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record operation latency as a histogram observation.

        Args:
            duration_seconds: Duration of the operation in seconds
            operation: Operation name (e.g. "tool_call", "llm_call", "compression")
            attributes: Metric label attributes
        """
        attrs = dict(attributes) if attributes else {}
        if self._session_id:
            attrs["session.id"] = self._session_id
        attrs["operation"] = operation

        if self._latency_histogram:
            try:
                self._latency_histogram.record(duration_seconds, attrs)
            except Exception as exc:
                logger.debug("latency_histogram.record failed: %s", exc)

    def record_error(
        self,
        error_type: str,
        operation: str,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment the error counter.

        Args:
            error_type: Error class name (e.g. "ValidationError", "TimeoutError")
            operation: Operation where the error occurred
            attributes: Metric label attributes
        """
        attrs = dict(attributes) if attributes else {}
        if self._session_id:
            attrs["session.id"] = self._session_id
        attrs["operation"] = operation
        attrs["error.type"] = error_type

        if self._error_counter:
            try:
                self._error_counter.add(1, attrs)
            except Exception as exc:
                logger.debug("error_counter.add failed: %s", exc)

    def record_span(
        self,
        span_name: str,
        duration_seconds: float,
        success: bool,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a span completion in metrics (used by ``_OtelSpan`` on exit).

        Records the span duration in the latency histogram and increments
        the span counter with success/failure status.
        """
        attrs = dict(attributes) if attributes else {}
        if self._session_id:
            attrs["session.id"] = self._session_id
        attrs["span.name"] = span_name
        attrs["success"] = "true" if success else "false"

        if self._latency_histogram:
            try:
                self._latency_histogram.record(duration_seconds, attrs)
            except Exception as exc:
                logger.debug("latency_histogram.record failed: %s", exc)

        if self._span_counter:
            try:
                self._span_counter.add(1, attrs)
            except Exception as exc:
                logger.debug("span_counter.add failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# _OtelSpan — context-manager wrapper around an OTel span
# ──────────────────────────────────────────────────────────────────────────────

class _OtelSpan:
    """
    Context manager that wraps an OpenTelemetry span.

    Returned by ``HermesTracer.span()`` and its convenience variants
    (``tool_call_span()``, ``agent_loop_span()``, etc.).
    """

    def __init__(
        self,
        tracer: HermesTracer,
        name: str,
        kind: str = "internal",
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._kind = kind
        self._attributes = attributes
        self._record_exception = record_exception
        self._span: Optional[Any] = None
        self._start_time: float = 0.0

    def __enter__(self) -> "_OtelSpan":
        self._start_time = time.time()
        self._span = self._tracer.start_span(
            name=self._name,
            kind=self._kind,
            attributes=self._attributes,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        success = exc_type is None
        duration = time.time() - self._start_time

        if self._span:
            if exc_val and self._record_exception:
                self._tracer.set_span_error(self._span, exc_val)
            elif success:
                try:
                    if _OTEL_AVAILABLE:
                        self._span.set_status(Status(StatusCode.OK))
                except Exception:
                    pass

            # Record latency and span metrics
            self._tracer.record_span(
                span_name=self._name,
                duration_seconds=duration,
                success=success,
                attributes={"span.kind": self._kind},
            )

            self._tracer.end_span(self._span)

        # Don't suppress exceptions
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Metric exporter helper (mirrors span exporter builder)
# ──────────────────────────────────────────────────────────────────────────────

def _build_metric_exporter(
    exporter_type: ExporterType,
    endpoint: str,
) -> Optional[Any]:
    """Build an OTel metric exporter from an ExporterType."""
    if not _OTEL_AVAILABLE:
        return None

    try:
        if exporter_type == ExporterType.OTLP_GRPC:
            from opentelemetry.exporter.otlp.proto.grpc.metric import OTLPMetricExporter as _m_exp
            return _m_exp(endpoint=endpoint, insecure=True)

        if exporter_type == ExporterType.OTLP_HTTP:
            from opentelemetry.exporter.otlp.proto.http.metric import OTLPMetricExporter as _m_exp
            return _m_exp(endpoint=endpoint)

        if exporter_type == ExporterType.CONSOLE:
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            return ConsoleMetricExporter()

        return None
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("Failed to build metric exporter: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatibility shim: keep the old OpenTelemetryBackend name
# ──────────────────────────────────────────────────────────────────────────────

class OpenTelemetryBackend(TelemetryBackend):
    """
    Exports events as OpenTelemetry spans.

    This backend is maintained for backward compatibility. For new code,
    prefer ``HermesTracer`` which provides a richer API.

    Creates a span for each event with event attributes as span attributes.
    """

    def __init__(
        self,
        service_name: str = "hermes-agent",
        endpoint: Optional[str] = None,
    ):
        self.service_name = service_name
        self._tracer: Optional[Any] = None
        self._span_exporter = None

        try:
            from opentelemetry import trace as _t
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)

            if endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    self._span_exporter = OTLPSpanExporter(endpoint=endpoint)
                    provider.add_span_processor(BatchSpanProcessor(self._span_exporter))
                except ImportError:
                    logger.warning("OTLP exporter not available, spans will not be exported")

            _t.set_tracer_provider(provider)
            self._tracer = _t.get_tracer(__name__)
        except ImportError:
            logger.warning("OpenTelemetry not available. Install opentelemetry-api and opentelemetry-sdk.")

    def emit(self, event: "Event") -> None:
        """Create a span for the event."""
        if self._tracer is None:
            return

        try:
            from opentelemetry import trace as _t

            data = self._event_to_dict(event)

            with self._tracer.start_as_current_span(event.type) as span:
                span.set_attribute("session.id", event.session_id or "")
                span.set_attribute("event.timestamp", data["timestamp"] or "")

                # Flatten simple payload values as span attributes
                for key, value in event.payload.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"payload.{key}", str(value)[:256])
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, (str, int, float, bool)):
                                span.set_attribute(f"payload.{key}.{k}", str(v)[:256])
        except Exception as e:
            logger.warning(f"OpenTelemetryBackend.emit failed: {e}")
