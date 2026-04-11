"""
Hermes-Agent Phase 1 Infrastructure Package.

Components:
- analytics: EventBus for analytics
- telemetry: Pluggable telemetry backends
- shutdown: Graceful shutdown manager
- structured_logging: Structured JSON logging with context
"""

from agent.hermes.analytics import Event, EventBus
from agent.hermes.telemetry import (
    TelemetryBackend,
    ConsoleBackend,
    FileBackend,
    OpenTelemetryBackend,
    TelemetryPipeline,
    HermesTracer,
    ExporterType,
    SpanContext,
)
from agent.hermes.shutdown import ShutdownManager
from agent.hermes.structured_logging import (
    HermesJSONFormatter,
    HermesStructuredLogHandler,
    StructuredLoggerAdapter,
    get_structured_logger,
    set_log_context,
    get_log_context,
    clear_log_context,
    setup_structured_logging,
)

__all__ = [
    "Event",
    "EventBus",
    "TelemetryBackend",
    "ConsoleBackend",
    "FileBackend",
    "OpenTelemetryBackend",
    "TelemetryPipeline",
    "HermesTracer",
    "ExporterType",
    "SpanContext",
    "ShutdownManager",
    "HermesJSONFormatter",
    "HermesStructuredLogHandler",
    "StructuredLoggerAdapter",
    "get_structured_logger",
    "set_log_context",
    "get_log_context",
    "clear_log_context",
    "setup_structured_logging",
]
