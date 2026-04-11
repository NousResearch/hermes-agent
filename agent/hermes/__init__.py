"""
Hermes-Agent Phase 1 Infrastructure Package.

Components:
- streaming: AsyncGenerator streaming support
- shutdown: Graceful shutdown manager
- analytics: EventBus for analytics
- telemetry: Pluggable telemetry backends
- mailbox: Priority queue for async message delivery
"""

from agent.hermes.streaming import Delta, stream_conversation
from agent.hermes.shutdown import ShutdownManager
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
from agent.hermes.mailbox import MailboxMessage, Mailbox
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
    "Delta",
    "stream_conversation",
    "ShutdownManager",
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
    "MailboxMessage",
    "Mailbox",
    "HermesJSONFormatter",
    "HermesStructuredLogHandler",
    "StructuredLoggerAdapter",
    "get_structured_logger",
    "set_log_context",
    "get_log_context",
    "clear_log_context",
    "setup_structured_logging",
]
