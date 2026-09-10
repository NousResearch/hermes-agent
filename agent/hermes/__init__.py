"""
Hermes-Agent Phase 1 Infrastructure Package.

Components:
- streaming: AsyncGenerator streaming support
- shutdown: Graceful shutdown manager
- analytics: EventBus for analytics
- telemetry: Pluggable telemetry backends
- mailbox: Priority queue for async message delivery
- resource_monitor: Real-time CPU/memory/token/latency metrics
- alert_manager: Configurable cost and resource alert thresholds
"""

from agent.hermes.streaming import Delta, stream_conversation
from agent.hermes.shutdown import ShutdownManager
from agent.hermes.analytics import Event, EventBus, EventType
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
from agent.hermes.resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    get_resource_monitor,
    stop_all_monitors,
)
from agent.hermes.alert_manager import (
    AlertManager,
    AlertThresholds,
    ResourceAlert,
    AlertCategory,
    AlertSeverity,
)

__all__ = [
    "Delta",
    "stream_conversation",
    "ShutdownManager",
    "Event",
    "EventBus",
    "EventType",
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
    # Resource monitor
    "ResourceMonitor",
    "ResourceSnapshot",
    "get_resource_monitor",
    "stop_all_monitors",
    # Alert manager
    "AlertManager",
    "AlertThresholds",
    "ResourceAlert",
    "AlertCategory",
    "AlertSeverity",
]
