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
)
from agent.hermes.mailbox import MailboxMessage, Mailbox

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
    "MailboxMessage",
    "Mailbox",
]
