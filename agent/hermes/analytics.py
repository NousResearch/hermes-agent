"""
Analytics EventBus for Hermes-Agent.

This module provides a thread-safe EventBus for internal analytics and observability.
It is a separate system from invoke_hook (external plugin extensibility).

Built-in event types:
    - tool.call: Fired when a tool is about to be executed
    - tool.result: Fired when a tool completes (success or failure)
    - llm.call: Fired when an LLM API call is made
    - llm.response: Fired when an LLM API response is received
    - session.start: Fired when a new session starts
    - session.end: Fired when a session ends
    - error: Fired when an error occurs

Usage:
    from agent.hermes.analytics import EventBus, Event

    bus = EventBus()

    def handler(event: Event):
        print(f"Event: {event.type}, payload: {event.payload}")

    bus.subscribe("session.start", handler)
    bus.emit(Event("session.start", {"session_id": "abc123"}))

Note:
    Phase 1 creates this module only. EventBus instantiation is deferred
    to Phase 2 Bootstrap (Step 2.5) to avoid double-initialization issues.

    EventBus and invoke_hook are COEXISTING systems:
    - EventBus: Internal analytics for structured event emission
    - invoke_hook: External plugin system for third-party extensibility
    They do NOT wrap or replace each other.
"""

import threading
from typing import Dict, List, Callable, Any, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Optional telemetry imports
try:
    from agent.hermes.telemetry import TelemetryPipeline
except ImportError:
    TelemetryPipeline = None

logger = logging.getLogger(__name__)


# Built-in event types
class EventType:
    """Standard event types for Hermes-Agent analytics."""
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    LLM_CALL = "llm.call"
    LLM_RESPONSE = "llm.response"
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    ERROR = "error"
    SEMANTIC_SEARCH = "semantic.search"
    SEMANTIC_ADD = "semantic.add"
    SEMANTIC_DELETE = "semantic.delete"
    LOG_RECORD = "log.record"
    METRICS_SAMPLE = "metrics.sample"
    ALERT_TRIGGERED = "alert.triggered"


# All built-in event types as a set for validation
BUILT_IN_EVENTS: Set[str] = {
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.LLM_CALL,
    EventType.LLM_RESPONSE,
    EventType.SESSION_START,
    EventType.SESSION_END,
    EventType.ERROR,
    EventType.SEMANTIC_SEARCH,
    EventType.SEMANTIC_ADD,
    EventType.SEMANTIC_DELETE,
    EventType.METRICS_SAMPLE,
    EventType.ALERT_TRIGGERED,
}


@dataclass
class Event:
    """
    Represents an analytics event.

    Attributes:
        type: Event type string (e.g., "tool.call", "session.start")
        payload: Dictionary of event data
        timestamp: When the event occurred (UTC)
        session_id: Session identifier (if applicable)
    """
    type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: str = ""

    def __post_init__(self):
        """Validate event type against built-in events."""
        if self.type not in BUILT_IN_EVENTS:
            logger.debug(f"Custom event type: {self.type}")


class EventBus:
    """
    Thread-safe event bus for analytics.

    Events are delivered synchronously to all subscribed handlers.
    Handlers must NOT block - they are called in the emitting thread.

    Thread safety:
        Uses threading.RLock for handler registration and list cloning.
        Concurrent emit() calls are safe.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._emitting = False  # Guard against re-entrant emit
        self._telemetry_pipeline: Optional[TelemetryPipeline] = None

    def subscribe(self, event_type: str, handler: Callable[["Event"], None]) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: The event type to subscribe to
            handler: Callback function that receives the Event
        """
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)
            logger.debug(f"Handler subscribed to: {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable[["Event"], None]) -> None:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: The event type to unsubscribe from
            handler: The handler function to remove
        """
        with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers.get(event_type, [])
                    if h != handler
                ]
                logger.debug(f"Handler unsubscribed from: {event_type}")

    def add_backend(self, backend: "TelemetryBackend") -> None:
        """
        Add a telemetry backend to the pipeline.

        Args:
            backend: A TelemetryBackend implementation
        """
        if TelemetryPipeline is None:
            return
        if self._telemetry_pipeline is None:
            self._telemetry_pipeline = TelemetryPipeline()
        self._telemetry_pipeline.add_backend(backend)

    def emit_telemetry(self, event: Event) -> None:
        """
        Emit an event to the telemetry pipeline (fire-and-forget).

        Args:
            event: The Event to emit
        """
        if self._telemetry_pipeline is None:
            return
        try:
            self._telemetry_pipeline.emit(event)
        except Exception as e:
            logger.warning(f"emit_telemetry failed: {e}")

    def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribed handlers.

        Delivery is SYNCHRONOUS - all handlers are called in this thread.
        Handlers must NOT block, as this would block the emitter.

        Exceptions in handlers are caught and logged but do not propagate.

        Args:
            event: The Event to emit
        """
        with self._lock:
            handlers = list(self._handlers.get(event.type, []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Event handler failed for {event.type}: {e}")

        # Also emit to telemetry pipeline (fire-and-forget)
        self.emit_telemetry(event)

    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        session_id: str = ""
    ) -> None:
        """
        Convenience method to emit an event by type and payload.

        Args:
            event_type: The event type string
            payload: Dictionary of event data
            session_id: Optional session identifier
        """
        event = Event(type=event_type, payload=payload, session_id=session_id)
        self.emit(event)

    def clear_handlers(self, event_type: str = None) -> None:
        """
        Clear handlers for a specific event type, or all handlers if None.

        Args:
            event_type: Event type to clear (or None to clear all)
        """
        with self._lock:
            if event_type is None:
                self._handlers.clear()
            elif event_type in self._handlers:
                self._handlers[event_type].clear()

    def get_handler_count(self, event_type: str = None) -> int:
        """
        Get the number of handlers registered.

        Args:
            event_type: Event type to count (or None for all)

        Returns:
            Number of handlers registered
        """
        with self._lock:
            if event_type is None:
                return sum(len(h) for h in self._handlers.values())
            return len(self._handlers.get(event_type, []))
