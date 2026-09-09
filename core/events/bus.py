"""
Event Bus for Hermes Core events.
Enables real-time event broadcasting to UI clients, protocol layer, and loggers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    AGENT_STARTED = "AgentStarted"
    AGENT_THINKING = "AgentThinking"
    AGENT_TOOL_CALL = "AgentToolCall"
    TOOL_STARTED = "ToolStarted"
    TOOL_OUTPUT = "ToolOutput"
    TOOL_ERROR = "ToolError"
    TASK_CREATED = "TaskCreated"
    TASK_COMPLETED = "TaskCompleted"
    MESSAGE_RECEIVED = "MessageReceived"
    MESSAGE_SENT = "MessageSent"
    MEMORY_UPDATED = "MemoryUpdated"
    SKILL_CREATED = "SkillCreated"
    MODEL_CHANGED = "ModelChanged"
    CONNECTION_CHANGED = "ConnectionChanged"


@dataclass
class AgentEvent:
    event_type: EventType
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "agent.event",
            "session_id": self.session_id,
            "event": self.event_type.value if isinstance(self.event_type, Enum) else str(self.event_type),
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


EventHandler = Callable[[AgentEvent], None]


class EventBus:
    """Central event bus for publishing and subscribing to agent events."""

    _instance: Optional["EventBus"] = None

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[EventHandler]] = {
            event_type: [] for event_type in EventType
        }
        self._global_handlers: List[EventHandler] = []
        self._history: List[AgentEvent] = []
        self._max_history = 1000

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def publish(self, event: AgentEvent) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Notify specific handlers
        event_handlers = self._handlers.get(event.event_type, [])
        for handler in event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error executing event handler for {event.event_type}: {e}")

        # Notify global handlers
        for handler in self._global_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error executing global event handler: {e}")

    def get_history(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> List[AgentEvent]:
        events = self._history
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        return events[-limit:]

    def clear(self) -> None:
        self._history.clear()
