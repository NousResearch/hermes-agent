from collections import defaultdict
from typing import Callable, Any
import uuid
from eclose.events.events import EventType, PerceptionEvent

Subscriber = Callable[[Any], None]


class EventBus:
    def __init__(self):
        self._subscribers: dict[EventType, list[tuple[str, Subscriber]]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Subscriber) -> str:
        """Subscribe to an event type. Returns subscription token."""
        token = str(uuid.uuid4())
        self._subscribers[event_type].append((token, handler))
        return token

    def unsubscribe(self, token: str):
        """Unsubscribe by token."""
        for event_type, handlers in self._subscribers.items():
            self._subscribers[event_type] = [
                (t, h) for t, h in handlers if t != token
            ]

    def publish(self, event: PerceptionEvent):
        """Publish an event to all subscribers."""
        handlers = self._subscribers.get(event.type, [])
        for token, handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log but don't fail
                print(f"Handler error: {e}")

    def clear(self):
        """Clear all subscribers."""
        self._subscribers.clear()


# Global event bus instance
_global_event_bus = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus
