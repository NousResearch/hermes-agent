"""Small, failure-isolated publish/subscribe bus for neural observations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from threading import RLock
from typing import DefaultDict

from neural.events.types import NeuralEvent

Handler = Callable[[NeuralEvent], None]


class NeuralBus:
    """Deliver neural events synchronously without coupling observers."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[str, list[Handler]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_type: str, handler: Handler) -> Callable[[], None]:
        if not event_type:
            raise ValueError("event_type must not be empty")
        with self._lock:
            self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(event_type)
                if handlers and handler in handlers:
                    handlers.remove(handler)
                if not handlers:
                    self._handlers.pop(event_type, None)

        return unsubscribe

    def publish(self, event: NeuralEvent) -> None:
        with self._lock:
            handlers = tuple(self._handlers.get(event.event_type, ()))

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # Neural observers are advisory. An observer must never take
                # down the agent's primary execution path.
                continue
