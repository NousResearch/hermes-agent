"""Adapters for turning runtime observations into neural events."""

from neural.events import NeuralEvent
from neural.perception.cells import (
    ConversationCell,
    EnvironmentCell,
    ErrorCell,
    SensoryCell,
    TaskCell,
    TerminalCell,
)


def observe(source: str, event_type: str, payload=None, *, importance=0.5, confidence=1.0) -> NeuralEvent:
    """Create a normalized neural observation without executing side effects."""
    return NeuralEvent(
        source=source,
        event_type=event_type,
        payload={} if payload is None else payload,
        importance=importance,
        confidence=confidence,
    )


__all__ = [
    "observe",
    "NeuralEvent",
    "SensoryCell",
    "ConversationCell",
    "TaskCell",
    "TerminalCell",
    "ErrorCell",
    "EnvironmentCell",
]
