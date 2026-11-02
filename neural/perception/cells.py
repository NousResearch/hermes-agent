"""Side-effect-free sensory cells for normalizing Hermes observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from neural.events import NeuralEvent


@dataclass(frozen=True, slots=True)
class SensoryCell:
    """A named adapter that converts an observed fact into a neural event."""

    name: str
    event_type: str

    def observe(
        self,
        payload: Mapping[str, Any],
        *,
        importance: float = 0.5,
        confidence: float = 1.0,
        correlation_id: str | None = None,
    ) -> NeuralEvent:
        if not self.name:
            raise ValueError("source must not be empty")
        if not self.event_type:
            raise ValueError("event_type must not be empty")
        if not payload:
            raise ValueError("payload must not be empty")
        return NeuralEvent(
            source=self.name,
            event_type=self.event_type,
            payload=payload,
            importance=importance,
            confidence=confidence,
            correlation_id=correlation_id,
        )


@dataclass(frozen=True, slots=True)
class ConversationCell(SensoryCell):
    name: str = "conversation"
    event_type: str = "conversation.observed"


@dataclass(frozen=True, slots=True)
class TaskCell(SensoryCell):
    name: str = "task"
    event_type: str = "task.observed"


@dataclass(frozen=True, slots=True)
class TerminalCell(SensoryCell):
    name: str = "terminal"
    event_type: str = "terminal.observed"


@dataclass(frozen=True, slots=True)
class ErrorCell(SensoryCell):
    name: str = "error"
    event_type: str = "error.observed"


@dataclass(frozen=True, slots=True)
class EnvironmentCell(SensoryCell):
    name: str = "environment"
    event_type: str = "environment.observed"


__all__ = [
    "SensoryCell",
    "ConversationCell",
    "TaskCell",
    "TerminalCell",
    "ErrorCell",
    "EnvironmentCell",
]
