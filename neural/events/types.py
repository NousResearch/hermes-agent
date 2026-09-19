"""Core immutable data structures for WIDDX neural communication."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


def _unit_interval(value: float, field_name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return value


@dataclass(frozen=True, slots=True)
class NeuralEvent:
    """An observation flowing through the neural bus.

    Events are immutable so consumers cannot silently alter history after it
    has been published. Payloads are copied at construction time so each event
    retains an isolated snapshot of caller-owned data.
    """

    source: str
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    confidence: float = 1.0
    id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must not be empty")
        if not self.event_type:
            raise ValueError("event_type must not be empty")
        object.__setattr__(self, "payload", deepcopy(self.payload))
        object.__setattr__(self, "importance", _unit_interval(self.importance, "importance"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))


@dataclass(frozen=True, slots=True)
class NeuralSignal:
    """A bounded activation signal emitted by a neural component."""

    source: str
    target: str
    value: float
    confidence: float = 1.0
    reason: str | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _unit_interval(self.value, "value"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
