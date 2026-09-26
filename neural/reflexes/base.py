"""Reflex primitives that cannot grant permission."""

from __future__ import annotations

from dataclasses import dataclass

from neural.events.types import NeuralEvent, NeuralSignal


@dataclass(frozen=True, slots=True)
class Reflex:
    name: str

    def trigger(self, event: NeuralEvent) -> NeuralSignal:
        return NeuralSignal(
            source=f"reflex:{self.name}",
            target=self.name,
            value=1.0,
            confidence=event.confidence,
            reason=event.event_type,
            correlation_id=event.correlation_id,
        )
