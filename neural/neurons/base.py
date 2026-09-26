"""Base neuron behavior."""

from __future__ import annotations

from dataclasses import dataclass

from neural.events.types import NeuralEvent, NeuralSignal


@dataclass(slots=True)
class Neuron:
    """A named deterministic processing unit.

    Domain-specific neurons can subclass this later. The base implementation
    only transforms an already-bounded activation into a signal; it does not
    execute tools or alter policy.
    """

    name: str

    def activate(self, event: NeuralEvent, *, value: float) -> NeuralSignal:
        return NeuralSignal(
            source=self.name,
            target=self.name,
            value=value,
            confidence=event.confidence,
            correlation_id=event.correlation_id,
        )
