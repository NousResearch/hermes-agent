"""Bounded synaptic connection state for future learning."""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class Synapse:
    """A bounded connection whose learning state is advisory only."""

    source: str
    target: str
    weight: float = 0.5
    confidence: float = 1.0
    uses: int = 0
    successes: int = 0
    failures: int = 0

    def __post_init__(self) -> None:
        self.weight = _clamp(float(self.weight))
        self.confidence = _clamp(float(self.confidence))
        if self.uses < 0 or self.successes < 0 or self.failures < 0:
            raise ValueError("synapse counters must not be negative")

    def reinforce(self, *, success: bool, amount: float = 0.1) -> None:
        amount = abs(float(amount))
        self.uses += 1
        if success:
            self.successes += 1
            self.weight = _clamp(self.weight + amount)
        else:
            self.failures += 1
            self.weight = _clamp(self.weight - amount)

        self.confidence = _clamp(self.successes / self.uses)
