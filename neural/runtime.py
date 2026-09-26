"""Runtime coordinator connecting an agent session to the WIDDX neural layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from neural.brain import Brain
from neural.bus import NeuralBus
from neural.events import NeuralEvent, NeuralSignal
from neural.guardian import Guardian
from neural.memory import Memory
from neural.perception import SensoryCell
from neural.processing import NeuralProcessor
from neural.self_model import SelfModel


@dataclass(slots=True)
class NeuralRuntime:
    """Own the advisory neural components for one agent instance.

    The runtime is deliberately side-effect free with respect to tools and policy.
    Publishing an observation must never be required for the primary agent path to
    succeed; callers can use ``safe_observe`` at integration boundaries.
    """

    bus: NeuralBus = field(default_factory=NeuralBus)
    brain: Brain = field(default_factory=Brain)
    memory: Memory = field(default_factory=Memory)
    guardian: Guardian = field(default_factory=Guardian)
    self_model: SelfModel = field(default_factory=SelfModel)
    processor: NeuralProcessor = field(default_factory=NeuralProcessor)

    def observe(
        self,
        source: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        importance: float = 0.5,
        confidence: float = 1.0,
        correlation_id: str | None = None,
    ) -> NeuralEvent:
        event = NeuralEvent(
            source=source,
            event_type=event_type,
            payload={} if payload is None else payload,
            importance=importance,
            confidence=confidence,
            correlation_id=correlation_id,
        )
        self.bus.publish(event)
        return event

    def sense(
        self,
        cell: SensoryCell,
        payload: Mapping[str, Any],
        *,
        importance: float = 0.5,
        confidence: float = 1.0,
        correlation_id: str | None = None,
    ) -> NeuralEvent:
        """Normalize and publish an already-observed fact through a sensory cell."""
        event = cell.observe(
            payload,
            importance=importance,
            confidence=confidence,
            correlation_id=correlation_id,
        )
        self.bus.publish(event)
        return event

    def process(
        self,
        event: NeuralEvent,
        *,
        reinforce: bool = False,
        success: bool = True,
    ) -> list[NeuralSignal]:
        """Process an observation through advisory neurons only."""
        return self.processor.process(event, reinforce=reinforce, success=success)

    def safe_observe(self, *args: Any, **kwargs: Any) -> NeuralEvent | None:
        """Best-effort observation boundary; never raises into the agent path."""
        try:
            return self.observe(*args, **kwargs)
        except Exception:
            return None


__all__ = ["NeuralRuntime"]
