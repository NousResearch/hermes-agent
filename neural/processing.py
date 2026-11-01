"""Deterministic, advisory processing for neural observations."""

from __future__ import annotations

from collections.abc import Iterable

from neural.events import NeuralEvent, NeuralSignal
from neural.neurons import Neuron
from neural.synapses import Synapse


class NeuralProcessor:
    """Route events through registered neurons without execution authority."""

    def __init__(
        self,
        neurons: Iterable[Neuron] = (),
        synapses: Iterable[Synapse] = (),
    ) -> None:
        self._neurons = list(neurons)
        self._synapses = list(synapses)

    def register_neuron(self, neuron: Neuron) -> None:
        if any(existing.name == neuron.name for existing in self._neurons):
            raise ValueError(f"neuron already registered: {neuron.name}")
        self._neurons.append(neuron)

    def register_synapse(self, synapse: Synapse) -> None:
        self._synapses.append(synapse)

    def process(
        self,
        event: NeuralEvent,
        *,
        reinforce: bool = False,
        success: bool = True,
    ) -> list[NeuralSignal]:
        signals: list[NeuralSignal] = []
        for neuron in tuple(self._neurons):
            try:
                signal = neuron.activate(event, value=event.importance)
            except Exception:
                continue
            signals.append(signal)
            if reinforce:
                for synapse in self._synapses:
                    if synapse.source == event.source and synapse.target == neuron.name:
                        synapse.reinforce(success=success)
        return signals


__all__ = ["NeuralProcessor"]
