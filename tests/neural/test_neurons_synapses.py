from neural.events.types import NeuralEvent, NeuralSignal
from neural.neurons.base import Neuron
from neural.synapses.model import Synapse


def test_neuron_emits_a_signal_when_activated():
    neuron = Neuron(name="risk")
    event = NeuralEvent(source="tool", event_type="tool.completed")

    signal = neuron.activate(event, value=0.8)

    assert isinstance(signal, NeuralSignal)
    assert signal.source == "risk"
    assert signal.value == 0.8
    assert signal.correlation_id == event.correlation_id


def test_synapse_reinforcement_is_clamped_and_tracks_outcomes():
    synapse = Synapse(source="planner", target="tool", weight=0.95)

    synapse.reinforce(success=True, amount=0.2)
    synapse.reinforce(success=False, amount=0.5)

    assert synapse.weight == 0.5
    assert synapse.uses == 2
    assert synapse.successes == 1
    assert synapse.failures == 1
