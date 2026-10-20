from neural.events import NeuralEvent, NeuralSignal
from neural.neurons import Neuron
from neural.processing import NeuralProcessor
from neural.synapses import Synapse


def test_processor_turns_event_into_bounded_signal():
    processor = NeuralProcessor()
    processor.register_neuron(Neuron("priority"))

    signals = processor.process(
        NeuralEvent(
            source="conversation",
            event_type="conversation.observed",
            payload={"text": "hello"},
            importance=0.8,
            confidence=0.9,
            correlation_id="turn-1",
        )
    )

    assert len(signals) == 1
    assert isinstance(signals[0], NeuralSignal)
    assert signals[0].value == 0.8
    assert signals[0].confidence == 0.9
    assert signals[0].correlation_id == "turn-1"


def test_processor_can_reinforce_matching_synapse():
    processor = NeuralProcessor()
    processor.register_neuron(Neuron("priority"))
    synapse = Synapse("conversation", "priority")
    processor.register_synapse(synapse)

    processor.process(
        NeuralEvent(source="conversation", event_type="conversation.observed", payload={"text": "hello"}),
        reinforce=True,
        success=True,
    )

    assert synapse.uses == 1
    assert synapse.successes == 1
    assert synapse.failures == 0
    assert synapse.weight == 0.6


def test_processor_is_deterministic_without_learning():
    processor = NeuralProcessor([Neuron("priority")])
    event = NeuralEvent(source="task", event_type="task.observed", payload={"id": "1"}, importance=0.4)

    assert processor.process(event) == processor.process(event)
