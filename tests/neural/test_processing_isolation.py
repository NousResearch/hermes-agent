from neural.events import NeuralEvent, NeuralSignal
from neural.neurons import Neuron
from neural.processing import NeuralProcessor


class FailingNeuron(Neuron):
    def activate(self, event: NeuralEvent, *, value: float) -> NeuralSignal:
        raise RuntimeError("neural observer failure")


def test_processor_isolates_a_failing_neuron():
    processor = NeuralProcessor([FailingNeuron("broken"), Neuron("healthy")])

    signals = processor.process(
        NeuralEvent(source="task", event_type="task.observed", payload={"id": "1"})
    )

    assert [signal.source for signal in signals] == ["healthy"]
