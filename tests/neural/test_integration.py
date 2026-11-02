from neural.events import NeuralEvent
from neural.neurons import Neuron
from neural.perception import ConversationCell
from neural.processing import NeuralProcessor


def test_sensory_to_signal_path_is_execution_free():
    cell = ConversationCell()
    processor = NeuralProcessor([Neuron("priority")])

    event = cell.observe({"text": "hello"}, importance=0.75)
    signals = processor.process(event)

    assert isinstance(event, NeuralEvent)
    assert event.source == "conversation"
    assert signals[0].value == 0.75
    assert signals[0].source == "priority"
