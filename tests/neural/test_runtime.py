from neural.events import NeuralEvent
from neural.neurons import Neuron
from neural.perception import ConversationCell
from neural.processing import NeuralProcessor
from neural.runtime import NeuralRuntime


def test_runtime_publishes_observations_and_preserves_correlation():
    runtime = NeuralRuntime()
    received = []
    runtime.bus.subscribe("session.started", received.append)

    event = runtime.observe(
        "agent",
        "session.started",
        {"session_id": "abc"},
        correlation_id="turn-1",
    )

    assert isinstance(event, NeuralEvent)
    assert received == [event]
    assert event.correlation_id == "turn-1"
    assert event.payload["session_id"] == "abc"


def test_safe_observe_does_not_escape_observer_failures():
    runtime = NeuralRuntime()
    runtime.bus.subscribe("tool.completed", lambda _event: (_ for _ in ()).throw(RuntimeError("boom")))

    event = runtime.safe_observe("tool", "tool.completed")

    assert event is not None


def test_runtime_exposes_advisory_neural_processing():
    processor = NeuralProcessor([Neuron("priority")])
    runtime = NeuralRuntime(processor=processor)

    signals = runtime.process(
        NeuralEvent(source="task", event_type="task.observed", payload={"id": "1"}, importance=0.6)
    )

    assert len(signals) == 1
    assert signals[0].value == 0.6


def test_runtime_can_publish_a_sensory_cell_observation():
    runtime = NeuralRuntime()
    received = []
    runtime.bus.subscribe("conversation.observed", received.append)

    event = runtime.sense(ConversationCell(), {"text": "hello"}, correlation_id="turn-2")

    assert event in received
    assert event.source == "conversation"
    assert event.event_type == "conversation.observed"
    assert event.correlation_id == "turn-2"
