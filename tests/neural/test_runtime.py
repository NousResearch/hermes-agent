from neural.events import NeuralEvent
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
