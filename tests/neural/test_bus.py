from neural.bus.in_process import NeuralBus
from neural.events.types import NeuralEvent


def test_publish_delivers_to_matching_subscribers():
    bus = NeuralBus()
    received = []
    bus.subscribe("tool.completed", received.append)

    event = NeuralEvent(source="tool", event_type="tool.completed")
    bus.publish(event)

    assert received == [event]


def test_unsubscribe_stops_delivery():
    bus = NeuralBus()
    received = []
    unsubscribe = bus.subscribe("tool.completed", received.append)

    unsubscribe()
    bus.publish(NeuralEvent(source="tool", event_type="tool.completed"))

    assert received == []


def test_handler_failure_does_not_break_other_handlers():
    bus = NeuralBus()
    received = []

    def broken(_event):
        raise RuntimeError("observer failed")

    bus.subscribe("tool.completed", broken)
    bus.subscribe("tool.completed", received.append)

    event = NeuralEvent(source="tool", event_type="tool.completed")
    bus.publish(event)

    assert received == [event]
