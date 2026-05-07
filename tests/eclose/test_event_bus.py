import pytest
from eclose.events.event_bus import EventBus
from eclose.events.events import (
    PerceptionEvent,
    EventType,
    PerceptionSource,
)

def test_event_bus_subscribe_and_publish():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe(EventType.PERCEPTION, handler)
    event = PerceptionEvent(
        source=PerceptionSource.PROJECT,
        data={"test": True},
    )
    bus.publish(event)

    assert len(received) == 1
    assert received[0].data["test"] is True


def test_event_bus_unsubscribe():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    token = bus.subscribe(EventType.PERCEPTION, handler)
    bus.unsubscribe(token)

    bus.publish(PerceptionEvent(source=PerceptionSource.PROJECT, data={}))
    assert len(received) == 0
