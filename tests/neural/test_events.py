from dataclasses import FrozenInstanceError

import pytest

from neural.events.types import NeuralEvent, NeuralSignal


def test_neural_event_has_safe_defaults_and_identity():
    event = NeuralEvent(source="terminal", event_type="command.completed", payload={"exit_code": 0})

    assert event.source == "terminal"
    assert event.event_type == "command.completed"
    assert event.payload == {"exit_code": 0}
    assert 0.0 <= event.confidence <= 1.0
    assert 0.0 <= event.importance <= 1.0
    assert event.id
    assert event.timestamp


def test_neural_event_is_immutable():
    event = NeuralEvent(source="test", event_type="observation")

    with pytest.raises(FrozenInstanceError):
        event.source = "other"


def test_neural_event_payload_is_isolated_from_caller_mutation():
    payload = {"nested": {"items": ["initial"]}}
    event = NeuralEvent(source="test", event_type="observation", payload=payload)

    payload["nested"]["items"].append("mutated")
    payload["nested"] = {"items": ["replaced"]}

    assert event.payload == {"nested": {"items": ["initial"]}}


def test_neural_signal_has_bounded_value_and_confidence():
    signal = NeuralSignal(source="risk", target="guardian", value=0.75, confidence=0.9)

    assert signal.source == "risk"
    assert signal.target == "guardian"
    assert signal.value == 0.75
    assert signal.confidence == 0.9


def test_neural_signal_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        NeuralSignal(source="a", target="b", value=1.1, confidence=0.5)

    with pytest.raises(ValueError):
        NeuralSignal(source="a", target="b", value=0.5, confidence=-0.1)
