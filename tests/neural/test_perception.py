import pytest

from neural.events import NeuralEvent
from neural.perception.cells import (
    ConversationCell,
    EnvironmentCell,
    ErrorCell,
    SensoryCell,
    TaskCell,
    TerminalCell,
)


@pytest.mark.parametrize(
    ("cell", "event_type"),
    [
        (ConversationCell(), "conversation.observed"),
        (TaskCell(), "task.observed"),
        (TerminalCell(), "terminal.observed"),
        (ErrorCell(), "error.observed"),
        (EnvironmentCell(), "environment.observed"),
    ],
)
def test_sensory_cells_emit_normalized_events(cell, event_type):
    event = cell.observe({"value": "observed"}, confidence=0.8, importance=0.7)

    assert isinstance(cell, SensoryCell)
    assert isinstance(event, NeuralEvent)
    assert event.source == cell.name
    assert event.event_type == event_type
    assert event.payload == {"value": "observed"}
    assert event.confidence == 0.8
    assert event.importance == 0.7


def test_sensory_cell_rejects_empty_observation():
    with pytest.raises(ValueError, match="payload"):
        ConversationCell().observe({})


def test_sensory_cell_payload_isolated_from_caller():
    payload = {"nested": {"items": ["initial"]}}
    event = TerminalCell().observe(payload)

    payload["nested"]["items"].append("mutated")

    assert event.payload == {"nested": {"items": ["initial"]}}
