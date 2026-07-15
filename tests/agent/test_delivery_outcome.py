import json
from types import SimpleNamespace

import pytest

from agent.delivery_outcome import (
    get_delivery_outcome,
    record_delivery_outcome,
    reset_delivery_outcome_turn,
)
from tools.todo_tool import TODO_SCHEMA, TodoStore, todo_tool


def _agent(turn_id="turn-1"):
    return SimpleNamespace(_current_turn_id=turn_id)


def test_todo_schema_exposes_exact_model_authored_protocol():
    schema = TODO_SCHEMA["parameters"]["properties"]["delivery_outcome"]
    assert schema["required"] == ["action", "reason"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["action"]["enum"] == ["deliver", "suppress"]


def test_record_and_read_are_bound_to_exact_turn():
    agent = _agent()
    reset_delivery_outcome_turn(agent, "turn-1")
    receipt = record_delivery_outcome(
        agent,
        {"action": "suppress", "reason": "nothing new"},
        originating_turn_id="turn-1",
    )

    assert receipt == {
        "recorded": True,
        "action": "suppress",
        "turn_id": "turn-1",
    }
    assert get_delivery_outcome(agent, "turn-1") == {
        "action": "suppress",
        "reason": "nothing new",
        "turn_id": "turn-1",
    }
    assert get_delivery_outcome(agent, "turn-2") is None


def test_new_turn_reset_clears_prior_outcome_and_rejects_late_worker():
    agent = _agent()
    reset_delivery_outcome_turn(agent, "turn-1")
    record_delivery_outcome(
        agent,
        {"action": "suppress", "reason": "nothing new"},
        originating_turn_id="turn-1",
    )

    agent._current_turn_id = "turn-2"
    reset_delivery_outcome_turn(agent, "turn-2")
    assert get_delivery_outcome(agent, "turn-2") is None
    with pytest.raises(ValueError, match="stale"):
        record_delivery_outcome(
            agent,
            {"action": "suppress", "reason": "late result"},
            originating_turn_id="turn-1",
        )


def test_later_model_choice_in_same_turn_replaces_earlier_choice():
    agent = _agent()
    reset_delivery_outcome_turn(agent, "turn-1")
    record_delivery_outcome(
        agent,
        {"action": "suppress", "reason": "initially nothing new"},
        originating_turn_id="turn-1",
    )
    record_delivery_outcome(
        agent,
        {"action": "deliver", "reason": "a report became available"},
        originating_turn_id="turn-1",
    )

    assert get_delivery_outcome(agent, "turn-1")["action"] == "deliver"


@pytest.mark.parametrize(
    "directive",
    [
        {"action": "hide", "reason": "no"},
        {"action": "suppress", "reason": ""},
        {"action": "suppress", "reason": "ok", "extra": True},
        ["suppress", "reason"],
    ],
)
def test_invalid_directives_fail_closed_for_suppression(directive):
    agent = _agent()
    reset_delivery_outcome_turn(agent, "turn-1")
    with pytest.raises(ValueError):
        record_delivery_outcome(
            agent,
            directive,
            originating_turn_id="turn-1",
        )
    assert get_delivery_outcome(agent, "turn-1") is None


def test_todo_returns_structured_receipt_without_changing_todo_semantics():
    agent = _agent()
    reset_delivery_outcome_turn(agent, "turn-1")
    result = json.loads(
        todo_tool(
            store=TodoStore(),
            delivery_outcome={"action": "deliver", "reason": "report ready"},
            delivery_outcome_recorder=lambda directive: record_delivery_outcome(
                agent,
                directive,
                originating_turn_id="turn-1",
            ),
        )
    )

    assert result["todos"] == []
    assert result["delivery_outcome"] == {
        "recorded": True,
        "action": "deliver",
        "turn_id": "turn-1",
    }
