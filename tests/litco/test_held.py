"""Which results the held-for-approval hook rewrites, and when (litco/held.py)."""

from __future__ import annotations

import json

from litco.held import HELD_STATUS, held_tool_result
from litco.litkit.context import TurnIdentity, turn_scope

GATE_HOLD = json.dumps({"output": "", "exit_code": -1, "error": "", "status": "pending_approval",
                        "approval_pending": True, "command": "env | grep LITCO", "pattern_key": "env_dump"})


def test_outside_a_litco_turn_the_result_is_left_alone():
    assert held_tool_result(tool_name="terminal", result=GATE_HOLD) is None


def test_in_a_turn_the_gate_hold_is_rewritten_once():
    with turn_scope(TurnIdentity(turn_id="t1", matter_id="m1")):
        held = held_tool_result(tool_name="terminal", result=GATE_HOLD)
        assert json.loads(held)["status"] == HELD_STATUS
        assert json.loads(held)["command"] == "env | grep LITCO"
        assert held_tool_result(tool_name="terminal", result=held) is None  # already rewritten


def test_the_action_gate_hold_is_rewritten_too():
    action_gate = {"approved": False, "pattern_key": "rm_rf", "status": "approval_required",
                   "command": "rm -rf scratch", "description": "recursive delete", "message": "Asking the user"}
    with turn_scope(TurnIdentity(turn_id="t1")):
        assert json.loads(held_tool_result(tool_name="terminal", result=action_gate))["status"] == HELD_STATUS


def test_results_that_are_not_the_gates_are_left_alone():
    with turn_scope(TurnIdentity(turn_id="t1")):
        # a LitKit body with a similar status, but no gate rule behind it
        assert held_tool_result(tool_name="litkit_proposals", result='{"status": "pending_approval", "id": "p1"}') is None
        assert held_tool_result(tool_name="terminal", result='{"output": "", "exit_code": 1, "error": null}') is None
        assert held_tool_result(tool_name="terminal", result="plain text") is None
