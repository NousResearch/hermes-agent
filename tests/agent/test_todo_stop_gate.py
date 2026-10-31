"""When the open-todo stop gate nudges and when a stop is an intentional pause (salvage of #80155)."""

import json
from types import SimpleNamespace

import pytest

from agent.todo_stop_gate import build_todo_stop_nudge
from tools.todo_tool import TodoStore

OPEN = [{"id": "s4-ci", "content": "Commit, push, and verify CI", "status": "in_progress"}]
TURN = [{"role": "user", "content": "finish the release"}]


@pytest.fixture
def gate_on(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("agent:\n  todo_stop_gate: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))


def _agent(todos, nudges=0):
    store = TodoStore()
    store.write(todos)
    return SimpleNamespace(_todo_store=store, _todo_stop_nudges=nudges)


def test_open_items_are_named_in_the_nudge(gate_on):
    nudge = build_todo_stop_nudge(_agent(OPEN), "Need commit.", TURN)
    assert nudge and "s4-ci" in nudge


def test_settled_list_and_spent_nudge_do_not_fire(gate_on):
    settled = [{"id": "a", "content": "x", "status": "completed"}, {"id": "b", "content": "y", "status": "cancelled"}]
    assert build_todo_stop_nudge(_agent(settled), "Done.", TURN) is None
    assert build_todo_stop_nudge(_agent(OPEN, nudges=1), "Need commit.", TURN) is None


@pytest.mark.parametrize("reply", [
    "I need your approval before pushing the release branch.",
    "需要你的确认才能推送到生产分支。",
])
def test_a_question_to_the_user_is_a_valid_stop(gate_on, reply):
    assert build_todo_stop_nudge(_agent(OPEN), reply, TURN) is None


def test_notifying_background_work_this_turn_is_a_valid_stop_but_an_earlier_turns_is_not(gate_on):
    started = {"role": "tool", "tool_call_id": "c1",
               "content": json.dumps({"output": "Background process started", "notify_on_complete": True})}
    assert build_todo_stop_nudge(_agent(OPEN), "CI is running.", [*TURN, started]) is None
    assert build_todo_stop_nudge(_agent(OPEN), "Need commit.", [started, *TURN]) is not None


def test_legacy_default_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    assert build_todo_stop_nudge(_agent(OPEN), "Need commit.", TURN) is None
