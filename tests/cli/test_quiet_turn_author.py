"""``hermes chat -Q`` passes the dispatcher's HERMES_TURN_AUTHOR to ``run_conversation`` as ``turn_author``.

A bot-to-bot delivery runs the recipient's turn as a ``-Q`` subprocess with that variable set.
A human's ``-Q`` run has it unset and the turn stays unattributed.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import cli
from agent.turn_author import TURN_AUTHOR_ENV

AUTHOR = {"id": "bot:coder", "name": "coder", "is_bot": True}


@pytest.fixture(autouse=True)
def _plain_one_shot_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)


def _run(monkeypatch, env_value, run_conversation=None):
    """One quiet turn with HERMES_TURN_AUTHOR set to ``env_value`` (unset for None); returns the recorded call kwargs."""
    if env_value is None:
        monkeypatch.delenv(TURN_AUTHOR_ENV, raising=False)
    else:
        monkeypatch.setenv(TURN_AUTHOR_ENV, env_value)
    recorded = []

    def record(**kwargs):
        recorded.append(kwargs)
        return {"final_response": "ok"}

    agent = SimpleNamespace(run_conversation=run_conversation or record, session_id="s-1")
    with pytest.raises(SystemExit) as exc:
        cli._run_quiet_single_query(SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1"), "hello")
    assert exc.value.code == 0
    return recorded


def test_quiet_one_shot_passes_turn_author_from_env(monkeypatch, capsys):
    recorded = _run(monkeypatch, json.dumps(AUTHOR))
    assert recorded == [{"user_message": "hello", "conversation_history": [], "turn_author": AUTHOR}]
    assert capsys.readouterr().out.strip() == "ok"


def test_quiet_one_shot_consumes_the_variable_before_the_turn(monkeypatch):
    """Tool subprocesses spawned during the turn must not see the dispatcher's author."""
    seen = {}

    def run_conversation(**kwargs):
        seen["env"] = os.environ.get(TURN_AUTHOR_ENV)
        return {"final_response": "ok"}

    _run(monkeypatch, json.dumps(AUTHOR), run_conversation)
    assert seen["env"] is None
    assert TURN_AUTHOR_ENV not in os.environ
