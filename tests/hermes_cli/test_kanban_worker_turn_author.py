"""A dispatcher-authored ``chat -q`` run carries its turn author, as ``chat -Q`` already does.

``HERMES_TURN_AUTHOR`` names who wrote a one-shot run's prompts. The non-quiet ``-q`` path ignored it,
so the turn reached memory providers as the operator's and tool subprocesses inherited the variable.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cli
from agent.turn_author import TURN_AUTHOR_ENV


def _one_shot_cli(monkeypatch):
    """A real HermesCLI whose ``chat`` runs the real agent-thread body against a recording agent."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)

    calls = []

    def run_conversation(**kwargs):
        calls.append({**kwargs, "env_author": os.environ.get(TURN_AUTHOR_ENV)})
        return {"final_response": "done"}

    the_cli = cli.HermesCLI.__new__(cli.HermesCLI)
    the_cli.session_id = "s-1"
    the_cli.conversation_history = [{"role": "user", "content": "work kanban task t_author"}]
    the_cli.agent = SimpleNamespace(run_conversation=Mock(side_effect=run_conversation))
    the_cli._flush_credit_notices = lambda: None
    the_cli._single_query_mode = False
    the_cli._claim_active_session = lambda *a, **k: True
    the_cli.console = SimpleNamespace(print=lambda *a, **k: None)
    the_cli._show_security_advisories = lambda: None
    the_cli._print_exit_summary = lambda **k: None
    the_cli._last_turn_result = {"failed": False}

    def chat(message, images=None):
        the_cli._chat_run_agent(cli._ChatTurn(), message)
        return "done"

    the_cli.chat = chat
    return the_cli, calls


def test_dispatcher_author_reaches_the_worker_turn_and_is_consumed(monkeypatch):
    monkeypatch.setenv(TURN_AUTHOR_ENV, '{"id": "bot:kanban", "name": "Kanban dispatcher", "is_bot": true}')
    the_cli, calls = _one_shot_cli(monkeypatch)
    with pytest.raises(SystemExit):
        cli._run_single_query_mode(the_cli, "work kanban task t_author", None, False, True)

    assert len(calls) == 1
    assert calls[0]["turn_author"] == {"id": "bot:kanban", "name": "Kanban dispatcher", "is_bot": True}
    # Tool subprocesses started during the turn must not inherit the dispatcher's author.
    assert calls[0]["env_author"] is None


def test_human_one_shot_keeps_an_unattributed_turn(monkeypatch):
    monkeypatch.delenv(TURN_AUTHOR_ENV, raising=False)
    the_cli, calls = _one_shot_cli(monkeypatch)
    with pytest.raises(SystemExit):
        cli._run_single_query_mode(the_cli, "hello", None, False, True)

    assert len(calls) == 1
    assert "turn_author" not in calls[0]
