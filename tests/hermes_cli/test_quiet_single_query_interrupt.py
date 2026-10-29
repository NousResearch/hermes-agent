"""``hermes chat -Q`` says why the turn ended when it is interrupted (#84207).

A quiet one-shot run that is interrupted mid-turn has no result record to carry the reason:
the process exits 130 with a bare session-id line and the user is left with no indication that
the turn was cut short rather than answered. The notice goes to stderr so stdout stays
byte-identical for the reply stream.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.cli_single_query import _run_quiet_single_query


@pytest.fixture(autouse=True)
def _plain_one_shot_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)


def _interrupt_during_turn(**_kwargs):
    raise KeyboardInterrupt


def _run_interrupted():
    agent = SimpleNamespace(run_conversation=_interrupt_during_turn, session_id="s-1")
    with pytest.raises(SystemExit) as exc:
        _run_quiet_single_query(
            SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1"), "hello")
    return exc.value.code


def test_interrupted_quiet_one_shot_tells_the_user_on_stderr(capsys):
    """The turn ended with no reply: the user must be told, not left with a bare exit code."""
    assert _run_interrupted() == 130
    captured = capsys.readouterr()
    assert "Turn interrupted." in captured.err
    # stdout is the reply stream: an interrupted run must not add anything to it.
    assert captured.out == ""


def test_interrupted_quiet_one_shot_still_reports_the_session_id(capsys):
    """The session-id line is the automation contract on stderr; the notice must not replace it."""
    _run_interrupted()
    assert "session_id: s-1" in capsys.readouterr().err
