"""Regression coverage for #110173: observational `hermes sessions` readers stay read-only."""

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

import hermes_cli.sessions_cmd as sessions_cmd


@pytest.mark.parametrize("action", ["list", "stats", "pinned"])
def test_observational_sessions_actions_open_a_read_only_store(monkeypatch, action):
    factory = MagicMock()
    monkeypatch.setattr("hermes_state.SessionDB", factory)
    monkeypatch.setitem(sessions_cmd._DB_HANDLERS, action, lambda _db, _args: None)

    sessions_cmd.cmd_sessions(Namespace(sessions_action=action))

    factory.assert_called_once_with(read_only=True)


def test_sessions_observational_commands_on_missing_store_stay_empty(monkeypatch, tmp_path, capsys):
    """Fresh profile: list/stats/pinned report empty without creating a writable store."""
    import hermes_state

    db_path = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)

    list_args = Namespace(sessions_action="list", source=None, limit=20, workspace=None)
    assert sessions_cmd.cmd_sessions(list_args) is None
    assert "No sessions found." in capsys.readouterr().out

    assert sessions_cmd.cmd_sessions(Namespace(sessions_action="stats")) is None
    stats_out = capsys.readouterr().out
    assert "Total sessions: 0" in stats_out
    assert "Total messages: 0" in stats_out

    pinned_args = Namespace(sessions_action="pinned", source=None, json=False)
    assert sessions_cmd.cmd_sessions(pinned_args) is None
    assert "No pinned sessions" in capsys.readouterr().out
    assert not db_path.exists()
