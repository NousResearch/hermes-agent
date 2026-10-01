"""Regression coverage for #110173: observational `hermes sessions` readers stay read-only."""

from argparse import Namespace


import hermes_cli.sessions_cmd as sessions_cmd




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


def test_doctor_write_probe_never_touches_a_store_a_live_writer_holds(monkeypatch, tmp_path):
    """The write probe runs against a read-only snapshot when a gateway holds state.db, in place otherwise."""
    import sqlite3

    import hermes_state_holders
    import hermes_state_repair
    from hermes_cli import doctor_state

    state_db = tmp_path / "state.db"
    sqlite3.connect(state_db).execute("CREATE TABLE sessions (id TEXT)").connection.close()
    probed: list = []
    monkeypatch.setattr(hermes_state_repair, "_db_opens_cleanly", lambda path: probed.append(path))

    monkeypatch.setattr(hermes_state_holders, "live_writer_holds_db", lambda *_a, **_k: True)
    assert doctor_state._write_health_reason(state_db, should_fix=False) is None
    monkeypatch.setattr(hermes_state_holders, "live_writer_holds_db", lambda *_a, **_k: False)
    assert doctor_state._write_health_reason(state_db, should_fix=False) is None

    assert probed[0] != state_db and probed[0].name == "state.db"
    assert probed[1] == state_db
