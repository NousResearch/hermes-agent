"""Regression coverage for #110173: observational CLI database readers stay read-only."""

from argparse import Namespace
from unittest.mock import MagicMock, call

import hermes_cli.sessions_cmd as sessions_cmd


def test_status_session_summary_opens_a_read_only_store(monkeypatch):
    from hermes_cli import status

    db = MagicMock()
    db.list_gateway_sessions.return_value = []
    factory = MagicMock(return_value=db)
    monkeypatch.setattr("hermes_state.SessionDB", factory)

    status._render_sessions(Namespace(config={}))

    factory.assert_called_once_with(read_only=True)
    db.close.assert_called_once()


def test_doctor_without_fix_counts_sessions_through_read_only_store(monkeypatch, tmp_path):
    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_health

    db_path = tmp_path / "state.db"
    db_path.touch()
    db = MagicMock()
    db.session_count.return_value = 3
    factory = MagicMock(return_value=db)
    monkeypatch.setattr("hermes_state.SessionDB", factory)
    monkeypatch.setattr("hermes_state_repair._db_opens_cleanly", lambda _path: None)

    _state_db_health(Finding(), False, db_path, "~/hermes")

    factory.assert_called_once_with(db_path=db_path, read_only=True)
    db.close.assert_called_once()


def test_doctor_with_fix_also_counts_through_read_only_store(monkeypatch, tmp_path):
    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_health

    db_path = tmp_path / "state.db"
    db_path.touch()
    db = MagicMock()
    db.session_count.return_value = 3
    factory = MagicMock(return_value=db)
    monkeypatch.setattr("hermes_state.SessionDB", factory)
    monkeypatch.setattr("hermes_state_repair._db_opens_cleanly", lambda _path: None)

    _state_db_health(Finding(), True, db_path, "~/hermes")

    factory.assert_called_once_with(db_path=db_path, read_only=True)
    db.close.assert_called_once()


def test_sessions_list_stats_and_pinned_open_a_read_only_store(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr("hermes_state.SessionDB", factory)
    monkeypatch.setitem(sessions_cmd._DB_HANDLERS, "list", lambda _db, _args: None)
    monkeypatch.setitem(sessions_cmd._DB_HANDLERS, "stats", lambda _db, _args: None)
    monkeypatch.setitem(sessions_cmd._DB_HANDLERS, "pinned", lambda _db, _args: None)

    sessions_cmd.cmd_sessions(Namespace(sessions_action="list"))
    sessions_cmd.cmd_sessions(Namespace(sessions_action="stats"))
    sessions_cmd.cmd_sessions(Namespace(sessions_action="pinned"))

    assert factory.call_args_list == [call(read_only=True), call(read_only=True), call(read_only=True)]


def test_mutating_sessions_action_keeps_a_writable_store(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr("hermes_state.SessionDB", factory)
    monkeypatch.setitem(sessions_cmd._DB_HANDLERS, "delete", lambda _db, _args: None)

    sessions_cmd.cmd_sessions(Namespace(sessions_action="delete"))

    factory.assert_called_once_with(read_only=False)


def test_sessions_stats_reader_does_not_disrupt_a_live_writer(monkeypatch, tmp_path):
    """The actual command reader leaves a writer's WAL generation untouched and usable."""
    from hermes_state import SessionDB
    import hermes_state

    db_path = tmp_path / "state.db"
    writer = SessionDB(db_path=db_path)
    try:
        writer.create_session("live", source="cli")
        wal_path = db_path.with_name("state.db-wal")
        before = wal_path.read_bytes() if wal_path.exists() else None
        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)

        sessions_cmd.cmd_sessions(Namespace(sessions_action="stats"))

        after = wal_path.read_bytes() if wal_path.exists() else None
        assert after == before
        writer.create_session("still-live", source="cli")
        assert writer.get_session("still-live") is not None
    finally:
        writer.close()


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


def test_doctor_without_fix_isolates_write_health_from_live_wal(monkeypatch, tmp_path):
    from pathlib import Path

    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_health
    from hermes_state import SessionDB
    import hermes_state_repair

    db_path = tmp_path / "state.db"
    writer = SessionDB(db_path=db_path)
    probed = []
    real = hermes_state_repair._db_opens_cleanly

    def _capture(path):
        probed.append(Path(path))
        return real(path)

    try:
        writer.create_session("live", source="cli")
        wal_path = db_path.with_name("state.db-wal")
        before = wal_path.read_bytes() if wal_path.exists() else b""
        monkeypatch.setattr(hermes_state_repair, "_db_opens_cleanly", _capture)

        _state_db_health(Finding(), False, db_path, "~/hermes")

        assert probed and probed[0] != db_path
        assert (wal_path.read_bytes() if wal_path.exists() else b"") == before
        writer.create_session("still-live", source="cli")
        assert writer.get_session("still-live") is not None
    finally:
        writer.close()
