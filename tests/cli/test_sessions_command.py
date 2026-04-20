from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command
from hermes_state import SessionDB
from tests.cli.test_cli_init import _make_cli


def _seed_session(db: SessionDB, session_id: str, *, title: str | None = None, user_message: str | None = None):
    db.create_session(session_id=session_id, source="cli")
    if title:
        db.set_session_title(session_id, title)
    if user_message:
        db.append_message(session_id, role="user", content=user_message)
        db.append_message(session_id, role="assistant", content=f"Reply to {user_message}")


def test_sessions_command_is_registered_for_cli_and_gateway():
    command = resolve_command("/sessions")
    assert command is not None
    assert command.name == "sessions"
    assert "sessions" in GATEWAY_KNOWN_COMMANDS


def test_sessions_list_shows_historical_sessions(capsys, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_session(db, "current", title="Current Session", user_message="current preview")
        _seed_session(db, "old_cli_001", title="Old Session", user_message="hello from the past")

        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = db

        cli.process_command("/sessions")
        output = capsys.readouterr().out

        assert "Old Session" in output
        assert "old_cli_001" in output
        assert "Current Session" not in output
    finally:
        db.close()


def test_sessions_view_accepts_unique_id_prefix(capsys, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_session(
            db,
            "20260411_010101_abcd12",
            title="Deep Work",
            user_message="show me the transcript preview",
        )

        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = db

        cli.process_command("/sessions view 20260411_010101_abcd")
        output = capsys.readouterr().out

        assert "Deep Work" in output
        assert "20260411_010101_abcd12" in output
        assert "show me the transcript preview" in output
    finally:
        db.close()


def test_sessions_rename_updates_historical_session_title(capsys, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_session(db, "current", title="Current Session", user_message="current preview")
        _seed_session(db, "old_cli_001", title="Old Session", user_message="old preview")

        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = db

        cli.process_command("/sessions rename old_cli_001 Renamed Session")
        _ = capsys.readouterr().out

        assert db.get_session_title("old_cli_001") == "Renamed Session"
    finally:
        db.close()


def test_sessions_delete_requires_yes_then_deletes(capsys, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_session(db, "current", title="Current Session", user_message="current preview")
        _seed_session(db, "old_cli_001", title="Old Session", user_message="old preview")

        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = db

        cli.process_command("/sessions delete old_cli_001")
        output = capsys.readouterr().out
        assert "--yes" in output
        assert db.get_session("old_cli_001") is not None

        cli.process_command("/sessions delete old_cli_001 --yes")
        _ = capsys.readouterr().out
        assert db.get_session("old_cli_001") is None
    finally:
        db.close()
