"""The classic CLI's session title follows a compression rotation.

A gateway-backed CLI keeps the conversation's creation id for its whole life, while every
compression moves the live segment, and the title with it, to a new row. Reading the creation
row's title made the status bar badge (and ``/title``) go blank after the first compression.
"""
from datetime import datetime

from cli import HermesCLI
from hermes_state import SessionDB


def _rotated(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("root", source="cli")
    db.append_message("root", role="user", content="hi")
    db.set_session_title("root", "weekly digest")
    db.publish_compression_child(
        parent_session_id="root", child_session_id="tip", source="cli",
        messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
    # The rotation moves the name onto the live segment (the creation row's copy is cleared).
    db.set_session_title("tip", "weekly digest")
    return db


def _cli(db, session_id):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id, cli._session_db, cli._pending_title = session_id, db, None
    cli.model, cli.agent, cli.session_start, cli.conversation_history = "m", None, datetime.now(), []
    return cli


def test_status_bar_and_title_command_read_the_compression_tip(tmp_path, capsys):
    db = _rotated(tmp_path)
    try:
        assert db.get_session_title("root") is None  # the precondition the CLI used to miss
        cli = _cli(db, "root")
        assert cli._get_status_bar_session_title() == "weekly digest"
        cli._cmd_title("/title")
        assert "weekly digest" in capsys.readouterr().out
        cli._cmd_title("/title renamed")
        assert (db.get_session_title("tip"), db.get_session_title("root")) == ("renamed", None)
    finally:
        db.close()
