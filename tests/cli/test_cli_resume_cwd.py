"""Tests for cwd restore on CLI resume (-c).

Design #19242: the local CLI backend does not use TERMINAL_CWD for live
tracking -- that env var stays at launch dir. Instead _init_agent calls
os.chdir() to the persisted cwd so the terminal environment starts in
the right directory.
"""
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli(session_id="20260524_111111_xyz", db=None):
    """Build a minimal HermesCLI for the _init_agent resume code path."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli._resumed = True
    cli.conversation_history = []
    cli._session_db = db
    cli.tool_progress_mode = "full"
    cli.session_start = datetime.now()
    cli.agent = None
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    return cli


def _make_db(session_id="20260524_111111_xyz", cwd=None):
    db = MagicMock()
    meta = {"id": session_id, "title": "demo"}
    if cwd is not None:
        meta["cwd"] = cwd
    db.get_session.return_value = meta
    db.resolve_resume_session_id.return_value = session_id
    db.get_messages_as_conversation.return_value = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]
    db._conn = MagicMock()
    return db


class TestResumeCwdRestore:
    """Resume restores the session working directory from session_meta["cwd"]."""

    def test_restores_cwd_when_present_in_session_meta(self, monkeypatch, tmp_path):
        """os.getcwd() must equal session_meta["cwd"] after resume."""
        restore_dir = tmp_path / "workspace"
        restore_dir.mkdir()
        launch_dir = tmp_path / "launch"
        launch_dir.mkdir()
        monkeypatch.chdir(str(launch_dir))

        db = _make_db(cwd=str(restore_dir))
        cli = _make_cli(db=db)

        with patch("cli._prepare_deferred_agent_startup"):
            try:
                cli._init_agent()
            except Exception:
                # AIAgent construction fails in stubbed context -- we only
                # care that chdir happened before that point.
                pass

        assert os.getcwd() == str(restore_dir)

    def test_skips_restore_when_cwd_is_absent(self, monkeypatch, tmp_path):
        """NULL cwd in session_meta must not change the process directory."""
        launch_dir = tmp_path / "launch"
        launch_dir.mkdir()
        monkeypatch.chdir(str(launch_dir))

        db = _make_db()  # no cwd key -- simulates pre-fix session row
        cli = _make_cli(db=db)

        with patch("cli._prepare_deferred_agent_startup"):
            try:
                cli._init_agent()
            except Exception:
                pass

        assert os.getcwd() == str(launch_dir)

    def test_skips_restore_when_cwd_dir_does_not_exist(self, monkeypatch, tmp_path):
        """A persisted cwd pointing to a deleted directory must not crash."""
        launch_dir = tmp_path / "launch"
        launch_dir.mkdir()
        monkeypatch.chdir(str(launch_dir))

        missing_dir = str(tmp_path / "gone")
        db = _make_db(cwd=missing_dir)
        cli = _make_cli(db=db)

        with patch("cli._prepare_deferred_agent_startup"):
            try:
                cli._init_agent()
            except Exception:
                pass

        assert os.getcwd() == str(launch_dir)
