"""Tests for /resume status lines going to stderr in quiet mode (#11793).

The fix in cli._init_agent routes three messages to stderr when
``tool_progress_mode == "off"`` (set by ``hermes chat --quiet``):

  * "Session not found: ..."
  * "↻ Resumed session ... (N user messages, M total messages)"
  * "Session ... found but has no messages. Starting fresh."

Interactive mode (tool_progress_mode == "full") still uses ChatConsole.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI


def _make_cli(quiet=False, session_id="20260524_111111_xyz", db=None):
    """Build a minimal HermesCLI bound to only what _init_agent needs for
    the resume code path: _resumed, _session_db, conversation_history,
    session_id, and tool_progress_mode."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli._resumed = True
    cli.conversation_history = []
    cli._session_db = db
    cli.tool_progress_mode = "off" if quiet else "full"
    cli.session_start = datetime.now()
    cli.agent = None
    # We need _init_agent to reach the resume block (line ~4757) but not
    # proceed into actual AIAgent construction. _ensure_runtime_credentials
    # must return True (False returns early at line 4743). _install_tool_callbacks,
    # _ensure_tirith_security are stubbed; the resume block will either return
    # False (session-not-found) or reach the eventual AIAgent() call which
    # we'll let raise — we only check stdout/stderr printed BEFORE that.
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    return cli


class TestResumeQuietStderr:
    def test_session_not_found_goes_to_stderr_in_quiet_mode(self, capsys):
        db = MagicMock()
        db.get_session.return_value = None
        cli = _make_cli(quiet=True, db=db)

        with patch("cli._prepare_deferred_agent_startup"):
            result = cli._init_agent()

        captured = capsys.readouterr()
        assert result is False
        # stdout must stay clean
        assert "Session not found" not in captured.out
        # the resume status goes to stderr
        assert "Session not found" in captured.err
        assert "hermes sessions list" in captured.err

    def test_session_not_found_goes_to_stdout_in_full_mode(self, capsys):
        db = MagicMock()
        db.get_session.return_value = None
        cli = _make_cli(quiet=False, db=db)

        with patch("cli._prepare_deferred_agent_startup"):
            result = cli._init_agent()

        captured = capsys.readouterr()
        assert result is False
        # Interactive mode keeps the existing _cprint path → stdout.
        assert "Session not found" in captured.out

    def test_resumed_banner_goes_to_stderr_in_quiet_mode(self, capsys):
        db = MagicMock()
        db.get_session.return_value = {"id": "20260524_111111_xyz", "title": "demo"}
        db.resolve_resume_session_id.return_value = "20260524_111111_xyz"
        db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        db._conn = MagicMock()  # for the reopen execute() call

        cli = _make_cli(quiet=True, db=db)
        # Stop _init_agent right after the resume banner: prevent it from
        # constructing a real AIAgent (the next code path).
        with patch("cli._prepare_deferred_agent_startup"):
            try:
                cli._init_agent()
            except Exception:
                # The post-resume agent-init machinery may fail in this
                # stubbed context (no API key, no real config) — we only
                # care about the printed banner that comes earlier.
                pass

        captured = capsys.readouterr()
        # Banner on stderr — stdout stays clean for automation.
        assert "↻ Resumed session" not in captured.out
        assert "↻ Resumed session" in captured.err
        assert "20260524_111111_xyz" in captured.err
        assert "demo" in captured.err

    def test_no_messages_goes_to_stderr_in_quiet_mode(self, capsys):
        db = MagicMock()
        db.get_session.return_value = {"id": "20260524_111111_xyz"}
        db.resolve_resume_session_id.return_value = "20260524_111111_xyz"
        db.get_messages_as_conversation.return_value = []
        db._conn = MagicMock()

        cli = _make_cli(quiet=True, db=db)
        with patch("cli._prepare_deferred_agent_startup"):
            try:
                cli._init_agent()
            except Exception:
                pass

        captured = capsys.readouterr()
        assert "has no messages" not in captured.out
        assert "has no messages" in captured.err
        assert "Starting fresh" in captured.err
