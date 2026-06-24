"""Tests for the interactive curses browser fallback in /resume.

When stdin/stdout are a TTY, bare ``/resume`` launches the full curses
session browser (``_session_browse_picker``) so CLI users get arrow-key
navigation matching the TUI and desktop. In non-interactive contexts
(piped input, CI, gateway worker) the static numbered table is used.

These tests mock both paths so they run under pytest without a real TTY.
"""

from unittest.mock import MagicMock, patch


def _make_cli():
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "current_session"
    cli_obj._resumed = False
    cli_obj._pending_title = None
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._session_db = MagicMock()
    cli_obj._pending_resume_sessions = None
    cli_obj.resume_display = "minimal"
    return cli_obj


class TestInteractiveBrowserFallback:
    """Bare /resume on a TTY should launch the curses browser."""

    def test_tty_launches_browser_and_resumes_picked_session(self):
        """On a TTY with sessions, /resume opens the browser, picks a
        session, then recurses into /resume <id> to switch."""
        cli_obj = _make_cli()
        sessions = [
            {"id": "sess_002", "title": "Coding", "preview": "build", "last_active": None},
            {"id": "sess_001", "title": "Research", "preview": "read", "last_active": None},
        ]
        cli_obj._list_recent_sessions = MagicMock(return_value=sessions)
        cli_obj._session_db.get_session.return_value = {
            "id": "sess_002",
            "title": "Coding",
        }
        cli_obj._session_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        cli_obj._session_db.resolve_resume_session_id.return_value = "sess_002"

        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
            patch("hermes_cli.main._session_browse_picker", return_value="sess_002") as mock_picker,
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint"),
            patch("cli._sync_process_session_id"),
        ):
            mock_stdin.isatty.return_value = True
            mock_stdout.isatty.return_value = True
            cli_obj._handle_resume_command("/resume")

        # Browser was called with the session list.
        mock_picker.assert_called_once_with(sessions)
        # Session switched to the picked ID.
        assert cli_obj.session_id == "sess_002"

    def test_tty_browser_cancelled_does_not_switch(self):
        """If the user cancels the browser (Esc/q), no session switch happens."""
        cli_obj = _make_cli()
        sessions = [{"id": "sess_002", "title": "Coding"}]
        cli_obj._list_recent_sessions = MagicMock(return_value=sessions)

        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
            patch("hermes_cli.main._session_browse_picker", return_value=None),
            patch("cli._cprint") as mock_cprint,
        ):
            mock_stdin.isatty.return_value = True
            mock_stdout.isatty.return_value = True
            cli_obj._handle_resume_command("/resume")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "Cancelled" in printed
        assert cli_obj.session_id == "current_session"

    def test_tty_no_sessions_prints_message(self):
        """On a TTY with no sessions, /resume prints a clear message."""
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[])

        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
            patch("cli._cprint") as mock_cprint,
        ):
            mock_stdin.isatty.return_value = True
            mock_stdout.isatty.return_value = True
            cli_obj._handle_resume_command("/resume")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "No previous sessions" in printed
        assert cli_obj.session_id == "current_session"

    def test_non_tty_falls_back_to_numbered_table(self):
        """In a non-interactive context (no TTY), /resume uses the static
        numbered table and arms the one-shot pending selection."""
        cli_obj = _make_cli()
        sessions = [
            {"id": "sess_002", "title": "Coding"},
            {"id": "sess_001", "title": "Research"},
        ]
        cli_obj._list_recent_sessions = MagicMock(return_value=sessions)
        cli_obj._show_recent_sessions = MagicMock(return_value=True)

        with (
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
            patch("cli._cprint"),
        ):
            mock_stdin.isatty.return_value = False
            mock_stdout.isatty.return_value = False
            cli_obj._handle_resume_command("/resume")

        # Non-TTY path arms the one-shot pending selection (see #34584).
        assert cli_obj._pending_resume_sessions == sessions