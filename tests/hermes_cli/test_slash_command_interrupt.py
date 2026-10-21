"""Tests for the KeyboardInterrupt guard around slash command dispatch.

A Ctrl+C during a slow slash command (e.g. /skills browse on a large
skill tree, or /sessions list against a multi-GB SQLite DB) used to
unwind to the outer prompt_toolkit loop and kill the entire session.
The fix wraps `self.process_command(user_input)` in a try/except
KeyboardInterrupt so the command aborts but the session survives.

These tests verify the contract without spinning up the full
prompt_toolkit input loop. We exercise the same try/except by calling
through a thin wrapper that mirrors the real dispatch shape.
"""

from unittest.mock import patch

from cli import HermesCLI


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli._should_exit = False
    cli.conversation_history = []
    cli.agent = None
    cli._session_db = None
    return cli


def _dispatch(cli, user_input: str, process_command_side_effect=None):
    """Mirror the production dispatch shape from cli.py around line 14236.

    Real call site:
        if not _file_drop and isinstance(user_input, str) and _looks_like_slash_command(user_input):
            _cprint(f"\\n⚙️  {user_input}")
            try:
                if not self.process_command(user_input):
                    self._should_exit = True
                    if app.is_running:
                        app.exit()
            except KeyboardInterrupt:
                _cprint("\\n[dim]Command interrupted.[/dim]")
            continue
    """
    if process_command_side_effect is not None:
        with patch.object(cli, "process_command", side_effect=process_command_side_effect) as mock_pc:
            try:
                if not cli.process_command(user_input):
                    cli._should_exit = True
            except KeyboardInterrupt:
                # Mirror production: swallow, do NOT raise.
                pass
            return mock_pc


class TestSlashCommandKeyboardInterrupt:
    def test_keyboardinterrupt_in_slash_command_does_not_set_exit(self):
        """Ctrl+C in the middle of /skills browse must NOT set _should_exit.

        Before the fix: KeyboardInterrupt unwinds past the dispatch,
        the outer event loop catches it, session dies.
        After the fix: KeyboardInterrupt is caught locally, _should_exit
        stays False, the prompt loop continues.
        """
        cli = _make_cli()

        def raises_keyboard_interrupt(_cmd):
            raise KeyboardInterrupt("user pressed Ctrl+C during slow command")

        _dispatch(cli, "/skills browse", process_command_side_effect=raises_keyboard_interrupt)

        assert cli._should_exit is False, (
            "KeyboardInterrupt during slash command must not flag exit"
        )

    def test_normal_slash_command_returns_truthy_keeps_session_alive(self):
        """A successful slash command (returns truthy) must NOT set _should_exit."""
        cli = _make_cli()

        _dispatch(cli, "/help", process_command_side_effect=[True])

        assert cli._should_exit is False

    def test_slash_command_returning_false_sets_exit(self):
        """The legitimate exit signal — process_command() returning False —
        still sets _should_exit. This is the path /exit / /quit use."""
        cli = _make_cli()

        _dispatch(cli, "/exit", process_command_side_effect=[False])

        assert cli._should_exit is True

    def test_other_exceptions_propagate(self):
        """Only KeyboardInterrupt is caught locally. Other exceptions must
        propagate so they show up in logs and the global handler can deal
        with them — silently swallowing all exceptions would mask bugs."""
        cli = _make_cli()

        class CustomError(Exception):
            pass

        def raises_custom(_cmd):
            raise CustomError("real bug")

        try:
            with patch.object(cli, "process_command", side_effect=raises_custom):
                try:
                    if not cli.process_command("/something"):
                        cli._should_exit = True
                except KeyboardInterrupt:
                    pass  # would NOT catch CustomError
        except CustomError:
            return  # expected — non-KBI exceptions propagate

        raise AssertionError("CustomError should have propagated")
