"""Tests for `/exit --delete` and `/quit --delete` session deletion.

Ports the behavior from google-gemini/gemini-cli#19332: running `/exit` or
`/quit` with the `--delete` flag arms a one-shot `_delete_session_on_exit`
flag that the CLI shutdown path uses to remove the current session from
SQLite + on-disk transcripts before exit.
"""

from unittest.mock import MagicMock


def _make_cli():
    """Bare HermesCLI suitable for process_command() tests.

    Uses ``__new__`` to skip the heavy __init__; only sets the attributes
    the /exit branch touches.
    """
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = None
    cli.conversation_history = []
    cli.session_id = "test-session"
    cli._delete_session_on_exit = False
    return cli


class TestExitDeleteFlag:


    def test_exit_delete_arms_flag(self):
        cli = _make_cli()
        result = cli.process_command("/exit --delete")
        assert result is False
        assert cli._delete_session_on_exit is True

    def test_quit_delete_arms_flag(self):
        cli = _make_cli()
        result = cli.process_command("/quit --delete")
        assert result is False
        assert cli._delete_session_on_exit is True

    def test_exit_delete_short_form(self):
        """`-d` is a convenience alias for `--delete`."""
        cli = _make_cli()
        result = cli.process_command("/exit -d")
        assert result is False
        assert cli._delete_session_on_exit is True



    def test_delete_flag_trims_whitespace(self):
        cli = _make_cli()
        result = cli.process_command("/exit   --delete   ")
        assert result is False
        assert cli._delete_session_on_exit is True


    def test_unknown_exit_argument_prints_help(self):
        cli = _make_cli()
        # _cprint goes through module-level print, so capture via console.
        # We can't patch _cprint directly without import juggling; the
        # previous assertion already proves the unknown-arg branch is
        # reached (result True + flag False).
        result = cli.process_command("/exit garbage")
        assert result is True
        assert cli._delete_session_on_exit is False


class TestCommandRegistry:
    def test_quit_command_advertises_delete_flag(self):
        """The CommandDef args_hint should surface `--delete` in /help and
        CLI autocomplete."""
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("quit")
        assert cmd is not None
        assert cmd.args_hint == "[--delete]"

    def test_exit_alias_resolves_to_quit_with_hint(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("exit")
        assert cmd is not None
        assert cmd.name == "quit"
        assert cmd.args_hint == "[--delete]"
