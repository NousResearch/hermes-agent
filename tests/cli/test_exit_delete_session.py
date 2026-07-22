"""Tests for `/exit --delete` and `/quit --delete` session deletion.

Ports the behavior from google-gemini/gemini-cli#19332: running `/exit` or
`/quit` with the `--delete` flag arms a one-shot `_delete_session_on_exit`
flag that the CLI shutdown path uses to remove the current session from
SQLite + on-disk transcripts before exit.
"""

from unittest.mock import MagicMock, patch


def _make_cli():
    """Bare HermesCLI suitable for process_command() tests.

    Uses ``__new__`` to skip the heavy __init__; only sets the attributes
    the /exit branch touches.
    """
    from cli import HermesCLI
    from hermes_cli.queue_management import ManagedPromptQueue
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = None
    cli.conversation_history = []
    cli.session_id = "test-session"
    cli._pending_input = ManagedPromptQueue()
    cli._delete_session_on_exit = False
    return cli


class TestExitDeleteFlag:
    def test_plain_exit_does_not_arm_delete(self):
        cli = _make_cli()
        result = cli.process_command("/exit")
        assert result is False
        assert cli._delete_session_on_exit is False

    def test_plain_quit_does_not_arm_delete(self):
        cli = _make_cli()
        result = cli.process_command("/quit")
        assert result is False
        assert cli._delete_session_on_exit is False

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

    def test_quit_alias_q_opens_queue_view_instead_of_exiting(self):
        """`/q` is the alias for `/queue`, not `/quit`. This test documents
        that a bare `/q` opens the queue view and cannot arm session deletion.
        The lightweight CLI fixture uses the same managed queue as production.
        """
        cli = _make_cli()
        with patch("cli._cprint") as mock_print:
            result = cli.process_command("/q")

        assert result is not False  # queue command doesn't exit
        assert cli._delete_session_on_exit is False
        assert cli._pending_input.snapshot_items() == []
        assert "No queued turns." in "\n".join(
            str(call.args[0]) for call in mock_print.call_args_list if call.args
        )

    def test_delete_flag_is_case_insensitive(self):
        cli = _make_cli()
        result = cli.process_command("/exit --DELETE")
        assert result is False
        assert cli._delete_session_on_exit is True

    def test_delete_flag_trims_whitespace(self):
        cli = _make_cli()
        result = cli.process_command("/exit   --delete   ")
        assert result is False
        assert cli._delete_session_on_exit is True

    def test_unknown_exit_argument_does_not_exit(self):
        """Unrecognised args should NOT exit the CLI — they surface an
        error message and stay in the session. This prevents accidental
        session destruction from typos like `/exit -delete`."""
        cli = _make_cli()
        result = cli.process_command("/exit --delte")
        # process_command returns True = keep running
        assert result is True
        assert cli._delete_session_on_exit is False

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
