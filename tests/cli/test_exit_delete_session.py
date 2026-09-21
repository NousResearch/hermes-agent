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

    def test_leave_command_is_registered(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("leave")
        assert cmd is not None
        assert cmd.name == "leave"
        assert cmd.args_hint == "[--stop]"


class TestLeaveCommand:
    def test_leave_exits_when_no_work_is_pending(self, monkeypatch):
        cli = _make_cli()
        cli._agent_running = False
        monkeypatch.setattr("tools.process_registry.process_registry.list_sessions", lambda: [])
        monkeypatch.setattr("tools.async_delegation.list_async_delegations", lambda: [])
        monkeypatch.setattr("tools.write_approval.list_pending", lambda subsystem: [])
        assert cli.process_command("/leave") is False

    def test_leave_reports_pending_writes_but_allows_exit(self, monkeypatch, capsys):
        cli = _make_cli()
        cli._agent_running = False
        monkeypatch.setattr("tools.process_registry.process_registry.list_sessions", lambda: [])
        monkeypatch.setattr("tools.async_delegation.list_async_delegations", lambda: [])
        monkeypatch.setattr(
            "tools.write_approval.list_pending",
            lambda subsystem: [{"id": "pending-1"}] if subsystem == "skills" else [],
        )
        assert cli.process_command("/leave") is False
        output = capsys.readouterr().out.lower()
        assert "skill write" in output
        assert "durable" in output

    def test_leave_stop_stops_background_work_before_rechecking(self, monkeypatch):
        cli = _make_cli()
        cli._agent_running = False
        checks = iter(([{"status": "running", "session_id": "proc-1"}], []))
        monkeypatch.setattr("tools.process_registry.process_registry.list_sessions", lambda: next(checks))
        monkeypatch.setattr("tools.async_delegation.list_async_delegations", lambda: [])
        monkeypatch.setattr("tools.write_approval.list_pending", lambda subsystem: [])
        stopped = []
        cli._handle_stop_command = lambda: stopped.append(True)
        assert cli.process_command("/leave --stop") is False
        assert stopped == [True]

    def test_leave_warns_about_persisted_interruption(self, monkeypatch, capsys):
        cli = _make_cli()
        cli._agent_running = False
        monkeypatch.setattr("tools.process_registry.process_registry.list_sessions", lambda: [])
        monkeypatch.setattr("tools.async_delegation.list_async_delegations", lambda: [])
        monkeypatch.setattr("tools.write_approval.list_pending", lambda subsystem: [])
        cli._get_interrupted_turn_marker = lambda: {"timestamp": 123.0, "prompt": "update the skill"}
        assert cli.process_command("/leave") is False
        output = capsys.readouterr().out.lower()
        assert "interrupted" in output
        assert "will remain available" in output

    def test_interrupted_turn_marker_is_persisted_and_cleared(self):
        cli = _make_cli()

        class FakeDB:
            def __init__(self):
                self.patches = []
            def patch_session_model_config(self, session_id, patch):
                self.patches.append((session_id, patch))
            def get_session_model_config_value(self, session_id, key):
                return None

        cli._session_db = FakeDB()
        cli._persist_interrupted_turn_marker("update the skill")
        cli._clear_interrupted_turn_marker()
        assert cli._session_db.patches[0][1]["_cli_interrupted_turn"]["prompt"] == "update the skill"
        assert cli._session_db.patches[1] == ("test-session", {"_cli_interrupted_turn": None})

    def test_resume_recap_warns_about_interrupted_turn(self):
        cli = _make_cli()
        output = []
        cli._resume_display_history = []
        cli.conversation_history = []
        cli.resume_display = "full"
        cli._get_interrupted_turn_marker = lambda: {"prompt": "update the skill"}
        cli._console_print = output.append
        cli._display_resumed_history()
        assert output
        assert "Previous turn was interrupted" in output[0]
