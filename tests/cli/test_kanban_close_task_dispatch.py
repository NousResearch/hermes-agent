from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.cli_commands_mixin import CLICommandsMixin


def _cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.session_id = "test-session"
    cli._pending_resume_sessions = None
    return cli


def test_cli_close_task_routes_to_canonical_adapter(monkeypatch):
    seen = {}

    def render(args):
        seen["args"] = args
        return "Close-task: done"

    monkeypatch.setattr("hermes_cli.kanban_close_task.run_close_task_slash_rendered", render)
    assert _cli().process_command("/close-task task-1 --result ok") is True
    assert seen["args"] == "/close-task task-1 --result ok"


def test_cli_close_task_has_one_canonical_handler():
    handler = CLICommandsMixin.__dict__["_handle_close_task_command"]
    assert list(CLICommandsMixin.__dict__).count("_handle_close_task_command") == 1
    assert handler is HermesCLI._handle_close_task_command


def test_cli_close_task_gate_is_enforced_by_adapter(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.kanban_close_task.close_task_command_enabled", lambda: False)
    with patch("hermes_cli.kanban_close_task.resolve_status_reference", side_effect=AssertionError("resolved")):
        assert _cli().process_command("/close-task task-1") is True
    assert "/close-task is disabled" in capsys.readouterr().out
