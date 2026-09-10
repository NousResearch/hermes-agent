from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.cli_commands_mixin import CLICommandsMixin


def _cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.session_id = "test-session"
    cli._pending_resume_sessions = None
    return cli


def test_cli_recover_routes_to_canonical_adapter(monkeypatch):
    seen = {}

    def render(args):
        seen["args"] = args
        return "Recover: ready"

    monkeypatch.setattr("hermes_cli.kanban_recover.run_recover_slash_rendered", render)
    assert _cli().process_command("/recover task-1 --requeue") is True
    assert seen["args"] == "/recover task-1 --requeue"


def test_cli_recover_has_one_canonical_handler():
    handler = CLICommandsMixin.__dict__["_handle_recover_command"]
    assert list(CLICommandsMixin.__dict__).count("_handle_recover_command") == 1
    assert handler is HermesCLI._handle_recover_command


def test_cli_recover_gate_is_enforced_by_adapter(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.kanban_recover.recover_command_enabled", lambda: False)
    with patch("hermes_cli.kanban_recover.resolve_status_reference", side_effect=AssertionError("resolved")):
        assert _cli().process_command("/recover task-1") is True
    assert "/recover is disabled" in capsys.readouterr().out
