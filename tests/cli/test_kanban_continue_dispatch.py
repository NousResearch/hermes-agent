from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.cli_commands_mixin import CLICommandsMixin


def _cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.session_id = "test-session"
    cli._pending_resume_sessions = None
    return cli


def test_cli_continue_routes_to_canonical_adapter(monkeypatch):
    seen = {}

    def render(args):
        seen["args"] = args
        return "Continue: ready-implementation"

    monkeypatch.setattr("hermes_cli.kanban_continue.run_continue_slash_rendered", render)
    assert _cli().process_command("/continue task-1 --board default") is True
    assert seen["args"] == "/continue task-1 --board default"


def test_cli_continue_has_one_canonical_handler():
    handler = CLICommandsMixin.__dict__["_handle_continue_command"]
    assert list(CLICommandsMixin.__dict__).count("_handle_continue_command") == 1
    assert handler is HermesCLI._handle_continue_command


def test_cli_continue_gate_is_enforced_by_adapter(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.kanban_continue.continue_command_enabled", lambda: False)
    with patch("hermes_cli.kanban_continue.resolve_status_reference", side_effect=AssertionError("resolved")):
        assert _cli().process_command("/continue task-1") is True
    assert "/continue is disabled" in capsys.readouterr().out
