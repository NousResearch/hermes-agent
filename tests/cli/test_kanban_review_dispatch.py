from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.cli_commands_mixin import CLICommandsMixin


def _cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.session_id = "test-session"
    cli._pending_resume_sessions = None
    return cli


def test_cli_review_gate_false_refuses_canonical_command_without_legacy_engine(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.kanban_review.review_command_enabled", lambda: False)
    with patch("agent.review_engine.start_review", side_effect=AssertionError("legacy review called")):
        assert _cli().process_command("/review task-1") is True
    assert "/review is disabled" in capsys.readouterr().out


def test_cli_review_routes_to_kanban_adapter_when_enabled(monkeypatch):
    seen = {}
    monkeypatch.setattr("hermes_cli.kanban_review.review_command_enabled", lambda: True)

    def render(args):
        seen["args"] = args
        return "Review started"

    monkeypatch.setattr("hermes_cli.kanban_review.run_review_slash_rendered", render)
    assert _cli().process_command("/review task-1 --board default") is True
    assert seen["args"] == "/review task-1 --board default"


def test_cli_has_one_review_handler_and_uses_canonical_mixin_surface():
    handler = CLICommandsMixin.__dict__["_handle_review_command"]
    assert list(CLICommandsMixin.__dict__).count("_handle_review_command") == 1
    assert handler is HermesCLI._handle_review_command
    assert handler.__module__ == "hermes_cli.cli_commands_mixin"
