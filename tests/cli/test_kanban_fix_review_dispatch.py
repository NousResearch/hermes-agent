from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.cli_commands_mixin import CLICommandsMixin


def _cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.session_id = "test-session"
    cli._pending_resume_sessions = None
    return cli


def test_cli_fix_review_gate_false_refuses_without_dispatch_or_legacy_engine(monkeypatch, capsys):
    monkeypatch.setattr("hermes_cli.kanban_fix_review.fix_review_command_enabled", lambda: False)
    with patch("agent.review_engine.start_review", side_effect=AssertionError("legacy review called")):
        assert _cli().process_command("/fix-review task-1") is True
    assert "/fix-review is disabled" in capsys.readouterr().out


def test_cli_fix_review_routes_to_canonical_adapter_when_enabled(monkeypatch):
    seen = {}
    monkeypatch.setattr("hermes_cli.kanban_fix_review.fix_review_command_enabled", lambda: True)

    def render(args):
        seen["args"] = args
        return "Fix-review correction started"

    monkeypatch.setattr("hermes_cli.kanban_fix_review.run_fix_review_slash_rendered", render)
    assert _cli().process_command("/fix-review task-1 --board default") is True
    assert seen["args"] == "/fix-review task-1 --board default"


def test_cli_has_one_fix_review_handler_without_shadowing_review_or_implement():
    handler = CLICommandsMixin.__dict__["_handle_fix_review_command"]
    assert list(CLICommandsMixin.__dict__).count("_handle_fix_review_command") == 1
    assert handler is HermesCLI._handle_fix_review_command
    assert handler.__module__ == "hermes_cli.cli_commands_mixin"
    assert HermesCLI._handle_review_command is not handler
    assert HermesCLI._handle_implement_command is not handler
