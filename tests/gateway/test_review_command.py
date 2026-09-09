"""Gateway registration and rendering for the canonical Kanban /review command."""

import pytest


class _Event:
    def __init__(self, args=""):
        self._args = args

    def get_command_args(self):
        return self._args


@pytest.mark.asyncio
async def test_review_command_reaches_canonical_adapter(monkeypatch):
    from gateway.run import GatewayRunner

    seen = {}

    def fake_render(args):
        seen["args"] = args
        return "Review started\nTask: t_123\nRun: 9\nReviewer: rozmilo-claude"

    monkeypatch.setattr("hermes_cli.kanban_review.run_review_slash_rendered", fake_render)
    runner = object.__new__(GatewayRunner)
    out = await runner._handle_review_command(_Event("t_123 --board default"))
    assert seen["args"] == "t_123 --board default"
    assert "Review started" in out
    assert "rozmilo-claude" in out


def test_review_command_registry_is_canonical_and_gated():
    from hermes_cli.commands import resolve_command

    command = resolve_command("review")
    assert command is not None
    assert command.name == "review"
    assert command.cli_only is True
    assert command.gateway_config_gate == "kanban.review_command"


@pytest.mark.asyncio
async def test_review_handler_converts_adapter_failure_to_safe_result(monkeypatch):
    from gateway.run import GatewayRunner

    def fail(_args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("hermes_cli.kanban_review.run_review_slash_rendered", fail)
    runner = object.__new__(GatewayRunner)
    out = await runner._handle_review_command(_Event("t_123"))
    assert out == "Review is unavailable. No action taken."
