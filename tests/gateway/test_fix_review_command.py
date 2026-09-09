"""Gateway registration and rendering for the canonical Kanban /fix-review command."""

import pytest


class _Event:
    def __init__(self, args=""):
        self._args = args

    def get_command_args(self):
        return self._args


@pytest.mark.asyncio
async def test_fix_review_command_reaches_canonical_adapter(monkeypatch):
    from gateway.run import GatewayRunner

    seen = {}

    def fake_render(args):
        seen["args"] = args
        return "Fix-review correction started\nTask: t_123\nRun: 9"

    monkeypatch.setattr("hermes_cli.kanban_fix_review.run_fix_review_slash_rendered", fake_render)
    runner = object.__new__(GatewayRunner)
    out = await runner._handle_fix_review_command(_Event("t_123 --board default"))
    assert seen["args"] == "t_123 --board default"
    assert "Fix-review correction started" in out


def test_fix_review_default_disabled_and_enabled_test_config_controls_gateway_help(monkeypatch):
    from hermes_cli import config as config_module
    from hermes_cli.commands import gateway_help_lines, resolve_command

    command = resolve_command("fix-review")
    assert command is not None
    assert command.name == "fix-review"
    assert command.cli_only is True
    assert command.gateway_config_gate == "kanban.fix_review_command"

    monkeypatch.setattr(config_module, "read_raw_config", lambda: {})
    assert not any(line.startswith("`/fix-review") for line in gateway_help_lines())

    monkeypatch.setattr(config_module, "read_raw_config", lambda: {"kanban": {"fix_review_command": True}})
    assert any(line.startswith("`/fix-review") for line in gateway_help_lines())


def test_fix_review_command_is_registered_once_without_review_collision():
    from gateway.slash_commands import GatewaySlashCommandsMixin
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    assert "fix-review" in GATEWAY_KNOWN_COMMANDS
    assert resolve_command("fix-review") is not resolve_command("review")
    assert list(vars(GatewaySlashCommandsMixin)).count("_handle_fix_review_command") == 1


@pytest.mark.asyncio
async def test_fix_review_handler_converts_adapter_failure_to_safe_result(monkeypatch):
    from gateway.run import GatewayRunner

    def fail(_args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("hermes_cli.kanban_fix_review.run_fix_review_slash_rendered", fail)
    runner = object.__new__(GatewayRunner)
    out = await runner._handle_fix_review_command(_Event("t_123"))
    assert out == "Fix-review is unavailable. No action taken."
