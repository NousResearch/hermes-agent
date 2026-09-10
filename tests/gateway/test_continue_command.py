import pytest


class _Event:
    def __init__(self, args=""):
        self._args = args

    def get_command_args(self):
        return self._args


@pytest.mark.asyncio
async def test_continue_gateway_reaches_canonical_adapter(monkeypatch):
    from gateway.run import GatewayRunner

    seen = {}

    def fake_render(args):
        seen["args"] = args
        return "Continue: ready-implementation"

    monkeypatch.setattr("hermes_cli.kanban_continue.run_continue_slash_rendered", fake_render)
    runner = object.__new__(GatewayRunner)
    out = await runner._handle_continue_command(_Event("t_123 --board default"))

    assert seen["args"] == "t_123 --board default"
    assert out == "Continue: ready-implementation"


def test_continue_registry_gate_and_busy_dispatch():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, gateway_help_lines, resolve_command
    from gateway.run_busy import GatewayBusySessionMixin

    command = resolve_command("continue")
    assert command is not None
    assert command.name == "continue"
    assert command.cli_only is True
    assert command.gateway_config_gate == "kanban.continue_command"
    assert "continue" in GATEWAY_KNOWN_COMMANDS
    assert "continue" in GatewayBusySessionMixin._PLAIN_COMMANDS
    assert not any(line.startswith("`/continue") for line in gateway_help_lines())


@pytest.mark.asyncio
async def test_continue_gateway_failure_is_safe(monkeypatch):
    from gateway.run import GatewayRunner

    def fail(_args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("hermes_cli.kanban_continue.run_continue_slash_rendered", fail)
    runner = object.__new__(GatewayRunner)
    assert await runner._handle_continue_command(_Event("t_123")) == "Continue is unavailable. No action taken."
