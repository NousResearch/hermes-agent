import pytest


class _Event:
    def __init__(self, args=""):
        self._args = args

    def get_command_args(self):
        return self._args


@pytest.mark.asyncio
async def test_recover_gateway_reaches_canonical_adapter(monkeypatch):
    from gateway.run import GatewayRunner

    seen = {}

    def fake_render(args):
        seen["args"] = args
        return "Recover: ready"

    monkeypatch.setattr("hermes_cli.kanban_recover.run_recover_slash_rendered", fake_render)
    runner = object.__new__(GatewayRunner)
    out = await runner._handle_recover_command(_Event("t_123 --requeue"))
    assert seen["args"] == "t_123 --requeue"
    assert out == "Recover: ready"


def test_recover_registry_gate_busy_policy_and_plain_dispatch():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command
    from gateway.run_busy import GatewayBusySessionMixin

    command = resolve_command("recover")
    assert command is not None
    assert command.name == "recover"
    assert command.cli_only is True
    assert command.gateway_config_gate == "kanban.recover_command"
    assert command.busy_policy == "dispatch"
    assert "recover" in GATEWAY_KNOWN_COMMANDS
    assert "recover" in GatewayBusySessionMixin._PLAIN_COMMANDS


@pytest.mark.asyncio
async def test_recover_gateway_failure_is_safe(monkeypatch):
    from gateway.run import GatewayRunner

    def fail(_args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("hermes_cli.kanban_recover.run_recover_slash_rendered", fail)
    runner = object.__new__(GatewayRunner)
    assert await runner._handle_recover_command(_Event("t_123")) == "Recover is unavailable. No action taken."
