import pytest


class _Event:
    def __init__(self, args=""):
        self._args = args

    def get_command_args(self):
        return self._args


@pytest.mark.asyncio
async def test_close_task_gateway_reaches_canonical_adapter(monkeypatch):
    from gateway.run import GatewayRunner

    seen = {}

    def fake_render(args):
        seen["args"] = args
        return "Close-task: done"

    monkeypatch.setattr("hermes_cli.kanban_close_task.run_close_task_slash_rendered", fake_render)
    runner = object.__new__(GatewayRunner)
    out = await runner._handle_close_task_command(_Event("t_123 --board default"))
    assert seen["args"] == "t_123 --board default"
    assert out == "Close-task: done"


def test_close_task_registry_gate_busy_policy_and_plain_dispatch():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command
    from gateway.run_busy import GatewayBusySessionMixin

    command = resolve_command("close-task")
    assert command is not None
    assert command.name == "close-task"
    assert command.cli_only is True
    assert command.gateway_config_gate == "kanban.close_task_command"
    assert command.busy_policy == "dispatch"
    assert "close-task" in GATEWAY_KNOWN_COMMANDS
    assert "close-task" in GatewayBusySessionMixin._PLAIN_COMMANDS


@pytest.mark.asyncio
async def test_close_task_gateway_failure_is_safe(monkeypatch):
    from gateway.run import GatewayRunner

    def fail(_args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("hermes_cli.kanban_close_task.run_close_task_slash_rendered", fail)
    runner = object.__new__(GatewayRunner)
    assert await runner._handle_close_task_command(_Event("t_123")) == "Close-task is unavailable. No action taken."
