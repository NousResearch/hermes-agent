import pytest


class _Event:
    def __init__(self, args=""):
        self._args = args

    def get_command_args(self):
        return self._args


@pytest.mark.asyncio
async def test_busy_review_dispatch_reaches_canonical_handler(monkeypatch):
    from gateway.run import GatewayRunner
    from hermes_cli.commands import resolve_command

    seen = {}

    async def handler(self, event):
        seen["args"] = event.get_command_args()
        return "canonical review"

    monkeypatch.setattr(GatewayRunner, "_handle_review_command", handler)
    runner = object.__new__(GatewayRunner)
    command = resolve_command("review")
    assert command is not None and command.busy_policy == "dispatch"
    assert "review" in runner._PLAIN_COMMANDS
    assert "review" not in runner._IDLE_COMMANDS
    assert await runner._dispatch_busy_slash_command(
        _Event("task-1"), command, "review", object()
    ) == "canonical review"
    assert seen["args"] == "task-1"
