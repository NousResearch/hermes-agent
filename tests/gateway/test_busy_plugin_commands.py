"""Plugin slash commands remain directly dispatchable while a session is busy."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_busy_plugin_command_dispatches_handler_with_authorization():
    runner = object.__new__(GatewayRunner)
    runner._check_slash_access = lambda source, canonical_cmd: None
    source = SimpleNamespace(user_id="42")
    event = SimpleNamespace(get_command_args=lambda: "routing docs")
    calls = []

    async def handler(raw_args):
        calls.append(raw_args)
        return "kb-result"

    with patch("hermes_cli.plugins.get_plugin_command_handler", return_value=handler):
        result = await runner._dispatch_plugin_slash_command(event, "kb", source)

    assert result == "kb-result"
    assert calls == ["routing docs"]


@pytest.mark.asyncio
async def test_busy_plugin_command_honors_slash_access_denial():
    runner = object.__new__(GatewayRunner)
    runner._check_slash_access = lambda source, canonical_cmd: "denied"
    event = SimpleNamespace(get_command_args=lambda: "status")

    with patch("hermes_cli.plugins.get_plugin_command_handler") as lookup:
        result = await runner._dispatch_plugin_slash_command(
            event, "mission", SimpleNamespace(user_id="not-fred")
        )

    assert result == "denied"
    lookup.assert_not_called()
