"""Only registered Hermes commands are slash commands; other text starting with '/' is a prompt."""
import asyncio
from unittest.mock import AsyncMock

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["/etc/hosts is wrong", "/tmp/build.log shows the failure", "/notacommand please"])
async def test_path_and_unknown_slash_text_is_submitted_as_the_prompt(text):
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots["s"] = {}

    async def rpc(method, **params):
        assert method == "prompt.submit"
        await agent._project({"session_id": "s", "type": "message.complete", "admission_id": "a",
                              "payload": {"outcome": "completed", "text": "ok"}})
        return {"admission_id": "a"}

    agent._gateway.rpc.side_effect = rpc
    response = await asyncio.wait_for(agent.prompt([acp.text_block(text)], "s"), 5)
    assert response.stop_reason == "end_turn"
    assert agent._gateway.rpc.await_args.kwargs["text"] == text


@pytest.mark.asyncio
async def test_registered_command_this_surface_does_not_carry_is_still_refused_locally():
    # A Hermes command is never forwarded as text the owner might execute as a control.
    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._snapshots["s"] = {}
    with pytest.raises(GatewayClientError, match="unsupported_command"):
        await agent.prompt([acp.text_block("/reset")], "s")
    agent._gateway.rpc.assert_not_awaited()
