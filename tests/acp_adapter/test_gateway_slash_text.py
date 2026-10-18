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
async def test_registered_command_the_owner_refuses_is_a_named_reply_not_a_failed_prompt():
    # A Hermes command is never forwarded as prompt text: the owner's slash.exec answers or
    # refuses it, and the refusal is the command's reply (#134889 ACP-17).
    from hermes_cli.gateway_client import GatewayRPCError
    agent = GatewayACPAgent()
    agent._gateway, agent._conn = AsyncMock(), AsyncMock()
    agent._snapshots["s"] = {}
    calls = []

    async def rpc(method, **params):
        calls.append((method, params))
        raise GatewayRPCError("unsupported_command")

    agent._gateway.rpc.side_effect = rpc
    response = await agent.prompt([acp.text_block("/rollback")], "s")
    assert response.stop_reason == "end_turn"
    assert calls == [("slash.exec", {"session_id": "s", "command": "/rollback"})]
    shown = agent._conn.session_update.await_args.kwargs["update"].content.text
    assert "/rollback is not available on the shared gateway" in shown
