"""An ACP editor survives a gateway restart: real websockets, the real GatewayClient, two owners."""
import asyncio
from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock

import acp
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import acp_adapter.gateway_server as server
from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClient


class Owner:
    """One gateway instance: answers the canonical RPCs ACP uses and records them."""

    def __init__(self, profile_id, epoch, *, die_on_submit=None):
        self.profile_id, self.epoch, self.die_on_submit = profile_id, epoch, die_on_submit
        self.calls = []
        self.server = None

    async def handler(self, ws):
        async for raw in ws:
            frame = json.loads(raw)
            method, params = frame["method"], frame["params"]
            self.calls.append(method)
            if method == "prompt.submit" and self.die_on_submit == "before_ack":
                await ws.close()  # the owner dies with the submit in flight
                return
            result = {"profile_id": self.profile_id} if method == "runtime.describe" else {}
            if method == "session.resume":
                result = {"session_id": params["session_id"], "messages": [], "pending": [], "prompts": [],
                          "replay_epoch": self.epoch, "last_sequence": 0}
            if method == "prompt.submit":
                result = {"admission_id": f"{self.epoch}-{params['input_id']}"}
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": result}))
            if method == "prompt.submit" and self.die_on_submit == "after_ack":
                await ws.close()
                return
            if method == "prompt.submit":
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {
                    "session_id": params["session_id"], "type": "message.complete", "replay_epoch": self.epoch,
                    "seq": 1, "admission_id": result["admission_id"],
                    "payload": {"outcome": "completed", "text": "AFTER-RESTART"}}}))

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"


@pytest.mark.asyncio
@pytest.mark.parametrize("die", ["before_ack", "after_ack"])
async def test_editor_reconnects_to_a_restarted_gateway_without_resending(monkeypatch, die):
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    home = str(agent._home)
    owners = [Owner(home, "first", die_on_submit=die), Owner(home, "second")]
    for owner in owners:
        owner.server = await serve(owner.handler, "127.0.0.1", 0)
    order = iter(owners)

    @asynccontextmanager
    async def connect_gateway():
        async with connect(next(order).url) as ws:
            async with GatewayClient(ws) as client:
                yield client

    monkeypatch.setattr(server, "connect_gateway", connect_gateway)
    try:
        await agent.load_session(cwd="/", session_id="s")
        # The in-flight turn's outcome is unknown: reported, never silently resubmitted.
        with pytest.raises(Exception, match="outcome is unknown"):
            await asyncio.wait_for(agent.prompt([acp.text_block("in flight")], "s"), 5)
        response = await asyncio.wait_for(agent.prompt([acp.text_block("after restart")], "s"), 5)
        assert response.stop_reason == "end_turn"
        assert agent._failure is None
        # The replacement was re-verified and re-subscribed before the new submit; the
        # lost input was submitted to the first owner only.
        assert owners[1].calls == ["runtime.describe", "session.resume", "prompt.submit"]
        assert owners[0].calls.count("prompt.submit") == 1
        assert agent._snapshots["s"]["replay_epoch"] == "second"
        texts = [call.kwargs["update"].content.text for call in agent._conn.session_update.await_args_list]
        assert texts.count("AFTER-RESTART") == 1
    finally:
        await agent.aclose()
        for owner in owners:
            owner.server.close()
