"""A reconnect that re-attaches a session with a lost turn tells the editor at once, like load/resume
(JM44 r1-P1-4): real websockets, the real GatewayClient, two owners."""
import asyncio
from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import acp_adapter.gateway_server as server
from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClient


def _owner(profile_id, epoch, pending):
    async def handler(ws):
        async for raw in ws:
            frame = json.loads(raw)
            result = {"profile_id": profile_id} if frame["method"] == "runtime.describe" else {}
            if frame["method"] == "session.resume":
                result = {"session_id": frame["params"]["session_id"], "messages": [], "prompts": [],
                          "pending": pending, "replay_epoch": epoch, "last_sequence": 0}
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": result}))
    return handler


@pytest.mark.asyncio
async def test_reattach_after_restart_posts_the_unknown_recovery_notice(monkeypatch):
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    home = str(agent._home)
    first = await serve(_owner(home, "first", []), "127.0.0.1", 0)
    second = await serve(_owner(home, "second", [{"admission_id": "lost-1", "status": "unknown"}]), "127.0.0.1", 0)
    order = iter([first, second])

    @asynccontextmanager
    async def connect_gateway():
        async with connect(f"ws://127.0.0.1:{next(order).sockets[0].getsockname()[1]}") as ws:
            async with GatewayClient(ws) as client:
                yield client

    monkeypatch.setattr(server, "connect_gateway", connect_gateway)
    try:
        await agent.load_session(cwd="/", session_id="s")
        before = agent._conn.session_update.await_count  # load's own notices (no unknown yet)
        first.close()  # the gateway restarts; its turn in flight becomes unknown on the new owner
        await first.wait_closed()
        await asyncio.wait_for(agent._event_task, 5)  # the pump records the transport loss
        await agent._client()
        (call,) = agent._conn.session_update.await_args_list[before:]
        assert call.kwargs["session_id"] == "s"
        assert "/discard lost-1" in call.kwargs["update"].content.text
    finally:
        await agent.aclose()
        second.close()

