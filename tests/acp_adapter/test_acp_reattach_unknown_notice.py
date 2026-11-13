"""A reconnect that re-attaches a session with a lost turn tells the editor at once, like load/resume
(JM44 r1-P1-4), and one that finds the turn discarded admits the next prompt (#140521): real
websockets, the real GatewayClient, two owners."""
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


def _owner(profile_id, epoch, pending, calls=None):
    async def handler(ws):
        async for raw in ws:
            frame = json.loads(raw)
            method = frame["method"]
            if calls is not None:
                calls.append(method)
            result = {"profile_id": profile_id} if method == "runtime.describe" else {}
            if method == "session.resume":
                result = {"session_id": frame["params"]["session_id"], "messages": [], "prompts": [],
                          "pending": pending, "replay_epoch": epoch, "last_sequence": 0}
            if method == "prompt.submit":
                result = {"admission_id": "next"}
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": result}))
            if method == "prompt.submit":
                await ws.send(json.dumps({"jsonrpc": "2.0", "method": "event", "params": {
                    "session_id": frame["params"]["session_id"], "type": "message.complete", "seq": 1,
                    "replay_epoch": epoch, "admission_id": "next", "payload": {"outcome": "completed"}}}))
    return handler


def _connector(monkeypatch, *servers):
    order = iter(servers)

    @asynccontextmanager
    async def connect_gateway():
        async with connect(f"ws://127.0.0.1:{next(order).sockets[0].getsockname()[1]}") as ws:
            async with GatewayClient(ws) as client:
                yield client

    monkeypatch.setattr(server, "connect_gateway", connect_gateway)


@pytest.mark.asyncio
async def test_reattach_after_restart_posts_the_unknown_recovery_notice(monkeypatch):
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    home = str(agent._home)
    first = await serve(_owner(home, "first", []), "127.0.0.1", 0)
    second = await serve(_owner(home, "second", [{"admission_id": "lost-1", "status": "unknown"}]), "127.0.0.1", 0)
    _connector(monkeypatch, first, second)
    try:
        await agent.load_session(cwd="/", session_id="s")
        before = agent._conn.session_update.await_count  # load's own notices (no unknown yet)
        first.close()  # the gateway restarts; its turn in flight becomes unknown on the new owner
        await first.wait_closed()
        await asyncio.wait_for(agent._event_task, 5)  # the pump records the transport loss
        await agent._client()
        (call,) = [call for call in agent._conn.session_update.await_args_list[before:]
                   if call.kwargs["update"].session_update == "agent_message_chunk"]
        assert call.kwargs["session_id"] == "s"
        assert "/discard lost-1" in call.kwargs["update"].content.text
    finally:
        await agent.aclose()
        second.close()


@pytest.mark.asyncio
async def test_next_prompt_reconnects_after_the_lost_turn_was_discarded_while_offline(monkeypatch):
    """The editor cached the lost turn, then lost its gateway connection (another restart, a dropped
    socket) while the classic chat discarded that turn: its next prompt reaches the owner."""
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    home, calls = str(agent._home), []
    first = await serve(_owner(home, "first", [{"admission_id": "lost-1", "status": "unknown"}]), "127.0.0.1", 0)
    second = await serve(_owner(home, "second", [], calls), "127.0.0.1", 0)
    _connector(monkeypatch, first, second)
    try:
        await agent.load_session(cwd="/", session_id="s")
        await asyncio.gather(*agent._background)  # load's post-attach reads stay on the first owner
        first.close()
        await first.wait_closed()
        await asyncio.wait_for(agent._event_task, 5)
        response = await asyncio.wait_for(agent.prompt([acp.text_block("next")], "s"), 5)
        assert response.stop_reason == "end_turn"
        assert calls == ["runtime.describe", "session.resume", "config.set", "prompt.submit"]
    finally:
        await agent.aclose()
        second.close()
