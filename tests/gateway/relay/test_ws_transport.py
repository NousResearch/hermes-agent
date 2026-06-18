"""WebSocketRelayTransport against a real in-process WebSocket server.

Exercises the production transport over an actual ``websockets`` server (no
mock socket): handshake (hello -> descriptor), inbound frame -> handler,
outbound request/response correlation, and follow_up routing. Proves the wire
framing (newline-delimited JSON) and the request/response future plumbing work
end to end on a live socket.

Skipped cleanly if the optional ``websockets`` dependency is absent.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio

from gateway.relay.ws_transport import WebSocketRelayTransport, WEBSOCKETS_AVAILABLE

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    import websockets


DESCRIPTOR = {
    "contract_version": 1,
    "platform": "discord",
    "label": "Discord",
    "max_message_length": 2000,
    "supports_draft_streaming": False,
    "supports_edit": True,
    "supports_threads": True,
    "markdown_dialect": "discord",
    "len_unit": "chars",
}


class _StubConnectorServer:
    """Minimal connector: answers hello with a descriptor, echoes outbound."""

    def __init__(self):
        self.received: list[dict] = []
        self._server = None
        self.url = ""
        # Push channel: tests set this to a frame dict to deliver inbound.
        self._to_push: list[dict] = []

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        sock = next(iter(self._server.sockets))
        port = sock.getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        async for raw in ws:
            for line in str(raw).split("\n"):
                if not line.strip():
                    continue
                frame = json.loads(line)
                self.received.append(frame)
                await self._on_frame(ws, frame)

    async def _on_frame(self, ws, frame):
        ftype = frame.get("type")
        if ftype == "hello":
            await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
            # Deliver any queued inbound frames right after handshake.
            for f in self._to_push:
                await ws.send(json.dumps(f) + "\n")
        elif ftype == "outbound":
            action = frame.get("action", {})
            # Echo a successful result correlated by requestId.
            result = {"success": True, "message_id": f"srv-{action.get('op')}"}
            await ws.send(
                json.dumps({"type": "outbound_result", "requestId": frame["requestId"], "result": result})
                + "\n"
            )


@pytest_asyncio.fixture
async def server():
    srv = _StubConnectorServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.mark.asyncio
async def test_handshake_negotiates_descriptor(server):
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    await t.connect()
    try:
        desc = await t.handshake()
        assert desc.platform == "discord"
        assert desc.max_message_length == 2000
        # The hello carried the platform + botId.
        hello = next(f for f in server.received if f["type"] == "hello")
        assert hello["platform"] == "discord"
        assert hello["botId"] == "appShared"
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_inbound_frame_reaches_handler(server):
    server._to_push = [
        {
            "type": "inbound",
            "event": {
                "text": "hello from connector",
                "message_type": "text",
                "source": {"platform": "discord", "chat_id": "chan1", "chat_type": "group", "guild_id": "guildA"},
            },
            "bufferId": "buf-1",
        }
    ]
    received = []
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    t.set_inbound_handler(lambda ev: received.append(ev) or asyncio.sleep(0))
    await t.connect()
    try:
        await t.handshake()
        # Give the reader a tick to deliver the pushed inbound frame.
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].text == "hello from connector"
        assert received[0].source.guild_id == "guildA"
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_outbound_round_trips_with_correlation(server):
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    await t.connect()
    try:
        await t.handshake()
        result = await t.send_outbound({"op": "send", "chat_id": "chan1", "content": "hi"})
        assert result["success"] is True
        assert result["message_id"] == "srv-send"
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_follow_up_round_trips(server):
    t = WebSocketRelayTransport(server.url, "discord", "appShared")
    await t.connect()
    try:
        await t.handshake()
        result = await t.send_follow_up(
            {"op": "follow_up", "session_key": "s1", "kind": "discord.interaction_token", "content": "fu"}
        )
        assert result["success"] is True
        assert result["message_id"] == "srv-follow_up"
        # The follow_up rode an outbound frame the connector saw.
        outbound = [f for f in server.received if f["type"] == "outbound"]
        assert any(f["action"]["op"] == "follow_up" for f in outbound)
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_disconnect_fails_pending_waiters_cleanly(server):
    t = WebSocketRelayTransport(server.url, "discord", "appShared", outbound_timeout_s=5)
    await t.connect()
    await t.handshake()
    await t.disconnect()
    # After disconnect, an outbound returns a structured failure rather than hanging.
    result = await t.send_outbound({"op": "send", "chat_id": "c", "content": "x"})
    assert result["success"] is False
