"""Integration: brain_rpc_request over WebSocketRelayTransport (in-process WS)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio

from gateway.relay.ws_transport import WEBSOCKETS_AVAILABLE, WebSocketRelayTransport
from tests.gateway.brain_rpc.conftest import make_request

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    import websockets


DESCRIPTOR = {
    "contract_version": 1,
    "platform": "lanyard_web",
    "label": "Lanyard",
    "max_message_length": 4000,
    "supports_draft_streaming": False,
    "supports_edit": False,
    "supports_threads": False,
    "markdown_dialect": "markdown",
    "len_unit": "chars",
}


class _BrainRpcConnector:
    """Connector stub that pushes one brain_rpc_request after hello."""

    def __init__(self, request_frame: dict):
        self.received: list[dict] = []
        self._request = request_frame
        self._server = None
        self.url = ""
        self.result: dict | None = None
        self._result_event = asyncio.Event()

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
                ftype = frame.get("type")
                if ftype == "hello":
                    await ws.send(
                        json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n"
                    )
                    await ws.send(json.dumps(self._request) + "\n")
                elif ftype == "brain_rpc_result":
                    self.result = frame
                    self._result_event.set()


@pytest.mark.asyncio
async def test_ws_transport_brain_rpc_roundtrip(
    vault_root: Path, profiles_dir: Path, monkeypatch
):
    monkeypatch.setenv("VAULT_ROOT", str(vault_root))
    monkeypatch.setenv("BRAIN_PROFILES_DIR", str(profiles_dir))
    monkeypatch.setenv("GATEWAY_RELAY_INSTANCE_ID", "inst_test")
    monkeypatch.setenv("BRAIN_TENANT_ID", "ten_test")
    monkeypatch.setenv("BRAIN_RPC_ENABLED", "1")

    # Reload host config for the process-wide dispatcher.
    from gateway.brain_rpc.dispatcher import reset_default_dispatcher

    reset_default_dispatcher()

    req = make_request("brain.ping", params={"echo": "ws"}, request_id="ws_req_1")
    srv = _BrainRpcConnector(req)
    await srv.start()
    try:
        t = WebSocketRelayTransport(srv.url, "lanyard_web", "bot1")
        await t.connect()
        try:
            await t.handshake()
            await asyncio.wait_for(srv._result_event.wait(), timeout=5.0)
        finally:
            await t.disconnect()
    finally:
        await srv.stop()

    assert srv.result is not None
    assert srv.result["type"] == "brain_rpc_result"
    assert srv.result["request_id"] == "ws_req_1"
    assert srv.result["ok"] is True
    assert srv.result["result"]["pong"] is True
    assert srv.result["result"]["echo"] == "ws"
