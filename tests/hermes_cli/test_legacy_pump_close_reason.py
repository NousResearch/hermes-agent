"""Only observed PTY EOF proves that the process exited."""
import asyncio
from unittest.mock import Mock

import pytest
from starlette.websockets import WebSocket

from hermes_cli.pty_session import WS_CLOSE_PROCESS_EXITED
from hermes_cli.web_server_chat import _legacy_pump


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [None, "send", "read"])
async def test_legacy_pump_close_distinguishes_eof_from_transport_failure(failure):
    incoming = asyncio.Queue()
    await incoming.put({"type": "websocket.connect"})
    closes = []

    async def send(message):
        if message["type"] == "websocket.send" and failure == "send":
            raise RuntimeError("transport failed")
        if message["type"] == "websocket.close":
            closes.append(message["code"])
            await incoming.put({"type": "websocket.disconnect", "code": message["code"]})

    ws = WebSocket({"type": "websocket"}, incoming.get, send)
    await ws.accept()
    bridge = Mock()
    bridge.read.side_effect = [RuntimeError("read failed")] if failure == "read" else [b"output", None]
    await asyncio.wait_for(_legacy_pump(ws, bridge), timeout=3)
    assert closes == [WS_CLOSE_PROCESS_EXITED if failure is None else 1000]
    bridge.close.assert_called()
