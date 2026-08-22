"""#88607 — WS rejection codes must survive a REAL uvicorn handshake.

uvicorn answers a pre-accept ``ws.close(code=44xx)`` with a bare HTTP 403
(no close frame), so a real browser sees 1006/"" and every 44xx-keyed
client path dies in production. Starlette's TestClient surfaces both
shapes identically as ``WebSocketDisconnect(code)``, which is exactly the
blind spot that let the bug ship — hence this test runs a real uvicorn
server and a real `websockets` client.
"""

import asyncio
import socket
import sys
import threading
from pathlib import Path

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

websockets = pytest.importorskip("websockets")
uvicorn = pytest.importorskip("uvicorn")

from starlette.applications import Starlette  # noqa: E402
from starlette.routing import WebSocketRoute  # noqa: E402

from hermes_cli.web_server import _ws_reject  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _reject_ep(ws):
    await _ws_reject(ws, 4401, "auth: token_mismatch")


@pytest.mark.asyncio
async def test_rejection_code_reaches_real_client():
    """accept-then-close delivers code+reason through uvicorn's stack."""
    app = Starlette(routes=[WebSocketRoute("/api/ws", _reject_ep)])
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve())

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        deadline = asyncio.get_event_loop().time() + 10
        while not server.started:
            if asyncio.get_event_loop().time() > deadline:
                pytest.fail("uvicorn did not start")
            await asyncio.sleep(0.05)

        # proxy=None: websockets 15 otherwise auto-discovers the system
        # proxy (e.g. macOS scutil) and demands python-socks for SOCKS
        # entries — an environment dependency this test does not need.
        async with websockets.connect(
            f"ws://127.0.0.1:{port}/api/ws", proxy=None
        ) as client:
            with pytest.raises(
                websockets.exceptions.ConnectionClosed,
            ) as excinfo:
                await asyncio.wait_for(client.recv(), timeout=5)
        # 4401 arrives intact instead of the 1006 a pre-accept close yields.
        assert excinfo.value.rcvd is not None
        assert excinfo.value.rcvd.code == 4401
        assert "token_mismatch" in (excinfo.value.rcvd.reason or "")
    finally:
        server.should_exit = True
        thread.join(timeout=10)
