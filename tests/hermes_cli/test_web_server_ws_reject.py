"""WebSocket rejections must reach the client as close codes.

``ws.close(code=44xx)`` issued *before* ``ws.accept()`` never arrives as a
close frame: uvicorn answers the upgrade with a bare HTTP 403 and drops the
ASGI code/reason, so a browser sees ``close code=1006 reason=""`` and the
dashboard's 4401/4403/4404 handling never fires. Starlette's ``TestClient``
does not emulate that (it surfaces the pre-accept close as
``WebSocketDisconnect(44xx)``), which is why the existing TestClient tests
stayed green while the codes were lost in production. Hence the real-uvicorn
test below: it is the fail-before for ``web_server._ws_reject``.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading

import pytest
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server


# ---------------------------------------------------------------------------
# Unit: helper contract on a fake socket.
# ---------------------------------------------------------------------------


class _FakeWs:
    def __init__(self, accept_error: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self._accept_error = accept_error

    async def accept(self) -> None:
        self.calls.append(("accept",))
        if self._accept_error is not None:
            raise self._accept_error

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.calls.append(("close", code, reason))


def test_ws_reject_accepts_then_closes_with_code_and_reason():
    ws = _FakeWs()
    asyncio.run(web_server._ws_reject(ws, 4401, "auth: token_mismatch"))
    assert ws.calls == [("accept",), ("close", 4401, "auth: token_mismatch")]


@pytest.mark.parametrize(
    "error", [WebSocketDisconnect(1006), OSError("peer gone")],
)
def test_ws_reject_swallows_peer_gone_on_accept(error):
    ws = _FakeWs(accept_error=error)
    asyncio.run(web_server._ws_reject(ws, 4403, "host"))
    # Nothing after the failed accept: no close attempted, no raise.
    assert ws.calls == [("accept",)]


def test_ws_reject_propagates_programming_errors():
    ws = _FakeWs(accept_error=RuntimeError("accepted twice"))
    with pytest.raises(RuntimeError):
        asyncio.run(web_server._ws_reject(ws, 4401))


# /api/ws stays close-before-accept until JsonRpcGatewayClient.connect()
# settles on the first frame instead of `open` (see the comment at the
# endpoint); everything else must reject through _ws_reject.
_PRE_ACCEPT_CLOSE_ALLOWED = {"/api/ws"}


def test_no_pre_accept_close_remains_in_websocket_endpoints():
    """Every ``ws.close(code=44xx)`` before the endpoint's ``accept()`` must go
    through ``_ws_reject`` (structural guard, not a count freeze)."""
    import inspect
    import re

    source = inspect.getsource(web_server).split("\n")
    endpoints = [i for i, l in enumerate(source) if l.startswith("@app.websocket(")]
    offenders = []
    for start in endpoints:
        route = re.search(r'@app\.websocket\("([^"]+)"', source[start])
        if route and route.group(1) in _PRE_ACCEPT_CLOSE_ALLOWED:
            continue
        accept_at = next(
            (
                j
                for j in range(start, len(source))
                if "await ws.accept()" in source[j] or "handle_ws(ws)" in source[j]
            ),
            None,
        )
        for j in range(start, accept_at if accept_at is not None else len(source)):
            if "await ws.close(code=44" in source[j]:
                offenders.append(f"{j + 1}: {source[j].strip()}")
    assert offenders == []


# ---------------------------------------------------------------------------
# E2E on a real uvicorn server: the code + reason must reach a real client.
# ---------------------------------------------------------------------------


@pytest.fixture
def _ws_reject_server(monkeypatch, _isolate_hermes_home):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("websockets")

    previous = {
        name: getattr(web_server.app.state, name, None)
        for name in ("auth_required", "bound_host")
    }
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

    # Pre-bind so the port is known before the server thread starts.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    port = sock.getsockname()[1]

    config = uvicorn.Config(
        web_server.app, host="127.0.0.1", port=port, log_level="error",
        ws_ping_interval=None, ws_ping_timeout=None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True,
    )
    thread.start()
    deadline = threading.Event()
    for _ in range(200):
        if server.started:
            break
        deadline.wait(0.05)
    assert server.started, "uvicorn did not start"
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        with contextlib.suppress(OSError):
            sock.close()
        for name, value in previous.items():
            if value is None:
                if hasattr(web_server.app.state, name):
                    delattr(web_server.app.state, name)
            else:
                setattr(web_server.app.state, name, value)


def _connect_and_capture_close(url: str, **kwargs) -> tuple[int, str]:
    """Return ``(close_code, reason)`` as observed by a real WebSocket client."""
    import websockets

    async def run() -> tuple[int, str]:
        async with websockets.connect(url, open_timeout=5, **kwargs) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=5)
            except websockets.ConnectionClosed as exc:
                assert exc.rcvd is not None, "closed without a close frame"
                return exc.rcvd.code, exc.rcvd.reason
        raise AssertionError("server did not close the socket")

    return asyncio.run(run())


def test_real_server_delivers_4401_close_code_for_bad_token(_ws_reject_server):
    code, reason = _connect_and_capture_close(
        f"{_ws_reject_server}/api/console?token=wrong",
    )
    assert (code, reason) == (4401, "auth: token_mismatch")


def test_real_server_delivers_4403_close_code_for_foreign_host(_ws_reject_server):
    web_server.app.state.bound_host = "127.0.0.1"
    token = web_server._SESSION_TOKEN
    code, reason = _connect_and_capture_close(
        f"{_ws_reject_server}/api/events?token={token}&channel=security-test",
        additional_headers={
            "Host": "evil.example",
            "Origin": "http://evil.example",
        },
    )
    assert code == 4403
