import pytest

from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
from hermes_cli.web_server import _WsAbsoluteUriMiddleware


@pytest.mark.asyncio
async def test_ws_absolute_uri_path_is_normalized_before_routing():
    captured_scope = {}

    async def downstream(scope, receive, send):
        captured_scope.update(scope)

    middleware = _WsAbsoluteUriMiddleware(downstream)
    scope = {
        "type": "websocket",
        "path": "ws://192.168.1.15:9119/api/ws?ticket=abc123",
        "raw_path": b"ws://192.168.1.15:9119/api/ws?ticket=abc123",
        "query_string": b"",
        "client": ("192.168.1.9", 62493),
        "headers": [],
    }

    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(message):
        return None

    await middleware(scope, receive, send)

    assert captured_scope["path"] == "/api/ws"
    assert captured_scope["raw_path"] == b"/api/ws"
    assert captured_scope["query_string"] == b"ticket=abc123"
    assert scope["path"] == "ws://192.168.1.15:9119/api/ws?ticket=abc123"


@pytest.mark.asyncio
async def test_ws_origin_form_path_is_left_unchanged():
    captured_scope = {}

    async def downstream(scope, receive, send):
        captured_scope.update(scope)

    middleware = _WsAbsoluteUriMiddleware(downstream)
    scope = {
        "type": "websocket",
        "path": "/api/events",
        "raw_path": b"/api/events",
        "client": ("127.0.0.1", 12345),
        "headers": [],
    }

    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(message):
        return None

    await middleware(scope, receive, send)

    assert captured_scope["path"] == "/api/events"
    assert captured_scope["raw_path"] == b"/api/events"


def _make_scope(path: str) -> dict:
    return {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "scheme": "ws",
        "http_version": "1.1",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "client": ("192.168.1.24", 53142),
        "server": ("192.168.1.15", 9119),
        "headers": [
            (b"host", b"192.168.1.15:9119"),
            (b"origin", b"http://192.168.1.15:9119"),
        ],
        "subprotocols": [],
    }


async def _drive_app(app, scope: dict) -> list[dict]:
    sent: list[dict] = []
    received = iter(
        [
            {"type": "websocket.connect"},
            {"type": "websocket.disconnect", "code": 1000},
        ]
    )

    async def receive():
        try:
            return next(received)
        except StopIteration:
            return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


@pytest.mark.asyncio
async def test_absolute_form_ws_route_accepts_ticket(monkeypatch):
    import hermes_cli.web_server as web_server

    async def fake_handle_ws(ws):
        await ws.accept()
        await ws.close(code=1000)

    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(
        web_server.app.state, "bound_host", "192.168.1.15", raising=False
    )
    monkeypatch.setattr("tui_gateway.ws.handle_ws", fake_handle_ws)

    ticket = mint_ticket(user_id="pytest", provider="pytest")
    sent = await _drive_app(
        web_server.app,
        _make_scope(f"ws://192.168.1.15:9119/api/ws?ticket={ticket}"),
    )

    assert any(msg["type"] == "websocket.accept" for msg in sent)
    assert sent[-1]["type"] == "websocket.close"
    assert sent[-1]["code"] == 1000
    assert all(msg.get("code") not in {4401, 4403, 4408} for msg in sent)


@pytest.mark.asyncio
async def test_absolute_form_pty_route_accepts_ticket(monkeypatch):
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(
        web_server.app.state, "bound_host", "192.168.1.15", raising=False
    )
    monkeypatch.setattr(web_server, "_PTY_BRIDGE_AVAILABLE", False)

    ticket = mint_ticket(user_id="pytest", provider="pytest")
    sent = await _drive_app(
        web_server.app,
        _make_scope(f"ws://192.168.1.15:9119/api/pty?ticket={ticket}"),
    )

    assert any(msg["type"] == "websocket.accept" for msg in sent)
    assert any(msg["type"] == "websocket.send" for msg in sent)
    assert sent[-1]["type"] == "websocket.close"
    assert sent[-1]["code"] == 1011
    assert all(msg.get("code") not in {4401, 4403, 4408} for msg in sent)
