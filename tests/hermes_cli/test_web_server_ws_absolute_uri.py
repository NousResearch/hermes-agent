import pytest

from hermes_cli.web_server import _WsAbsoluteUriMiddleware


@pytest.mark.asyncio
async def test_ws_absolute_uri_path_is_normalized_before_routing():
    captured_scope = {}

    async def downstream(scope, receive, send):
        captured_scope.update(scope)

    middleware = _WsAbsoluteUriMiddleware(downstream)
    scope = {
        "type": "websocket",
        "path": "ws://192.168.1.15:9119/api/ws",
        "raw_path": b"ws://192.168.1.15:9119/api/ws",
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
    assert scope["path"] == "ws://192.168.1.15:9119/api/ws"


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
