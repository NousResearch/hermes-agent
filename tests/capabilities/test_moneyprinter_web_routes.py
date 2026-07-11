import base64

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient as AiohttpTestClient, TestServer


@pytest.fixture
def clients(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(app.state, "auth_required", False, raising=False)
    anonymous = TestClient(app)
    authenticated = TestClient(app)
    authenticated.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return anonymous, authenticated


def test_minimax_clone_route_requires_session_and_reaches_adapter(clients, monkeypatch):
    from capabilities.moneyprinter import adapter

    anonymous, authenticated = clients
    captured = {}

    async def fake_proxy_json(method, upstream_path, body=None, **kwargs):
        captured.update({"body": body, "method": method, "upstream_path": upstream_path})
        return 200, {"status": 200, "data": {"voice_id": "MiniMaxDemo001"}}

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)
    body = {
        "clone_audio": {
            "contentBase64": base64.b64encode(b"clone-audio").decode("ascii"),
            "filename": "clone.wav",
        },
        "voice_id": "MiniMaxDemo001",
    }

    assert anonymous.post("/api/capabilities/moneyprinter/minimax/voices/clone", json=body).status_code == 401

    response = authenticated.post("/api/capabilities/moneyprinter/minimax/voices/clone", json=body)

    assert response.status_code == 200
    assert response.json()["data"]["voice_id"] == "MiniMaxDemo001"
    assert captured["method"] == "POST"
    assert captured["upstream_path"] == "/api/v1/minimax/voices/clone"
    assert "sourcePath" not in captured["body"]["clone_audio"]


def test_minimax_music_route_requires_session_and_reaches_adapter(clients, monkeypatch):
    from capabilities.moneyprinter import adapter

    anonymous, authenticated = clients
    captured = {}

    async def fake_proxy_json(method, upstream_path, body=None, **kwargs):
        captured.update({"body": body, "method": method, "upstream_path": upstream_path})
        return 200, {"status": 200, "data": {"bgm": {"file": "generated.mp3"}}}

    monkeypatch.setattr(adapter, "_proxy_json", fake_proxy_json)
    body = {"is_instrumental": True, "prompt": "technology intro", "save_as_bgm": True}

    assert anonymous.post("/api/capabilities/moneyprinter/minimax/music", json=body).status_code == 401

    response = authenticated.post("/api/capabilities/moneyprinter/minimax/music", json=body)

    assert response.status_code == 200
    assert response.json()["data"]["bgm"]["file"] == "generated.mp3"
    assert captured["method"] == "POST"
    assert captured["upstream_path"] == "/api/v1/minimax/music"


def test_minimax_route_surface_is_complete():
    from hermes_cli.web_server import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert {
        "/api/capabilities/moneyprinter/minimax/lyrics",
        "/api/capabilities/moneyprinter/minimax/music",
        "/api/capabilities/moneyprinter/minimax/tts",
        "/api/capabilities/moneyprinter/minimax/voices",
        "/api/capabilities/moneyprinter/minimax/voices/clone",
    }.issubset(paths)


@pytest.mark.asyncio
async def test_api_server_body_limit_only_allows_large_clone_requests():
    from gateway.platforms.api_server import (
        MAX_MONEYPRINTER_CLONE_REQUEST_BYTES,
        MAX_REQUEST_BYTES,
        MONEYPRINTER_CLONE_PATH,
        body_limit_middleware,
    )

    async def read_body(request):
        return web.json_response({"size": len(await request.read())})

    app = web.Application(
        middlewares=[body_limit_middleware],
        client_max_size=MAX_MONEYPRINTER_CLONE_REQUEST_BYTES,
    )
    app.router.add_post(MONEYPRINTER_CLONE_PATH, read_body)
    app.router.add_post("/normal", read_body)
    client = AiohttpTestClient(TestServer(app))
    await client.start_server()
    payload = b"x" * (MAX_REQUEST_BYTES + 1)
    try:
        clone_response = await client.post(MONEYPRINTER_CLONE_PATH, data=payload)
        normal_response = await client.post("/normal", data=payload)

        assert clone_response.status == 200
        assert (await clone_response.json())["size"] == len(payload)
        assert normal_response.status == 413
    finally:
        await client.close()
