"""The tailcat share listener admits paired devices only.

``tailcat serve`` delivers every remote peer from 127.0.0.1, so nothing that
the main listener trusts about loopback callers may reach the app through it.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import tailcat_share_store as store
from hermes_cli.web_server_share import PAIR_PATH, ShareGate

PROCESS_TOKEN = "process-session-token"
UPSTREAM = "127.0.0.1:9119"


async def _echo(request):
    return JSONResponse({
        "token": request.headers.get("x-hermes-session-token"),
        "host": request.headers.get("host"),
        "cookie": request.headers.get("cookie"),
    })


async def _ws(websocket):
    await websocket.accept()
    await websocket.send_json({"token": websocket.query_params.get("token")})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "get_routing_process_hermes_home", lambda: tmp_path)
    inner = Starlette(routes=[
        Route("/", _echo), Route("/api/sessions", _echo), Route("/api/share/status", _echo),
        WebSocketRoute("/api/ws", _ws),
    ])
    gate = ShareGate(inner, session_token=lambda: PROCESS_TOKEN, upstream_host=lambda: UPSTREAM)
    return TestClient(gate)


def _pair(client) -> str:
    code = store.mint_code("tc" + "A" * 40, 41234)
    reply = client.post(PAIR_PATH, json={"code": code.render(), "name": "laptop"})
    assert reply.status_code == 200
    return reply.json()["token"]


def test_nothing_reaches_the_app_without_a_paired_token(client):
    assert client.get("/").status_code == 404
    assert client.get("/api/sessions").status_code == 401
    # The process token is what the main listener accepts; the share must not.
    assert client.get("/api/sessions", headers={"X-Hermes-Session-Token": PROCESS_TOKEN}).status_code == 401
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/api/ws?token={PROCESS_TOKEN}") as ws:
            ws.receive_json()


def test_paired_device_is_recredentialed_but_never_reaches_owner_routes(client):
    token = _pair(client)

    reply = client.get("/api/sessions", headers={"X-Hermes-Session-Token": token, "Cookie": "a=b"})
    assert reply.status_code == 200
    assert reply.json() == {"token": PROCESS_TOKEN, "host": UPSTREAM, "cookie": None}
    with client.websocket_connect(f"/api/ws?token={token}") as ws:
        assert ws.receive_json() == {"token": PROCESS_TOKEN}
    # Minting codes and revoking devices stay with the machine's owner.
    assert client.get("/api/share/status", headers={"X-Hermes-Session-Token": token}).status_code == 401


def test_a_code_pairs_one_device_and_revoking_it_ends_access(client):
    code = store.mint_code("tc" + "A" * 40, 41234)
    first = client.post(PAIR_PATH, json={"code": code.render()})
    again = client.post(PAIR_PATH, json={"code": code.render()})
    assert (first.status_code, again.status_code) == (200, 403)

    token = first.json()["token"]
    store.revoke_device(first.json()["device"]["id"])
    assert client.get("/api/sessions", headers={"X-Hermes-Session-Token": token}).status_code == 401
