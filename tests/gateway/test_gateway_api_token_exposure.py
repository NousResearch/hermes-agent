"""The always-on gateway HTTP listener must not hand its bearer token to any loopback peer.

Local clients authenticate with single-use tickets minted on the owner-only control socket
(SO_PEERCRED). ``GET /`` on the dashboard SPA injects ``window.__HERMES_SESSION_TOKEN__``
whenever the auth gate is off, and the gateway listener runs gate-off on loopback, so any
other OS user on the host could read the token and use every ``/api/*`` route
(``/api/env/reveal``) and ``/api/ws?token=`` without ever touching the socket.
"""
import types

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient


@pytest.mark.asyncio
async def test_gateway_listener_never_publishes_the_session_token(tmp_path, monkeypatch):
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server

    dist = tmp_path / "web_dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><head></head><body>spa</body></html>", encoding="utf-8")
    monkeypatch.setattr(web_server, "WEB_DIST", dist)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    # The process app mounted its SPA at import against the packaged dist; mount the same
    # routes over a built dist so GET / renders the bootstrap script the token rides in.
    spa = FastAPI()
    web_server.mount_spa(spa)
    assert web_server._SESSION_TOKEN in TestClient(spa).get("/").text  # standalone dashboard: unchanged
    runner = types.SimpleNamespace(session_runtime_descriptor=None, session_authority=None)

    handle = await start_gateway_api(runner)
    try:
        async with httpx.AsyncClient(base_url=handle.api_origin, trust_env=False) as client:
            assert (await client.get("/api/status")).status_code != 500
        page = TestClient(spa).get("/")
        assert page.status_code == 200 and "spa" in page.text
        assert web_server._SESSION_TOKEN not in page.text
    finally:
        await stop_gateway_api(handle)
    assert web_server._SESSION_TOKEN in TestClient(spa).get("/").text
