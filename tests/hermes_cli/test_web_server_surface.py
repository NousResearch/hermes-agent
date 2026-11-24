"""Routes only the Webapp or native Desktop's Web Access pane use exist only on the surfaces that use them.

One dashboard app serves every surface: the standalone ``hermes dashboard``, the headless
``hermes serve``, the gateway's own listener for native clients, and the gateway's Webapp. The surface
policy (``hermes_cli.web_server_surface``) says which of them serve:

- ``/api/webapp-access/*``: native Desktop's Web Access pane, which calls the gateway's own listener
  (the private one, or the Webapp's when it is on);
- the browser's file routes, ``/api/chat/file-upload`` and ``/api/files/ticket``, and the cap on chat
  upload bodies before they are parsed: the Webapp only.

Everywhere else these routes answer 404 before their handlers run, and chat uploads reach their
handlers as on main.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from tests.hermes_cli.native_gateway_client import native_desktop_client

WEB_ACCESS = (("get", "/api/webapp-access/status", None), ("post", "/api/webapp-access/start", {}),
              ("post", "/api/webapp-access/stop", None), ("post", "/api/webapp-access/restart", None))


@pytest.fixture(autouse=True)
def nothing_acts(monkeypatch):
    """No route under test may build, register, write config or restart a gateway: a handler that
    runs answers 500 instead of acting."""
    import hermes_cli.web_server_gateway as gateway
    from hermes_cli import webapp, webapp_access

    def refuse(*_args, **_kwargs):
        raise AssertionError("a Web Access handler ran")

    monkeypatch.setattr(gateway, "_restart_gateway_after", refuse)
    monkeypatch.setattr(webapp, "prepare_webapp_renderer", refuse)
    monkeypatch.setattr(webapp_access, "turn_off", refuse)
    monkeypatch.setattr(webapp_access, "access_status", lambda: {"enabled": False})
    for key in ("ui_surface", "gateway_runner", "listener_public_url", "webapp_home"):
        if hasattr(web_server.app.state, key):
            monkeypatch.delattr(web_server.app.state, key)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)


def _owner_client() -> TestClient:
    """The loopback session token's holder: the standalone dashboard's page, ``hermes serve``'s
    spawner, or a tab of the loopback Webapp."""
    client = TestClient(web_server.app, raise_server_exceptions=False)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


def _browser_file_statuses(client: TestClient) -> tuple[int, int]:
    upload = client.post("/api/chat/file-upload", files={"file": ("note.txt", b"hello")})
    ticket = client.post("/api/files/ticket", json={})
    return upload.status_code, ticket.status_code


def _web_access_statuses(client: TestClient) -> list[int]:
    return [getattr(client, method)(path, **({} if body is None else {"json": body})).status_code
            for method, path, body in WEB_ACCESS]


@pytest.mark.parametrize("serves_spa", [True, False], ids=["hermes-dashboard", "hermes-serve"])
def test_the_standalone_dashboard_and_serve_have_neither_web_access_nor_browser_file_routes(
        monkeypatch, serves_spa):
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(web_server.app.state, "serves_spa", serves_spa, raising=False)
    client = _owner_client()

    assert _browser_file_statuses(client) == (404, 404)
    assert _web_access_statuses(client) == [404] * len(WEB_ACCESS)
    assert not (get_hermes_home() / "uploads").exists()


def test_the_gateways_own_listener_serves_web_access_but_not_the_browser_file_routes(monkeypatch):
    """With the Webapp off, native Desktop's Web Access pane works on the gateway's listener; the
    browser's file routes, which only the Webapp calls, are not there."""
    from hermes_constants import get_hermes_home

    native = native_desktop_client(monkeypatch)

    assert native.get("/api/webapp-access/status").status_code == 200
    assert native.post("/api/webapp-access/start", json={}).status_code == 422  # reached its handler
    assert _browser_file_statuses(native) == (404, 404)
    assert not (get_hermes_home() / "uploads").exists()


def test_the_webapp_serves_web_access_and_the_browser_file_routes(monkeypatch):
    native = native_desktop_client(monkeypatch)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)

    assert native.get("/api/webapp-access/status").status_code == 200
    upload, ticket = _browser_file_statuses(_owner_client())
    assert (upload, ticket) == (200, 422)


@pytest.mark.parametrize(("surface", "capped"), [(None, False), ("webapp", True)])
def test_chat_upload_bodies_are_capped_before_parsing_only_on_the_webapp(surface, capped):
    """An image paste larger than the cap reaches ``/api/chat/image-upload``'s own handler on every
    surface but the Webapp, as on main; the Webapp refuses it with 413 before reading the body."""
    import asyncio

    from hermes_cli.web_upload_limit import CHAT_IMAGE_BODY_MAX_BYTES, ChatFileUploadLimitMiddleware

    reached, sent = [], []

    async def handler(scope, receive, send):
        reached.append(scope["path"])

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    state = SimpleNamespace() if surface is None else SimpleNamespace(ui_surface=surface)
    scope = {"type": "http", "method": "POST", "path": "/api/chat/image-upload", "root_path": "",
             "headers": [(b"content-length", str(CHAT_IMAGE_BODY_MAX_BYTES + 1).encode())],
             "app": SimpleNamespace(state=state)}
    asyncio.run(ChatFileUploadLimitMiddleware(handler)(scope, receive, send))

    assert reached == ([] if capped else ["/api/chat/image-upload"])
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == ([413] if capped else [])
