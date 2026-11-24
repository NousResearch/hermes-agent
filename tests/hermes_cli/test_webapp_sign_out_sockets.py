"""Signing the Webapp out closes every socket the old browser credential opened.

``hermes webapp --sign-out`` (control verb ``webapp-sign-out``) replaces the session token and closes
the live sockets that token authenticated, directly or through a ticket minted with it: each tab's
host terminal and event stream close at once with code 4401, and the media and download links it
minted stop working. A socket opened with another credential stays connected. The verb runs on a
control-socket worker thread, not on the sockets' event loops.
"""
import json
from types import SimpleNamespace

import anyio
import pytest
from starlette.testclient import TestClient

from tests.hermes_cli.test_web_host_terminal_persistence import metadata
from tests.hermes_cli.test_webapp_socket_tickets import _TOKEN, _mint, serve_webapp


def _next_message(ws, seconds=5.0):
    """The next frame the server sent *ws*, or None when none arrives within *seconds*."""
    async def receive():
        with anyio.move_on_after(seconds):
            return await ws._send_rx.receive()
        return None
    return ws.portal.call(receive)


def _close_code(ws, seconds=5.0):
    """The code the server closed *ws* with, or None while it stays open."""
    while (message := _next_message(ws, seconds)) is not None:
        if message["type"] == "websocket.close":
            return message["code"]
    return None


@pytest.fixture
def webapp(tmp_path, monkeypatch):
    return serve_webapp(tmp_path, monkeypatch)


@pytest.fixture
def sign_out(webapp, tmp_path):
    """The ``webapp-sign-out`` verb of a gateway serving *webapp*, asked over its control socket."""
    from gateway.control_socket import GatewayControlServer
    from gateway.webapp_listener import TOKEN_PATH, register_control_verbs

    server = webapp.server
    home = tmp_path / ".hermes"
    runner = SimpleNamespace(
        session_control_server=GatewayControlServer(home),
        webapp_status={"state": "serving", "url": "http://localhost:9120", "reason": None},
        session_api=SimpleNamespace(app=SimpleNamespace(web=server)))
    register_control_verbs(runner, home)

    def ask():
        raw = json.dumps({"protocol": 1, "verb": "webapp-sign-out", "id": 1, "params": {}}).encode()
        answer = json.loads(runner.session_control_server.handle_request_line(raw, "same-user"))
        assert answer["ok"], answer
        assert (home / TOKEN_PATH).read_text(encoding="utf-8").strip() == server._SESSION_TOKEN != _TOKEN
        return answer["result"]
    return ask


def test_signing_out_closes_the_terminal_and_streams_the_session_token_opened(webapp, sign_out):
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket

    client, host = webapp.client, webapp.host
    other = mint_ticket(user_id="someone-else", provider="stub")
    with TestClient(host) as shells, \
            shells.websocket_connect(f"ws://localhost/api/host-terminal?ticket={_mint(client, 'host-terminal')}") as shell, \
            client.websocket_connect(f"ws://localhost/api/events?channel=tab-1&token={_TOKEN}") as events, \
            client.websocket_connect(f"ws://localhost/api/events?channel=kept&ticket={other}") as kept:
        metadata(shell)
        assert _close_code(shell, 0.2) is None and _close_code(events, 0.2) is None
        assert sign_out() == {"signed_out": True, "serving": True, "sign_in_gate": False, "connections_closed": 2}
        assert _close_code(shell) == 4401
        assert _close_code(events) == 4401
        assert _close_code(kept, 0.5) is None
        # The shell itself outlives its viewer, as after any disconnect, until its retention ends.
        assert len(webapp.shells) == 1 and webapp.shells[0].is_alive()


def test_signing_out_with_no_open_tabs_still_replaces_the_credential(webapp, sign_out):
    client = webapp.client
    assert sign_out() == {"signed_out": True, "serving": True, "sign_in_gate": False, "connections_closed": 0}
    assert client.post("/api/webapp/socket-ticket?socket=gateway", headers={"X-Hermes-Session-Token": _TOKEN}).status_code == 401


@pytest.mark.parametrize("route", ["stream", "download"])
def test_signing_out_ends_the_file_links_the_session_token_minted(webapp, sign_out, tmp_path, monkeypatch, route):
    """A media element's stream link is reusable for an hour and a download link for a few retries;
    after sign-out, with uses left, neither reads anything."""
    from hermes_cli import web_server_file_tickets

    client = webapp.client
    root = tmp_path / "files"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"hello")
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    minted = client.post("/api/files/ticket", json={"route": route, "path": str(root / "clip.mp4")},
                         headers={"X-Hermes-Session-Token": _TOKEN})
    assert minted.status_code == 200, minted.text
    link = {"path": str(root / "clip.mp4"), "ticket": minted.json()["ticket"]}
    endpoint = web_server_file_tickets.ROUTES[route]
    assert client.get(endpoint, params=link).content == b"hello"
    sign_out()
    assert client.get(endpoint, params=link).status_code == 401


def test_a_media_link_minted_while_signing_out_streams_nothing(webapp, sign_out, tmp_path, monkeypatch):
    """The request asking for the link was admitted with the old credential, and the sign-out
    finished before the link was stored: the link ends with the credential that asked for it."""
    from hermes_cli import web_server_file_tickets

    client = webapp.client
    root = tmp_path / "files"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"hello")
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    mint = web_server_file_tickets.mint

    def mint_after_sign_out(*args, **kwargs):
        sign_out()
        return mint(*args, **kwargs)

    monkeypatch.setattr(web_server_file_tickets, "mint", mint_after_sign_out)
    minted = client.post("/api/files/ticket", json={"route": "stream", "path": str(root / "clip.mp4")},
                         headers={"X-Hermes-Session-Token": _TOKEN})
    assert minted.status_code == 200, minted.text
    stream = {"path": str(root / "clip.mp4"), "ticket": minted.json()["ticket"]}
    assert client.get("/api/files/stream", params=stream).status_code == 401


@pytest.mark.parametrize("route", ["stream", "download"])
def test_a_file_link_another_credential_minted_keeps_working_after_sign_out(webapp, sign_out, tmp_path,
                                                                          monkeypatch, route):
    """Signing out ends the links minted under the browser credential and no others: a link minted
    with another credential (a service token, a native client's grant) keeps working."""
    from hermes_cli import web_server_file_tickets

    client = webapp.client
    root = tmp_path / "files"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"hello")
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    path = str(root / "clip.mp4")
    ticket = web_server_file_tickets.mint(
        route, {"path": path}, authority=web_server_file_tickets.FileAuthority(root / "clip.mp4", ()))
    link = {"path": path, "ticket": ticket}
    endpoint = web_server_file_tickets.ROUTES[route]
    assert client.get(endpoint, params=link).content == b"hello"
    sign_out()
    assert client.get(endpoint, params=link).content == b"hello"


def test_a_socket_admitted_while_the_token_is_replaced_closes_at_once(monkeypatch):
    """The gate checked the old token, then sign-out replaced it and closed the sockets it knew of
    before this one was tracked: tracking it closes it."""
    import asyncio

    from hermes_cli import browser_credential, web_server, web_session_sockets

    closed = []

    class Socket:
        def __init__(self):
            self.ended = asyncio.Event()

        async def close(self, code, reason):
            closed.append((code, reason))
            self.ended.set()

    async def admit():
        socket = Socket()
        web_session_sockets.track(socket, browser_credential.generation_of(_TOKEN))
        await asyncio.wait_for(socket.ended.wait(), timeout=5)

    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "b" * 43)
    asyncio.run(admit())
    assert closed == [(4401, "Signed out")]
