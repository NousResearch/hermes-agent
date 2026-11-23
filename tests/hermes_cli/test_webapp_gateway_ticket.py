"""The Webapp's live socket ticket is minted by the server that serves the page.

``POST /api/webapp/socket-ticket?socket=gateway&profile=<name>`` hands the browser a one-use ticket for
``/api/ws`` on the same origin (offered as the ``hermes-gateway-ticket.`` subprotocol). It names the
caller's login (and the profile it names, ``test_webapp_ticket_profile.py``), and the upgrade gate
redeems it exactly once.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_session_sockets import GATEWAY_SOCKET

_MINT = "/api/webapp/socket-ticket?socket=gateway"

_TOKEN = "a" * 43


@pytest.fixture
def webapp(monkeypatch, tmp_path):
    from hermes_cli import web_server as server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", _TOKEN)
    return server, TestClient(server.app, base_url="http://127.0.0.1")


def _socket(ticket):
    """A stand-in for the browser's ``/api/ws`` upgrade, good enough for ``_ws_auth_ok``."""
    from hermes_cli import web_server_chat as chat

    class _Query(dict):
        def get(self, key, default=""):
            return super().get(key, default)

    protocols = f"{chat._GATEWAY_WS_PROTOCOL}, {chat._GATEWAY_WS_TICKET_PROTOCOL_PREFIX}{ticket}"
    return SimpleNamespace(query_params=_Query(native_dial="browser"), headers={"sec-websocket-protocol": protocols},
                           client=SimpleNamespace(host="127.0.0.1"), url=SimpleNamespace(path="/api/ws"))


def test_loopback_ticket_names_the_session_token_owner_and_is_spent_once(webapp):
    from hermes_cli import web_server_chat as chat

    _server, client = webapp
    assert client.post(_MINT).status_code == 401
    minted = client.post(_MINT, headers={"X-Hermes-Session-Token": _TOKEN})
    assert minted.status_code == 200, minted.text
    assert minted.headers["cache-control"] == "no-store"
    body = minted.json()
    assert set(body) == {"ticket", "ttl_seconds"} and body["ttl_seconds"] > 0
    first = _socket(body["ticket"])
    assert chat._ws_auth_ok(first, socket=GATEWAY_SOCKET) is True
    # The same identity ``chat_ws.gateway_ws`` gives a ``?token=`` socket, and only the public protocol.
    assert first._hermes_auth_identity == {"user_id": "legacy-token-owner", "provider": "session-token"}
    assert first._hermes_ws_subprotocol == chat._GATEWAY_WS_PROTOCOL
    assert not hasattr(first, "_hermes_gateway_home")
    assert chat._ws_auth_ok(_socket(body["ticket"]), socket=GATEWAY_SOCKET) is False


def test_a_missing_profile_is_refused_before_a_ticket_exists(webapp):
    _server, client = webapp
    refused = client.post(_MINT + "&profile=missing", headers={"X-Hermes-Session-Token": _TOKEN})
    assert refused.status_code == 404


@pytest.mark.parametrize("surface", ["dashboard", "serve"])
def test_other_surfaces_do_not_mint_webapp_socket_tickets(webapp, monkeypatch, surface):
    server, client = webapp
    monkeypatch.setattr(server.app.state, "ui_surface", surface)
    refused = client.post(_MINT, headers={"X-Hermes-Session-Token": _TOKEN})
    assert refused.status_code == 404
