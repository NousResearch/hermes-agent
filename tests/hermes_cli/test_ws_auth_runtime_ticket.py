"""Loopback-mode WS auth must accept the gateway's own runtime ticket.

A gateway-hosted dashboard serves the sibling streams the Desktop opens on the
same listener: the live dictation socket ``/api/audio/transcribe-stream`` and
the TTS socket ``/api/audio/speak-stream``. Those sockets are plain
WebSockets, so the Desktop cannot hand them a header or a subprotocol - it puts
its one-use ``interactive`` gateway ticket in the query string instead.

The runtime ticket store (``gateway/runtime_bootstrap.py``) and the dashboard's
browser-ticket store (``dashboard_auth/ws_tickets.py``) are different issuers,
so before this the query ticket was refused and the audio streams could never
authenticate on a loopback bind: live dictation silently fell back to the
file path and TTS fell through to ``POST /api/audio/speak``.

These tests pin the contract: the loopback branch redeems the same runtime
store the core ``/api/ws`` socket uses, and refuses a ticket that is not one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import web_server
import hermes_cli.web_server_chat as _web_server_chat


class _StubStore:
    """Minimal stand-in for ``gateway.runtime_bootstrap.TicketStore``."""

    def __init__(self, grants: dict[str, str] | None = None) -> None:
        self.grants = grants or {}
        self.redeemed: list[tuple[str, str | None, str | None]] = []

    def redeem(self, ticket, *, profile_id=None, purpose=None):
        self.redeemed.append((ticket, profile_id, purpose))
        if ticket not in self.grants:
            raise PermissionError("invalid bootstrap ticket")
        return {"profile_id": self.grants[ticket], "purpose": purpose}


def _fake_ws(*, query: dict, path: str = "/api/audio/transcribe-stream"):
    """Stand-in for starlette.WebSocket good enough for _ws_auth_reason."""

    class _QP:
        def __init__(self, q):
            self._q = q

        def get(self, k, default=""):
            return self._q.get(k, default)

    return SimpleNamespace(
        query_params=_QP(query),
        headers={},
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path=path),
    )


@pytest.fixture
def loopback_app():
    """web_server.app in loopback mode with a stubbed runtime ticket store."""
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    prev_runner = getattr(web_server.app.state, "gateway_runner", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    store = _StubStore({"good-ticket": "/home/profile"})
    web_server.app.state.gateway_runner = SimpleNamespace(session_ticket_store=store)
    try:
        yield store
    finally:
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required
        web_server.app.state.gateway_runner = prev_runner


def test_loopback_accepts_gateway_runtime_ticket(loopback_app):
    """A valid runtime ticket authenticates a loopback WS upgrade."""
    ws = _fake_ws(query={"ticket": "good-ticket"})

    reason, credential = _web_server_chat._ws_auth_reason(ws)

    assert reason is None, f"runtime ticket refused: {reason}"
    assert credential == "ticket"
    assert ws._hermes_auth_identity["provider"] == "local"


def test_loopback_redeems_ticket_as_interactive_owner(loopback_app):
    """The ticket is redeemed against the same store/purpose ``/api/ws`` uses."""
    _web_server_chat._ws_auth_reason(_fake_ws(query={"ticket": "good-ticket"}))

    assert loopback_app.redeemed == [("good-ticket", None, "interactive")]


def test_loopback_rejects_unknown_ticket(loopback_app):
    """An unknown ticket is refused, not silently accepted."""
    reason, credential = _web_server_chat._ws_auth_reason(
        _fake_ws(query={"ticket": "bogus"}))

    assert reason == "ticket_invalid"
    assert credential == "ticket"


def test_loopback_without_any_credential_is_still_refused(loopback_app):
    """The no-credential answer is unchanged for a bare upgrade."""
    reason, credential = _web_server_chat._ws_auth_reason(_fake_ws(query={}))

    assert reason == "no_credential"
    assert credential == "none"


def test_loopback_session_token_still_wins_over_ticket(loopback_app):
    """A valid ``?token=`` is still accepted when a ticket rides along."""
    ws = _fake_ws(query={"token": web_server._SESSION_TOKEN, "ticket": "bogus"})

    reason, credential = _web_server_chat._ws_auth_reason(ws)

    assert reason is None
    assert credential == "token"
