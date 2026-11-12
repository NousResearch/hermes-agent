"""The display bridge admits only a display ticket minted for THIS profile's socket: a gateway ticket,
an expired ticket, or one for another provider is refused before any socket is dialled."""

from __future__ import annotations

from hermes_cli.dashboard_auth import ws_tickets
from hermes_cli.web_routers import display


class _Ws:
    def __init__(self, **params):
        self.query_params = params


def test_display_ticket_must_be_a_bot_desktop_ticket_pinned_to_a_profile_home(monkeypatch):
    ws_tickets._reset_for_tests()
    gateway_ticket = ws_tickets.mint_ticket(user_id="u", provider="google")
    assert display._consume_display_ticket(_Ws(display_ticket=gateway_ticket)) is None

    unpinned = ws_tickets.mint_ticket(user_id="display:v", provider="bot-desktop")
    assert display._consume_display_ticket(_Ws(display_ticket=unpinned)) is None

    good = ws_tickets.mint_ticket(user_id="display:v", provider="bot-desktop",
                                  extra={"hermes_home": "/srv/hermes/bot-a", "viewer_id": "v"})
    info = display._consume_display_ticket(_Ws(display_ticket=good))
    assert info and info["hermes_home"] == "/srv/hermes/bot-a" and info["viewer_id"] == "v"
    assert display._consume_display_ticket(_Ws(display_ticket=good)) is None, "single use"


def test_a_bad_ticket_is_refused_with_a_close_frame_the_renderer_can_read(monkeypatch):
    """Closing BEFORE accept surfaces to a browser as an HTTP 403 handshake failure with no close
    code; the renderer's close listener never learns WHY (4401 = ticket, 4001 = desktop gone) and
    cannot pick between re-observing and showing "not running". The handshake must complete and
    the reason travel in the close frame."""
    import pytest
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from hermes_cli import web_server

    ws_tickets._reset_for_tests()
    prev = {k: getattr(web_server.app.state, k, None) for k in ("auth_required", "bound_host")}
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    try:
        used = ws_tickets.mint_ticket(user_id="display:v", provider="bot-desktop",
                                      extra={"hermes_home": "/srv/hermes/bot-a", "viewer_id": "v"})
        ws_tickets.consume_ticket(used)
        with client.websocket_connect(f"/api/display/ws?display_ticket={used}") as conn:  # handshake completes
            with pytest.raises(WebSocketDisconnect) as exc:
                conn.receive_bytes()
        assert exc.value.code == 4401
    finally:
        client.close()
        ws_tickets._reset_for_tests()
        for k, v in prev.items():
            if v is None:
                if hasattr(web_server.app.state, k):
                    delattr(web_server.app.state, k)
            else:
                setattr(web_server.app.state, k, v)


def _viewer_app(monkeypatch):
    """The display route on the loopback Webapp; the RFB bridge stands in as a viewer that stays
    connected until the server closes it."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from hermes_cli import web_server

    async def watch(ws, _info):
        while (await ws.receive())["type"] != "websocket.disconnect":
            pass

    monkeypatch.setattr(display, "_bridge", watch)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "a" * 43)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", None, raising=False)
    app = FastAPI()
    app.include_router(display.router)
    return TestClient(app)


def _viewer_ticket(**extra):
    return ws_tickets.mint_ticket(user_id="display:v", provider="bot-desktop",
                                  extra={"hermes_home": "/srv/hermes/bot-a", "viewer_id": "v", **extra})


def test_signing_out_closes_a_screen_viewer_a_browser_tab_opened(monkeypatch):
    """A viewer opened from a signed-in tab watches, and while it holds the lease drives, the agent's
    screen: it closes with the tab's other sockets when the browser signs out. A viewer a native
    client opened (no browser credential on its ticket) stays."""
    from hermes_cli import browser_credential, web_server, web_session_sockets
    from tests.hermes_cli.test_webapp_sign_out_sockets import _close_code

    ws_tickets._reset_for_tests()
    client = _viewer_app(monkeypatch)
    tab = _viewer_ticket(admitted_with=browser_credential.generation_of("a" * 43))
    native = _viewer_ticket()
    with client.websocket_connect(f"/api/display/ws?display_ticket={tab}") as from_tab, \
            client.websocket_connect(f"/api/display/ws?display_ticket={native}") as from_native:
        assert _close_code(from_tab, 0.2) is None
        monkeypatch.setattr(web_server, "_SESSION_TOKEN", "b" * 43)
        web_session_sockets.close_all()
        assert _close_code(from_tab) == 4401
        assert _close_code(from_native, 0.5) is None


def test_a_screen_viewer_ticket_from_a_signed_out_tab_opens_nothing(monkeypatch):
    from hermes_cli import browser_credential, web_server
    from tests.hermes_cli.test_webapp_sign_out_sockets import _close_code

    ws_tickets._reset_for_tests()
    client = _viewer_app(monkeypatch)
    stale = _viewer_ticket(admitted_with=browser_credential.generation_of("a" * 43))
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "b" * 43)
    with client.websocket_connect(f"/api/display/ws?display_ticket={stale}") as conn:
        assert _close_code(conn, 2.0) == 4401
