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
