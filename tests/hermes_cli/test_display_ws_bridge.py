"""Regression for #141629: a reachability failure inside the display bridge arrives at the renderer
as the 4001 "Bot Desktop is not running" close frame, for OSError AND RuntimeError. Before the fix,
open_rfb_stream()'s RuntimeError (marker-named container, empty terminal env registry) escaped
`except OSError` and killed the WebSocket with no close code — a silent black pane."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli.dashboard_auth import ws_tickets
from hermes_cli.web_routers import display


@pytest.mark.parametrize("exc", [OSError("rfb.sock missing"),
                                 RuntimeError("the sandbox hosting this screen is not running")])
def test_a_reachability_failure_closes_with_desktop_gone(monkeypatch, exc):
    from hermes_cli import web_server

    ws_tickets._reset_for_tests()
    prev = {k: getattr(web_server.app.state, k, None) for k in ("auth_required", "bound_host")}
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    try:
        ticket = ws_tickets.mint_ticket(user_id="display:v", provider="bot-desktop",
                                        extra={"hermes_home": "/srv/hermes/bot-a", "viewer_id": "v"})

        def _refuse(_profile_home):
            raise exc

        monkeypatch.setattr(display, "_open_rfb", _refuse)
        with client.websocket_connect(f"/api/display/ws?display_ticket={ticket}") as conn:
            with pytest.raises(WebSocketDisconnect) as closed:
                conn.receive_bytes()
        assert closed.value.code == display._CLOSE_DESKTOP_GONE
    finally:
        client.close()
        ws_tickets._reset_for_tests()
        for k, v in prev.items():
            if v is None:
                if hasattr(web_server.app.state, k):
                    delattr(web_server.app.state, k)
            else:
                setattr(web_server.app.state, k, v)
