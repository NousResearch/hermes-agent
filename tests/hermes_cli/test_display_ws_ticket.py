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
