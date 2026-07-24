"""#68802: profile_routes from_number / sender_id discriminator tests."""

from __future__ import annotations

import pytest

from gateway.profile_routing import (
    ProfileRoute,
    parse_profile_routes,
    match_profile_route,
)


class TestSenderRouting:
    def test_from_number_match(self):
        r = ProfileRoute(
            name="work", platform="whatsapp", profile="work",
            from_number="+15551234567",
        )
        assert r.matches("whatsapp", sender_id="+15551234567")
        assert not r.matches("whatsapp", sender_id="+15559999999")
        assert not r.matches("whatsapp")  # no sender → no match

    def test_sender_id_alias(self):
        r = ProfileRoute(
            name="sig", platform="signal", profile="family",
            sender_id="uuid-abcd",
        )
        assert r.matches("signal", sender_id="uuid-abcd")
        assert not r.matches("signal", sender_id="uuid-xyz")

    def test_specificity_sender(self):
        r = ProfileRoute(
            name="s", platform="whatsapp", profile="p",
            from_number="+15551234567",
        )
        assert r.specificity == 5

    def test_specificity_sender_plus_chat(self):
        r = ProfileRoute(
            name="sc", platform="whatsapp", profile="p",
            chat_id="123", from_number="+15551234567",
        )
        assert r.specificity == 9  # 4 + 5

    def test_sender_more_specific_than_guild(self):
        """from_number (5) should rank between chat (6) and guild (2)."""
        sender_route = ProfileRoute(
            name="sender", platform="whatsapp", profile="work",
            from_number="+15551234567",
        )
        guild_route = ProfileRoute(
            name="guild", platform="discord", profile="server",
            guild_id="111",
        )
        assert sender_route.specificity > guild_route.specificity

    def test_no_sender_match_falls_through(self):
        routes = [
            ProfileRoute(name="work", platform="whatsapp", profile="work",
                         from_number="+15551234567"),
            ProfileRoute(name="default", platform="whatsapp", profile="family"),
        ]
        # Unknown sender → no from_number match → falls to default
        matched = match_profile_route(routes, "whatsapp", sender_id="+19999999999")
        assert matched is not None
        assert matched.profile == "family"

    def test_parse_from_number(self):
        raw = [{"name": "w", "platform": "whatsapp", "profile": "work",
                "from_number": "+15551234567"}]
        routes = parse_profile_routes(raw)
        assert len(routes) == 1
        assert routes[0].from_number == "+15551234567"

    def test_parse_sender_id(self):
        raw = [{"name": "s", "platform": "signal", "profile": "fam",
                "sender_id": "uuid-xyz"}]
        routes = parse_profile_routes(raw)
        assert len(routes) == 1
        assert routes[0].sender_id == "uuid-xyz"

    def test_existing_routes_still_match_without_sender(self):
        """Existing routes without from_number/sender_id must still work."""
        r = ProfileRoute(name="c", platform="discord", profile="helper",
                         chat_id="222")
        assert r.matches("discord", chat_id="222")