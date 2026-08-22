"""Tests for gateway/profile_routing.py — profile-based routing."""

import json

import pytest
from gateway.profile_routing import (
    ProfileRoute,
    parse_profile_routes,
    match_profile_route,
)


class TestProfileRoute:
    def test_specificity_thread(self):
        r = ProfileRoute(name="t", platform="discord", profile="p",
                         guild_id="g", chat_id="c", thread_id="t")
        assert r.specificity == 14  # 2 + 4 + 8


    def test_frozen(self):
        r = ProfileRoute(name="x", platform="discord", profile="p")
        with pytest.raises(AttributeError):
            r.name = "y"


class TestProfileRouteMatching:
    def test_exact_thread_match(self):
        r = ProfileRoute(name="t", platform="discord", profile="trader",
                         guild_id="111", chat_id="222", thread_id="333")
        assert r.matches("discord", guild_id="111", chat_id="222", thread_id="333")
        assert not r.matches("discord", guild_id="111", chat_id="222", thread_id="444")


    def test_guild_and_chat_are_conjunctive(self):
        # A route declaring BOTH guild_id and chat_id requires both to match.
        # Regression guard: previously chat_id was checked first and returned
        # True before guild_id was ever consulted.
        r = ProfileRoute(name="gc", platform="discord", profile="scoped",
                         guild_id="111", chat_id="222")
        # Both match (direct channel) -> match
        assert r.matches("discord", guild_id="111", chat_id="222")
        # Both match via parent (thread inside the channel) -> match
        assert r.matches("discord", guild_id="111", chat_id="333", parent_chat_id="222")
        # chat matches but guild differs -> NO match (the bug this guards)
        assert not r.matches("discord", guild_id="999", chat_id="222")
        # guild matches but chat differs -> NO match
        assert not r.matches("discord", guild_id="111", chat_id="333")


class TestWhatsAppAliasMatching:
    """A WhatsApp route keyed on one id form must match the other.

    The bridge hands the same person over as either ``<msisdn>@s.whatsapp.net``
    or ``<lid>@lid``. Exact-string routing therefore misses a msisdn-keyed
    route the moment WhatsApp switches that sender to their LID, and the
    message lands on the default profile — observed live, with the operator's
    own DM falling out of its profile. Authorization resolves the two forms
    through ``gateway.whatsapp_identity``; routing reads the same mapping.
    """

    PHONE = "351912345678"
    LID = "77214955630717"

    def _write_lid_mapping(self):
        """Mirror what the JS bridge writes: phone→lid and lid→phone."""
        from hermes_constants import get_hermes_home

        session_dir = get_hermes_home() / "whatsapp" / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / f"lid-mapping-{self.PHONE}.json").write_text(
            json.dumps(self.LID), encoding="utf-8"
        )
        (session_dir / f"lid-mapping-{self.LID}_reverse.json").write_text(
            json.dumps(self.PHONE), encoding="utf-8"
        )

    def _route(self, chat_id, platform="whatsapp"):
        return ProfileRoute(
            name="admin-dm", platform=platform, profile="ops", chat_id=chat_id
        )

    def test_phone_route_matches_lid_chat_id(self):
        self._write_lid_mapping()
        route = self._route(f"{self.PHONE}@s.whatsapp.net")

        assert route.matches("whatsapp", chat_id=f"{self.LID}@lid")

    def test_lid_route_matches_phone_chat_id(self):
        self._write_lid_mapping()
        route = self._route(f"{self.LID}@lid")

        assert route.matches("whatsapp", chat_id=f"{self.PHONE}@s.whatsapp.net")

    def test_bare_phone_route_matches_lid_chat_id(self):
        """Operators write routes as plain phone numbers, not JIDs."""
        self._write_lid_mapping()
        route = self._route(f"+{self.PHONE}")

        assert route.matches("whatsapp", chat_id=f"{self.LID}@lid")

    def test_unmapped_sender_does_not_match(self):
        self._write_lid_mapping()
        route = self._route(f"{self.PHONE}@s.whatsapp.net")

        assert not route.matches("whatsapp", chat_id="99999999999999@lid")

    def test_without_a_mapping_file_matching_stays_exact(self):
        route = self._route(f"{self.PHONE}@s.whatsapp.net")

        assert not route.matches("whatsapp", chat_id=f"{self.LID}@lid")
        assert route.matches("whatsapp", chat_id=f"{self.PHONE}@s.whatsapp.net")

    def test_group_route_is_never_alias_matched(self):
        """Group JIDs are a different id space — a digit collision with a LID
        must not route a DM into the group's profile."""
        self._write_lid_mapping()
        route = self._route(f"{self.LID}@g.us")

        assert not route.matches("whatsapp", chat_id=f"{self.LID}@lid")
        assert route.matches("whatsapp", chat_id=f"{self.LID}@g.us")

    def test_other_platforms_keep_exact_matching(self):
        self._write_lid_mapping()
        route = self._route(f"{self.PHONE}@s.whatsapp.net", platform="telegram")

        assert not route.matches("telegram", chat_id=f"{self.LID}@lid")

    def test_match_profile_route_resolves_the_alias(self):
        """The alias set is resolved once for the whole route list."""
        self._write_lid_mapping()
        routes = [
            ProfileRoute(
                name="other", platform="whatsapp", profile="misc",
                chat_id="120363001234567890@g.us",
            ),
            self._route(f"{self.PHONE}@s.whatsapp.net"),
        ]

        matched = match_profile_route(
            routes, "whatsapp", chat_id=f"{self.LID}@lid"
        )

        assert matched is not None
        assert matched.profile == "ops"


class TestParseProfileRoutes:
    def test_empty(self):
        assert parse_profile_routes(None) == []
        assert parse_profile_routes([]) == []


class TestMatchProfileRoute:


    def test_no_match_returns_none(self):
        routes = [
            ProfileRoute(name="r", platform="telegram", profile="p"),
        ]
        assert match_profile_route(routes, "discord") is None


class TestSessionKeyIntegration:
    def test_default_profile_key(self):
        from gateway.session import build_session_key, SessionSource, Platform
        src = SessionSource(platform=Platform.DISCORD, chat_id="123",
                            chat_type="channel", user_id="456")
        key = build_session_key(src)
        assert key.startswith("agent:main:")


class TestParentChatIdMatching:
    """Thread messages carry thread_id as chat_id; parent_chat_id is the channel."""

    def test_channel_route_matches_via_parent_chat_id(self):
        r = ProfileRoute(name="ch", platform="discord", profile="trader",
                         chat_id="222")
        assert r.matches("discord", chat_id="333", parent_chat_id="222")


    def test_match_profile_route_with_parent_chat_id(self):
        routes = [
            ProfileRoute(name="ch", platform="discord", profile="trader",
                         chat_id="222"),
        ]
        m = match_profile_route(routes, "discord", chat_id="333", parent_chat_id="222")
        assert m is not None
        assert m.profile == "trader"


class TestForumPostMatching:
    """Test that forum posts match via parent_chat_id (direct parent)."""


    def test_forum_post_comment_matches_channel_not_thread_id(self):
        """Verify that thread_id matching is distinct from parent_chat_id matching."""
        routes = [
            ProfileRoute(name="forum", platform="discord", profile="forum_profile",
                         chat_id="forum_channel_123"),
            ProfileRoute(name="post", platform="discord", profile="post_profile",
                         thread_id="post_thread_456"),
        ]
        # A comment on the forum post should match the forum channel route, not the thread route
        m = match_profile_route(routes, "discord", chat_id="post_thread_456", 
                                 parent_chat_id="forum_channel_123")
        assert m is not None
        assert m.profile == "forum_profile"
