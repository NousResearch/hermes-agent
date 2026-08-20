"""Tests for gateway/profile_routing.py — profile-based routing."""

import logging

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


class TestParseProfileRoutes:
    def test_empty(self):
        assert parse_profile_routes(None) == []
        assert parse_profile_routes([]) == []


class TestNumericRouteIds:
    """Unquoted numeric IDs in YAML parse as int; SessionSource fields are str.

    parse_profile_routes must coerce them so the route can actually match
    (previously an int discriminator compared int != str and silently never
    matched, falling through to the default profile).
    """

    def test_numeric_ids_are_coerced_to_str(self):
        # Simulates `guild_id: 123456789012345678` in config.yaml
        routes = parse_profile_routes([
            {
                "name": "numeric",
                "platform": "discord",
                "profile": "server-profile",
                "guild_id": 123456789012345678,
                "chat_id": 111222333,
                "thread_id": 444555666,
            },
        ])
        assert len(routes) == 1
        r = routes[0]
        assert r.guild_id == "123456789012345678"
        assert r.chat_id == "111222333"
        assert r.thread_id == "444555666"

    def test_numeric_route_matches_str_source(self):
        routes = parse_profile_routes([
            {
                "name": "numeric",
                "platform": "discord",
                "profile": "server-profile",
                "guild_id": 123456789012345678,
            },
        ])
        assert match_profile_route(
            routes, "discord", guild_id="123456789012345678"
        ) is not None

    def test_non_string_id_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="gateway.profile_routing"):
            parse_profile_routes([
                {
                    "name": "numeric",
                    "platform": "discord",
                    "profile": "server-profile",
                    "guild_id": 123456789012345678,
                },
            ])
        assert any(
            "not a string" in rec.message and "guild_id" in rec.message
            for rec in caplog.records
        )

    def test_string_ids_unchanged(self):
        routes = parse_profile_routes([
            {
                "name": "quoted",
                "platform": "discord",
                "profile": "server-profile",
                "guild_id": "123",
                "chat_id": "456",
            },
        ])
        assert routes[0].guild_id == "123"
        assert routes[0].chat_id == "456"

    def test_float_id_warns_and_never_matches(self, caplog):
        """A YAML float (e.g. `guild_id: 123.0`) must not be silently
        stringified: str(123.0) == "123.0" never equals the str source
        "123", recreating the silent-no-match this coercion exists to fix.
        The value stays a string (so the route is never unconstrained) but
        the warning says outright it can never match."""
        with caplog.at_level(logging.WARNING, logger="gateway.profile_routing"):
            routes = parse_profile_routes([
                {
                    "name": "floaty",
                    "platform": "discord",
                    "profile": "server-profile",
                    "guild_id": 123.0,
                },
            ])
        r = routes[0]
        assert r.guild_id == "123.0"  # kept as str, never None
        # The float stringification does NOT match the int-coerced str form.
        assert match_profile_route(routes, "discord", guild_id="123") is None
        assert any(
            "can never match" in rec.message and "guild_id" in rec.message
            for rec in caplog.records
        )

    def test_bool_id_warns_and_never_matches(self, caplog):
        """bool is an int subclass in Python, so the bool branch must be
        checked BEFORE int — `guild_id: true` would otherwise be coerced
        like a numeric ID. A boolean can never name a route; warn loudly
        and keep it a string (never None/unconstrained)."""
        with caplog.at_level(logging.WARNING, logger="gateway.profile_routing"):
            routes = parse_profile_routes([
                {
                    "name": "booly",
                    "platform": "discord",
                    "profile": "server-profile",
                    "guild_id": True,
                },
            ])
        r = routes[0]
        assert r.guild_id == "True"  # kept as str, never None
        assert match_profile_route(routes, "discord", guild_id="true") is None
        assert any(
            "boolean" in rec.message and "guild_id" in rec.message
            for rec in caplog.records
        )

    def test_non_string_ids_never_become_none(self):
        """None would make the discriminator unconstrained (matches() uses
        truthiness) and turn the route into an accidental wildcard — the
        coercion contract is: every non-None input yields a str."""
        routes = parse_profile_routes([
            {
                "name": "mixed",
                "platform": "discord",
                "profile": "server-profile",
                "guild_id": 123.0,
                "chat_id": True,
                "thread_id": 456,
            },
        ])
        r = routes[0]
        assert isinstance(r.guild_id, str)
        assert isinstance(r.chat_id, str)
        assert isinstance(r.thread_id, str)

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
