"""Tests for gateway/profile_routing.py — profile-based routing."""

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

    def test_specificity_channel(self):
        r = ProfileRoute(name="c", platform="discord", profile="p",
                         guild_id="g", chat_id="c")
        assert r.specificity == 6  # 2 + 4

    def test_specificity_guild(self):
        r = ProfileRoute(name="g", platform="discord", profile="p",
                         guild_id="g")
        assert r.specificity == 2

    def test_specificity_minimal(self):
        r = ProfileRoute(name="m", platform="telegram", profile="p")
        assert r.specificity == 0

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

    def test_channel_match(self):
        r = ProfileRoute(name="c", platform="discord", profile="helper",
                         chat_id="222")
        assert r.matches("discord", chat_id="222")
        assert not r.matches("discord", chat_id="333")
        assert not r.matches("telegram", chat_id="222")

    def test_guild_match(self):
        r = ProfileRoute(name="g", platform="discord", profile="server",
                         guild_id="111")
        assert r.matches("discord", guild_id="111")
        assert not r.matches("discord", guild_id="222")

    def test_disabled_route_no_match(self):
        r = ProfileRoute(name="d", platform="discord", profile="off",
                         guild_id="111", enabled=False)
        assert not r.matches("discord", guild_id="111")

    def test_guild_route_matches_any_channel_in_guild(self):
        r = ProfileRoute(name="g", platform="discord", profile="server",
                         guild_id="111")
        assert r.matches("discord", guild_id="111", chat_id="222")
        assert r.matches("discord", guild_id="111", chat_id="222", thread_id="333")

    def test_extra_fields_ignored(self):
        r = ProfileRoute(name="g", platform="discord", profile="server",
                         guild_id="111")
        assert r.matches("discord", guild_id="111", chat_id="any")


class TestParseProfileRoutes:
    def test_empty(self):
        assert parse_profile_routes(None) == []
        assert parse_profile_routes([]) == []

    def test_valid_routes_sorted_by_specificity(self):
        raw = [
            {"name": "guild", "platform": "discord", "profile": "p", "guild_id": "1"},
            {"name": "thread", "platform": "discord", "profile": "p",
             "guild_id": "1", "chat_id": "2", "thread_id": "3"},
            {"name": "channel", "platform": "discord", "profile": "p", "chat_id": "2"},
        ]
        routes = parse_profile_routes(raw)
        names = [r.name for r in routes]
        assert names == ["thread", "channel", "guild"]

    def test_skips_invalid(self):
        raw = [
            {"platform": "discord"},
            {"profile": "p"},
            "not a dict",
            {"name": "ok", "platform": "telegram", "profile": "p"},
        ]
        routes = parse_profile_routes(raw)
        assert len(routes) == 1
        assert routes[0].name == "ok"

    def test_enabled_flag(self):
        raw = [
            {"name": "off", "platform": "discord", "profile": "p",
             "guild_id": "1", "enabled": False},
            {"name": "on", "platform": "discord", "profile": "p", "guild_id": "1"},
        ]
        routes = parse_profile_routes(raw)
        assert not routes[0].enabled
        assert routes[1].enabled


class TestMatchProfileRoute:
    def test_no_routes(self):
        assert match_profile_route([], "discord") is None

    def test_returns_first_match(self):
        routes = [
            ProfileRoute(name="thread", platform="discord", profile="trader",
                         guild_id="1", chat_id="2", thread_id="3"),
            ProfileRoute(name="channel", platform="discord", profile="helper",
                         chat_id="2"),
        ]
        m = match_profile_route(routes, "discord", guild_id="1", chat_id="2", thread_id="3")
        assert m is not None
        assert m.profile == "trader"

    def test_falls_through_to_channel(self):
        routes = [
            ProfileRoute(name="thread", platform="discord", profile="trader",
                         guild_id="1", chat_id="2", thread_id="3"),
            ProfileRoute(name="channel", platform="discord", profile="helper",
                         chat_id="2"),
        ]
        m = match_profile_route(routes, "discord", guild_id="1", chat_id="2")
        assert m is not None
        assert m.profile == "helper"

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

    def test_custom_profile_key(self):
        from gateway.session import build_session_key, SessionSource, Platform
        src = SessionSource(platform=Platform.DISCORD, chat_id="123",
                            chat_type="channel", user_id="456")
        key = build_session_key(src, profile="trader")
        assert key.startswith("agent:trader:")
        assert key == "agent:trader:discord:channel:123:456"

    def test_isolated_sessions(self):
        from gateway.session import build_session_key, SessionSource, Platform
        src = SessionSource(platform=Platform.DISCORD, chat_id="123",
                            chat_type="channel", user_id="456")
        key_default = build_session_key(src)
        key_trader = build_session_key(src, profile="trader")
        assert key_default != key_trader

    def test_dm_profile_scoped(self):
        from gateway.session import build_session_key, SessionSource, Platform
        src = SessionSource(platform=Platform.DISCORD, chat_id="999",
                            chat_type="dm", user_id="111")
        key = build_session_key(src, profile="bot2")
        assert key == "agent:bot2:discord:dm:999"



class TestParentChatIdMatching:
    """Thread messages carry thread_id as chat_id; parent_chat_id is the channel."""

    def test_channel_route_matches_via_parent_chat_id(self):
        r = ProfileRoute(name="ch", platform="discord", profile="trader",
                         chat_id="222")
        assert r.matches("discord", chat_id="333", parent_chat_id="222")

    def test_channel_route_no_match_wrong_parent(self):
        r = ProfileRoute(name="ch", platform="discord", profile="trader",
                         chat_id="222")
        assert not r.matches("discord", chat_id="333", parent_chat_id="444")

    def test_match_profile_route_with_parent_chat_id(self):
        routes = [
            ProfileRoute(name="ch", platform="discord", profile="trader",
                         chat_id="222"),
        ]
        m = match_profile_route(routes, "discord", chat_id="333", parent_chat_id="222")
        assert m is not None
        assert m.profile == "trader"

    def test_thread_id_does_not_match_parent_chat_id(self):
        """thread_id only matches the actual thread_id, never parent_chat_id.
        Discord snowflakes are globally unique, so thread_id != channel_id."""
        r = ProfileRoute(name="th", platform="discord", profile="helper",
                         thread_id="555")
        assert r.matches("discord", thread_id="555")
        assert not r.matches("discord", parent_chat_id="555")

    def test_no_parent_chat_id_still_works(self):
        r = ProfileRoute(name="ch", platform="discord", profile="trader",
                         chat_id="222")
        assert r.matches("discord", chat_id="222")

    def test_guild_route_matches_with_parent_chat_id(self):
        """Guild routes should match regardless of chat_id or parent_chat_id."""
        r = ProfileRoute(name="g", platform="discord", profile="server",
                         guild_id="111")
        assert r.matches("discord", guild_id="111", chat_id="333", parent_chat_id="444")


class TestForumPostMatching:
    """Test that forum posts match via parent_chat_id (direct parent)."""

    def test_forum_channel_route_matches_forum_post(self):
        """A route on a forum channel should match comments on posts in that forum.
        
        In Discord, forum posts (threads) have parent_chat_id = forum channel ID.
        No cache is needed — the parent relationship is direct.
        """
        r = ProfileRoute(name="forum", platform="discord", profile="forum_profile",
                         chat_id="forum_channel_123")
        # A comment on a forum post: chat_id=post_thread_id, parent_chat_id=forum_channel_id
        assert r.matches("discord", chat_id="post_thread_456", parent_chat_id="forum_channel_123")

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
