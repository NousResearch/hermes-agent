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

    def test_specificity_bot(self):
        r = ProfileRoute(name="b", platform="telegram", profile="p",
                         bot="my_bot", chat_id="c")
        assert r.specificity == 20  # 4 + 16

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

    def test_bot_exact_and_case_insensitive_matching(self):
        r = ProfileRoute(name="b", platform="telegram", profile="team_b", bot="my_bot")
        # Exact match
        assert r.matches("telegram", bot="my_bot")
        # Case insensitive and ignores leading @
        assert r.matches("telegram", bot="@MY_BOT")
        assert r.matches("telegram", bot="My_Bot")
        # Mismatch
        assert not r.matches("telegram", bot="other_bot")
        assert not r.matches("telegram", bot=None)

    def test_adapter_profile_prevents_generic_route_hijack(self):
        # Generic route with chat_id but no bot discriminator
        r = ProfileRoute(name="admin_dm", platform="telegram", profile="ops", chat_id="72719239")
        # Normal primary adapter (adapter_profile=None) matches
        assert r.matches("telegram", chat_id="72719239", adapter_profile=None)
        # Dedicated secondary adapter (adapter_profile="team_b") does NOT match (not hijacked)
        assert not r.matches("telegram", chat_id="72719239", adapter_profile="team_b")

    def test_adapter_profile_allows_explicit_bot_route(self):
        # Explicit route with bot discriminator matches even if adapter_profile is set
        r = ProfileRoute(
            name="explicit_override", platform="telegram", profile="ops",
            chat_id="72719239", bot="team_b_bot"
        )
        assert r.matches("telegram", chat_id="72719239", bot="team_b_bot", adapter_profile="team_b")
        assert not r.matches("telegram", chat_id="72719239", bot="other_bot", adapter_profile="team_b")


class TestParseProfileRoutes:
    def test_empty(self):
        assert parse_profile_routes(None) == []
        assert parse_profile_routes([]) == []

    def test_coerces_yaml_native_int_ids_to_str(self):
        # PyYAML loads unquoted snowflakes / negative Telegram ids as int;
        # inbound SessionSource ids are str, so un-coerced routes never match.
        routes = parse_profile_routes([
            {"name": "server", "platform": "discord", "profile": "p",
             "guild_id": 111, "chat_id": 222, "thread_id": 333},
            {"name": "tg", "platform": "telegram", "profile": "p",
             "chat_id": -1001234567890},
            {"name": "platform-only", "platform": "discord", "profile": "p"},
        ])
        by_name = {r.name: r for r in routes}
        assert (by_name["server"].guild_id, by_name["server"].chat_id,
                by_name["server"].thread_id) == ("111", "222", "333")
        assert match_profile_route(
            routes, "discord", guild_id="111", chat_id="222", thread_id="333",
        ).name == "server"
        assert match_profile_route(
            routes, "telegram", chat_id="-1001234567890",
        ).name == "tg"
        assert (by_name["platform-only"].guild_id, by_name["platform-only"].chat_id,
                by_name["platform-only"].thread_id) == (None, None, None)

    def test_non_int_numeric_ids_warn_instead_of_silently_coercing(self, caplog):
        # #86470 nuance: float/bool stringify to values that can never match
        # an inbound id, so surface the misconfiguration at load time.
        with caplog.at_level("WARNING", logger="gateway.profile_routing"):
            routes = parse_profile_routes([
                {"name": "f", "platform": "discord", "profile": "p", "chat_id": 123.0},
                {"name": "b", "platform": "discord", "profile": "p", "guild_id": True},
            ])
        assert {r.name for r in routes} == {"f", "b"}
        assert match_profile_route(routes, "discord", chat_id="123") is None
        assert sum("can never match" in rec.message for rec in caplog.records) == 2

    def test_parse_bot_variants(self):
        routes = parse_profile_routes([
            {"name": "b1", "platform": "telegram", "profile": "p1", "bot": "@my_bot"},
            {"name": "b2", "platform": "telegram", "profile": "p2", "bot_id": 123456},
            {"name": "b3", "platform": "telegram", "profile": "p3", "bot_username": "other_bot"},
        ])
        by_name = {r.name: r for r in routes}
        assert by_name["b1"].bot == "my_bot"
        assert by_name["b2"].bot == "123456"
        assert by_name["b3"].bot == "other_bot"


class TestMatchProfileRoute:

    def test_no_match_returns_none(self):
        routes = [
            ProfileRoute(name="r", platform="telegram", profile="p"),
        ]
        assert match_profile_route(routes, "discord") is None

    def test_match_profile_route_bot_and_adapter_profile(self):
        routes = [
            ProfileRoute(name="generic_dm", platform="telegram", profile="ops", chat_id="12345"),
            ProfileRoute(name="bot_specific", platform="telegram", profile="custom", bot="special_bot"),
        ]
        # Primary adapter matches generic
        assert match_profile_route(routes, "telegram", chat_id="12345", adapter_profile=None).name == "generic_dm"
        # Secondary adapter skips generic
        assert match_profile_route(routes, "telegram", chat_id="12345", adapter_profile="team_b") is None
        # Secondary adapter with matching bot matches
        assert match_profile_route(routes, "telegram", chat_id="12345", bot="special_bot", adapter_profile="team_b").name == "bot_specific"


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


class TestWhatsAppChatIdIdentityMatching:
    """WhatsApp ``chat_id`` routes match across number / JID / LID forms (the
    same alias canonicalization allowlists and session keys already use);
    every other platform, and WhatsApp groups, stay exact-compare."""

    PHONE = "15551234567"
    LID = "999999999999999"

    def _write_lid_mapping(self, tmp_path, monkeypatch):
        mapping_dir = tmp_path / "platforms" / "whatsapp" / "session"
        mapping_dir.mkdir(parents=True)
        (mapping_dir / f"lid-mapping-{self.PHONE}.json").write_text(json.dumps(f"{self.LID}@lid"))
        (mapping_dir / f"lid-mapping-{self.LID}_reverse.json").write_text(
            json.dumps(f"{self.PHONE}@s.whatsapp.net")
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def test_number_route_matches_jid_and_mapped_lid_forms(self, tmp_path, monkeypatch):
        self._write_lid_mapping(tmp_path, monkeypatch)
        for platform in ("whatsapp", "whatsapp_cloud"):
            r = ProfileRoute(name="owner", platform=platform, profile="owner", chat_id=self.PHONE)
            assert r.matches(platform, chat_id=f"{self.PHONE}@s.whatsapp.net")
            assert r.matches(platform, chat_id=f"{self.PHONE}:47@s.whatsapp.net")
            assert r.matches(platform, chat_id=f"{self.LID}@lid")
            # Alias fallback also applies to the thread-parent slot.
            assert r.matches(platform, chat_id="thread-1", parent_chat_id=f"{self.LID}@lid")
            assert not r.matches(platform, chat_id="15550001111@s.whatsapp.net")

    def test_groups_and_other_platforms_stay_exact(self, tmp_path, monkeypatch):
        self._write_lid_mapping(tmp_path, monkeypatch)
        group = "120363012345678901@g.us"
        owner = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=self.PHONE)
        assert not owner.matches("whatsapp", chat_id=group)
        grp = ProfileRoute(name="grp", platform="whatsapp", profile="grp", chat_id=group)
        assert grp.matches("whatsapp", chat_id=group)
        assert not grp.matches("whatsapp", chat_id=f"{self.PHONE}@s.whatsapp.net")
        # Stripping @g.us must never turn a group into a phone-identity match.
        assert not ProfileRoute(
            name="oops", platform="whatsapp", profile="owner", chat_id=group.split("@", 1)[0]
        ).matches("whatsapp", chat_id=group)
        tg = ProfileRoute(name="tg", platform="telegram", profile="owner", chat_id="640466638")
        assert tg.matches("telegram", chat_id="640466638")
        assert not tg.matches("telegram", chat_id="640466638@s.whatsapp.net")
