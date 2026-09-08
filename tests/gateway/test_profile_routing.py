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

    def test_user_id_matches_exactly_and_is_conjunctive_with_chat_id(self):
        sender = ProfileRoute(
            name="sender", platform="teams", profile="owner", user_id="aad-456",
        )
        assert sender.matches("teams", user_id="aad-456")
        assert not sender.matches("teams", user_id="AAD-456")
        assert not sender.matches("teams")

        sender_chat = ProfileRoute(
            name="sender-chat", platform="teams", profile="owner",
            user_id="aad-456", chat_id="conversation-1",
        )
        assert sender_chat.matches(
            "teams", user_id="aad-456", chat_id="conversation-1",
        )
        assert not sender_chat.matches(
            "teams", user_id="aad-456", chat_id="conversation-2",
        )

    def test_sender_routes_split_one_chat_and_outrank_its_location_route(self):
        routes = parse_profile_routes([
            {"name": "shared", "platform": "teams", "profile": "shared",
             "chat_id": "shared-chat"},
            {"name": "alice", "platform": "teams", "profile": "alice",
             "chat_id": "shared-chat", "user_id": "user-a"},
            {"name": "bob", "platform": "teams", "profile": "bob", "user_id": "user-b"},
        ])
        for user, profile in (("user-a", "alice"), ("user-b", "bob"), ("user-c", "shared")):
            assert match_profile_route(
                routes, "teams", chat_id="shared-chat", user_id=user,
            ).profile == profile


class TestParseProfileRoutes:
    def test_empty(self):
        assert parse_profile_routes(None) == []
        assert parse_profile_routes([]) == []

    def test_coerces_yaml_native_int_ids_to_str(self):
        # PyYAML loads unquoted snowflakes / negative Telegram ids as int;
        # inbound SessionSource ids are str, so un-coerced routes never match.
        routes = parse_profile_routes([
            {"name": "server", "platform": "discord", "profile": "p",
             "guild_id": 111, "chat_id": 222, "thread_id": 333, "user_id": 444},
            {"name": "tg", "platform": "telegram", "profile": "p",
             "chat_id": -1001234567890},
            {"name": "platform-only", "platform": "discord", "profile": "p"},
        ])
        by_name = {r.name: r for r in routes}
        assert (by_name["server"].guild_id, by_name["server"].chat_id,
                by_name["server"].thread_id, by_name["server"].user_id) == (
                    "111", "222", "333", "444",
                )
        assert match_profile_route(
            routes, "discord", guild_id="111", chat_id="222", thread_id="333",
            user_id="444",
        ).name == "server"
        assert match_profile_route(
            routes, "telegram", chat_id="-1001234567890",
        ).name == "tg"
        assert (by_name["platform-only"].guild_id, by_name["platform-only"].chat_id,
                by_name["platform-only"].thread_id) == (None, None, None)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_user_id_is_inert_never_a_broader_route(self, blank):
        # Unlike the location fields, a blank sender does not fall back to "unconstrained".
        routes = parse_profile_routes([
            {"name": "blank", "platform": "teams", "profile": "owner", "user_id": blank},
            {"name": "legacy", "platform": "teams", "profile": "shared"},
        ])
        assert match_profile_route(routes, "teams", user_id=blank).name == "legacy"
        assert match_profile_route(routes, "teams", user_id="anyone").name == "legacy"

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


class TestGatewayConfigRoundtrip:
    def test_routes_survive_to_dict_from_dict_with_user_id_and_enabled(self):
        # GatewayConfig serializes routes with asdict(); a field added to ProfileRoute but not
        # re-read by parse_profile_routes would silently widen or disable routes on reload.
        from gateway.config import GatewayConfig

        config = GatewayConfig(profile_routes=parse_profile_routes([
            {"name": "sender", "platform": "teams", "profile": "owner", "user_id": "aad-456"},
            {"name": "off", "platform": "teams", "profile": "owner",
             "chat_id": "conversation-1", "enabled": False},
        ]))
        restored = GatewayConfig.from_dict(config.to_dict()).profile_routes

        assert [(r.name, r.user_id, r.enabled, r.specificity) for r in restored] == [
            ("sender", "aad-456", True, 16), ("off", None, False, 4),
        ]
