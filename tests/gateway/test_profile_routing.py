"""Tests for gateway/profile_routing.py — profile-based routing."""

import json

import pytest
from gateway.profile_routing import (
    ProfileRoute,
    parse_profile_routes,
    match_profile_route,
)
from hermes_constants import get_hermes_dir

PHONE = "15551234567"
LID = "999999999999999"
JID = f"{PHONE}@s.whatsapp.net"
LID_JID = f"{LID}@lid"
GROUP = "120363012345678901@g.us"


def _write_lid_mapping(phone=PHONE, lid=LID):
    """Mirror the JS bridge: phone→lid and lid→phone (reverse)."""
    session_dir = get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / f"lid-mapping-{phone}.json").write_text(
        json.dumps(f"{lid}@lid"), encoding="utf-8"
    )
    (session_dir / f"lid-mapping-{lid}_reverse.json").write_text(
        json.dumps(f"{phone}@s.whatsapp.net"), encoding="utf-8"
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
    """profile_routes chat_id should match WhatsApp JID/LID/number forms.

    Adapter allowlists and session keys already canonicalize via
    gateway.whatsapp_identity; routes used exact string compare and missed.
    """

    def test_number_route_matches_jid_inbound(self):
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=PHONE)
        assert r.matches("whatsapp", chat_id=JID)
        matched = match_profile_route([r], "whatsapp", chat_id=JID)
        assert matched is not None
        assert matched.profile == "owner"

    def test_number_route_matches_lid_inbound_with_mapping(self):
        _write_lid_mapping()
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=PHONE)
        assert r.matches("whatsapp", chat_id=LID_JID)
        matched = match_profile_route([r], "whatsapp", chat_id=LID_JID)
        assert matched is not None
        assert matched.profile == "owner"

    def test_lid_route_matches_jid_inbound_with_mapping(self):
        _write_lid_mapping()
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=LID_JID)
        assert r.matches("whatsapp", chat_id=JID)

    def test_jid_route_matches_lid_inbound_with_mapping(self):
        _write_lid_mapping()
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=JID)
        assert r.matches("whatsapp", chat_id=LID_JID)

    def test_no_mapping_number_still_matches_jid_and_device_suffix(self):
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=PHONE)
        assert r.matches("whatsapp", chat_id=JID)
        assert r.matches("whatsapp", chat_id=f"{PHONE}:47@s.whatsapp.net")
        assert r.matches("whatsapp", chat_id=f"+{PHONE}")

    def test_plus_prefixed_route_matches_jid(self):
        r = ProfileRoute(
            name="owner", platform="whatsapp", profile="owner", chat_id=f"+{PHONE}"
        )
        assert r.matches("whatsapp", chat_id=JID)

    def test_no_mapping_number_does_not_match_unrelated_lid(self):
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=PHONE)
        assert not r.matches("whatsapp", chat_id=LID_JID)
        assert match_profile_route([r], "whatsapp", chat_id=LID_JID) is None

    def test_exact_jid_still_matches(self):
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=JID)
        assert r.matches("whatsapp", chat_id=JID)

    def test_unrelated_phone_does_not_match(self):
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=PHONE)
        assert not r.matches("whatsapp", chat_id="15550001111@s.whatsapp.net")

    def test_telegram_numeric_ids_remain_exact(self):
        r = ProfileRoute(
            name="tg", platform="telegram", profile="owner", chat_id="640466638"
        )
        assert r.matches("telegram", chat_id="640466638")
        assert not r.matches("telegram", chat_id="640466639")
        # Must not apply WhatsApp JID stripping to Telegram routes.
        assert not r.matches("telegram", chat_id="640466638@s.whatsapp.net")

    def test_group_jid_does_not_match_phone_route(self):
        r = ProfileRoute(name="owner", platform="whatsapp", profile="owner", chat_id=PHONE)
        assert not r.matches("whatsapp", chat_id=GROUP)

    def test_group_route_stays_exact_and_is_not_a_sender(self):
        r = ProfileRoute(name="group", platform="whatsapp", profile="group", chat_id=GROUP)
        assert r.matches("whatsapp", chat_id=GROUP)
        assert not r.matches("whatsapp", chat_id=JID)
        # Stripping @g.us must not turn a group into a phone-identity match.
        stripped = GROUP.split("@", 1)[0]
        numeric = ProfileRoute(
            name="oops", platform="whatsapp", profile="owner", chat_id=stripped
        )
        assert not numeric.matches("whatsapp", chat_id=GROUP)

    def test_whatsapp_cloud_number_matches_jid(self):
        r = ProfileRoute(
            name="owner", platform="whatsapp_cloud", profile="owner", chat_id=PHONE
        )
        assert r.matches("whatsapp_cloud", chat_id=JID)
