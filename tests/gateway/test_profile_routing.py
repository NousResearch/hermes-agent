"""Platform-agnostic profile routing table — resolve_profile_route."""

from gateway.routing import resolve_profile_route
from gateway.session import SessionSource
from gateway.platforms.base import Platform


def _src(**kw):
    base = dict(platform=Platform.DISCORD, chat_id="c1", chat_type="channel")
    base.update(kw)
    return SessionSource(**base)


def test_empty_table_returns_none():
    assert resolve_profile_route(None, _src()) is None
    assert resolve_profile_route({"routes": []}, _src()) is None


def test_exact_thread_wins_over_channel():
    table = {"routes": [
        {"platform": "discord", "channel_id": "c1", "profile": "chan"},
        {"platform": "discord", "channel_id": "c1", "thread_id": "t9", "profile": "thread"},
    ]}
    assert resolve_profile_route(table, _src(thread_id="t9")) == "thread"


def test_thread_inherits_parent_channel():
    table = {"routes": [{"platform": "discord", "channel_id": "c1", "profile": "chan"}]}
    s = _src(chat_id="t5", thread_id="t5", parent_chat_id="c1")
    assert resolve_profile_route(table, s) == "chan"


def test_guild_then_platform_fallback():
    table = {"routes": [
        {"platform": "discord", "guild_id": "g1", "profile": "guild"},
        {"platform": "slack", "profile": "work"},
    ]}
    assert resolve_profile_route(table, _src(guild_id="g1")) == "guild"
    assert resolve_profile_route(table, _src(platform=Platform.SLACK)) == "work"


def test_channel_beats_guild():
    table = {"routes": [
        {"platform": "discord", "guild_id": "g1", "profile": "guild"},
        {"platform": "discord", "channel_id": "c1", "profile": "chan"},
    ]}
    assert resolve_profile_route(table, _src(guild_id="g1")) == "chan"


def test_user_id_route():
    table = {"routes": [{"user_id": "42100", "profile": "admin"}]}
    assert resolve_profile_route(table, _src(user_id="42100")) == "admin"


def test_user_id_alt_matches():
    table = {"routes": [{"user_id": "uuid-9", "profile": "admin"}]}
    assert resolve_profile_route(table, _src(user_id="raw", user_id_alt="uuid-9")) == "admin"


def test_no_match_returns_none():
    table = {"routes": [{"platform": "telegram", "profile": "x"}]}
    assert resolve_profile_route(table, _src()) is None


def test_default_used_when_no_route_matches():
    table = {"default": "fallback", "routes": [{"platform": "telegram", "profile": "x"}]}
    assert resolve_profile_route(table, _src()) == "fallback"


def test_non_dict_routes_ignored():
    table = {"routes": ["bogus", {"platform": "discord", "channel_id": "c1", "profile": "chan"}]}
    assert resolve_profile_route(table, _src()) == "chan"
