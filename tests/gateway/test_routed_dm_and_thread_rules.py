"""Regression guards: per-peer DM isolation + thread→parent-channel routing,
both under Tier-2 routing so they can't silently regress."""

from gateway.routing import resolve_profile_route
from gateway.session import build_session_key, SessionSource
from gateway.platforms.base import Platform


def _dm(chat_id):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm", user_id=chat_id)


def test_distinct_dm_peers_get_distinct_keys_under_same_profile():
    a = build_session_key(_dm("111"), profile="support")
    b = build_session_key(_dm("222"), profile="support")
    assert a != b
    assert a.startswith("agent:support:") and b.startswith("agent:support:")


def test_thread_without_own_route_inherits_parent_channel_profile():
    table = {"routes": [{"platform": "discord", "channel_id": "chan1", "profile": "support"}]}
    # A message in thread t77 whose parent channel is chan1, with no thread route.
    thread_src = SessionSource(
        platform=Platform.DISCORD, chat_id="t77", thread_id="t77",
        parent_chat_id="chan1", chat_type="channel",
    )
    profile = resolve_profile_route(table, thread_src)
    assert profile == "support"
    # And the folded session key carries that inherited profile.
    assert build_session_key(thread_src, profile=profile).startswith("agent:support:")


def test_explicit_thread_route_still_beats_parent():
    table = {"routes": [
        {"platform": "discord", "channel_id": "chan1", "profile": "support"},
        {"platform": "discord", "channel_id": "chan1", "thread_id": "t77", "profile": "vip"},
    ]}
    thread_src = SessionSource(
        platform=Platform.DISCORD, chat_id="t77", thread_id="t77",
        parent_chat_id="chan1", chat_type="channel",
    )
    assert resolve_profile_route(table, thread_src) == "vip"
