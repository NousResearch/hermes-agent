"""Identity is canonicalized FIRST on every ingress path (#88715 phase 3).

Two Telegram bots on one multiplexed gateway see the same ``chat.id == user.id`` for a DM with the
same human; every adapter-side lane (batch dict, ``_active_sessions``, the busy guard, control
commands, clarify replies) must be keyed by the receiving bot's identity before anything else
derives a key. Real ``GatewayRunner`` resolvers, real ``BasePlatformAdapter`` ingress, temp home.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.profile_routing import parse_profile_routes
from gateway.session_identity import identity_of

UID = "72719239"  # Telegram DM: chat.id == user.id, identical for every bot


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _adapter(runner, owner=None):
    adapter = _Stub.__new__(_Stub)
    BasePlatformAdapter.__init__(adapter, PlatformConfig(enabled=True, extra={}), Platform.TELEGRAM)
    adapter.gateway_runner = runner
    adapter._session_store = SimpleNamespace(_resolve_profile_for_key=lambda s: "default")
    if owner:
        adapter.set_owner_profile(owner)
    return adapter


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Default home owns bot A; ``team_b`` owns bot B; satellite ``ops`` is routed through bot A
    for chat 5150; a route to unserved ``ghost`` for chat 4040."""
    from gateway.run import GatewayRunner

    home = tmp_path / "hh"
    for name in ("ops", "team_b"):
        (home / "profiles" / name).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = parse_profile_routes([
        {"name": "ops-dm", "platform": "telegram", "profile": "ops", "chat_id": "5150"},
        {"name": "ghost", "platform": "telegram", "profile": "ghost", "chat_id": "4040"},
    ])
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    bot_a, bot_b = _adapter(runner), _adapter(runner, owner="team_b")
    runner.adapters = {Platform.TELEGRAM: bot_a}
    runner._profile_adapters = {"team_b": {Platform.TELEGRAM: bot_b}, "ops": {}}
    served = [("default", home), ("ops", home / "profiles" / "ops"), ("team_b", home / "profiles" / "team_b")]
    with patch("gateway.profile_serving.profiles_to_serve", return_value=served), \
            patch("profiles.paths.get_profile_dir", side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("profiles.registry.profile_exists", return_value=True):
        yield SimpleNamespace(runner=runner, home=home, bot_a=bot_a, bot_b=bot_b)


def _event(adapter, chat_id, text, **kw):
    source = adapter.build_source(chat_id=chat_id, chat_type="dm", user_id=chat_id)
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id="m", **kw)


async def _hold(adapter, key):
    guard = asyncio.Event()
    adapter._active_sessions[key] = guard
    adapter._session_tasks[key] = asyncio.create_task(asyncio.sleep(30))
    return guard


@pytest.mark.asyncio
async def test_two_bots_same_chat_have_distinct_lanes_and_stop_cannot_cross(rig):
    """Acceptance row A: the same human DMs both bots. Batching, the active-session guard and a
    ``/stop`` all resolve on the receiving bot's lane; ``/stop`` on A never touches B's run, and the
    clarify reply on A resolves A's pending prompt, not B's."""
    seen = []

    async def handler(event):
        seen.append((identity_of(event.source).transport_profile, event.text))
        return None

    for bot in (rig.bot_a, rig.bot_b):
        bot.set_message_handler(handler)

    ev_a, ev_b = _event(rig.bot_a, UID, "hello A"), _event(rig.bot_b, UID, "hello B")
    rig.bot_a._enqueue_text_event(ev_a)
    rig.bot_b._enqueue_text_event(ev_b)
    key_a, key_b = rig.bot_a._event_session_key(ev_a), rig.bot_b._event_session_key(ev_b)
    assert key_a == f"agent:main:telegram:dm:{UID}" and key_b == f"agent:team_b:telegram:dm:{UID}"
    assert list(rig.bot_a._pending_text_batches) == [key_a] and list(rig.bot_b._pending_text_batches) == [key_b]
    # The identity was pinned BEFORE the key was derived, and the two lanes never share a source.
    assert identity_of(ev_a.source).transport_profile == "default"
    assert identity_of(ev_b.source).transport_profile == "team_b"
    for bot in (rig.bot_a, rig.bot_b):
        for task in list(bot._pending_text_batch_tasks.values()):
            task.cancel()
        bot._pending_text_batches.clear()

    # Both bots busy on "their" chat; /stop on A interrupts A's guard only and B's task survives.
    guard_a, guard_b = await _hold(rig.bot_a, key_a), await _hold(rig.bot_b, key_b)
    task_b = rig.bot_b._session_tasks[key_b]
    await rig.bot_a.handle_message(_event(rig.bot_a, UID, "/stop", allow_gateway_control=True))
    await asyncio.sleep(0)
    assert seen == [("default", "/stop")]
    assert key_a not in rig.bot_a._session_tasks
    assert rig.bot_b._session_tasks[key_b] is task_b and not task_b.cancelled()
    assert rig.bot_b._active_sessions[key_b] is guard_b and not guard_b.is_set()
    assert not guard_a.is_set()  # the command path replaced A's guard; the run was cancelled, not flagged

    # Clarify: pending prompts on both lanes; the plain-text answer on A resolves A only.
    from tools import clarify_gateway
    clarify_gateway.register("c-a", key_a, "Which?", ["x", "y"])
    clarify_gateway.register("c-b", key_b, "Which?", ["x", "y"])
    clarify_gateway.mark_awaiting_text("c-a")
    clarify_gateway.mark_awaiting_text("c-b")
    try:
        await _hold(rig.bot_a, key_a)
        await rig.bot_a.handle_message(_event(rig.bot_a, UID, "x", allow_gateway_control=True))
        assert seen[-1] == ("default", "x")  # routed through A's inline clarify intercept
        assert clarify_gateway.get_pending_for_session(key_b, include_choice_prompts=True) is not None
        assert key_b not in rig.bot_a._pending_messages and key_a not in rig.bot_b._pending_messages
    finally:
        clarify_gateway.clear_session(key_a)
        clarify_gateway.clear_session(key_b)
        for bot in (rig.bot_a, rig.bot_b):
            for task in bot._session_tasks.values():
                task.cancel()


@pytest.mark.asyncio
async def test_shared_bot_routed_chat_runs_as_satellite_and_unserved_route_is_dropped_everywhere(rig):
    """Acceptance row B: a chat routed through the shared bot to satellite ``ops`` keys into
    ``agent:ops`` on every ingress path (fresh, batch, busy) while the transport stays the receiving
    bot. A route to an unserved profile is dropped at every path — never ``agent:main``."""
    calls = []

    async def handler(event):
        calls.append(event.text)
        return None

    async def busy(event, session_key):
        calls.append(("busy", session_key, identity_of(event.source).runtime_profile))
        return True

    rig.bot_a.set_message_handler(handler)
    rig.bot_a.set_busy_session_handler(busy)

    routed = _event(rig.bot_a, "5150", "hi ops")
    key = rig.bot_a._event_session_key(routed)
    identity = identity_of(routed.source)
    assert key == "agent:ops:telegram:dm:5150"
    assert (identity.transport_profile, identity.runtime_profile) == ("default", "ops")
    assert identity.adapter() is rig.bot_a and identity.runtime_home == rig.home / "profiles" / "ops"

    await _hold(rig.bot_a, key)
    await rig.bot_a.handle_message(_event(rig.bot_a, "5150", "follow-up"))
    assert calls[-1] == ("busy", key, "ops")

    # Unserved route: fresh, batched, busy and control paths all drop; no lane is created or touched.
    main_key = "agent:main:telegram:dm:4040"
    guard = await _hold(rig.bot_a, main_key)
    for text, kw in (("hello", {}), ("/stop", {"allow_gateway_control": True})):
        await rig.bot_a.handle_message(_event(rig.bot_a, "4040", text, **kw))
    rig.bot_a._enqueue_text_event(_event(rig.bot_a, "4040", "chunk"))
    assert calls == [("busy", key, "ops")]
    assert not rig.bot_a._pending_text_batches and main_key not in rig.bot_a._pending_messages
    assert rig.bot_a._active_sessions[main_key] is guard and main_key in rig.bot_a._session_tasks
    for task in rig.bot_a._session_tasks.values():
        task.cancel()
