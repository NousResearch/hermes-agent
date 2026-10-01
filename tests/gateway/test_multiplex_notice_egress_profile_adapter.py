"""Out-of-band gateway egress for a SECONDARY profile session leaves through that profile's bot.

Shutdown/restart/update notices and ``/loop`` wakeups resolved their adapter by bare platform from
``runner.adapters`` — the default profile's map — so a secondary session's notice landed in the
user's chat with the wrong bot (and for ``/loop`` the wakeup ran the default profile's turn).  They
must use the session's own profile adapter and fail closed when that profile has none.
"""
import json
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.slash_commands import _restart_notify_payload


class _Adapter:
    def __init__(self):
        self.sent, self.handled = [], []

    async def send(self, chat_id, content=None, metadata=None, **kw):
        self.sent.append(chat_id)
        return SimpleNamespace(success=True, message_id="m", error=None)

    async def handle_message(self, event):
        self.handled.append(event.text)


def _runner():
    r = object.__new__(GatewayRunner)
    r.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    r.adapters = {Platform.TELEGRAM: _Adapter()}
    r._profile_adapters = {"sec": {Platform.TELEGRAM: _Adapter()}, "nobot": {}}
    r._primary_profile_name = "default"
    r.session_store = None
    r._session_sources = None
    r._running_agents = {}
    r._restart_requested = False
    r._restart_command_source = None
    return r


@pytest.mark.asyncio
async def test_shutdown_notice_for_secondary_session_uses_its_own_bot():
    r = _runner()
    r._running_agents["agent:sec:telegram:dm:42"] = object()
    r._running_agents["agent:nobot:telegram:dm:43"] = object()
    r._snapshot_running_agents = lambda: list(r._running_agents)

    await r._notify_active_sessions_of_shutdown()

    assert r._profile_adapters["sec"][Platform.TELEGRAM].sent == ["42"]
    # The default bot never speaks for a secondary session — not even one whose bot is down.
    assert r.adapters[Platform.TELEGRAM].sent == []


@pytest.mark.asyncio
async def test_restart_marker_from_secondary_session_notifies_via_its_own_bot(tmp_path, monkeypatch):
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    r = _runner()
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42", profile="sec")
    event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=src, message_id="7")
    (tmp_path / ".restart_notify.json").write_text(json.dumps(_restart_notify_payload(event)))

    assert await r._send_restart_notification() == ("telegram", "42", None)
    assert r._profile_adapters["sec"][Platform.TELEGRAM].sent == ["42"]
    assert r.adapters[Platform.TELEGRAM].sent == []


@pytest.mark.asyncio
async def test_loop_wakeup_from_secondary_route_fires_through_its_own_bot(monkeypatch):
    from agent import estop
    from hermes_cli import loops

    fired = []

    class _Mgr:
        state = SimpleNamespace(ticks_fired=1)

        def __init__(self, session_id=None):
            pass

        def is_due(self, now):
            return True

        def fire_tick(self):
            fired.append(1)
            return "tick"

        def complete_tick(self, *_):
            return {}

    monkeypatch.setattr(loops, "LoopManager", _Mgr)
    monkeypatch.setattr(loops, "goal_blocks_loop_tick", lambda sid: False)
    r = _runner()
    r._running = True
    route = {"platform": "telegram", "chat_id": "42", "chat_type": "dm", "user_id": "42", "profile": "sec"}
    state = SimpleNamespace(awaiting_response=False, next_due_at=0, route=route)

    # `hermes pause`: loop wakeups are internal events that bypass the inbound estop gate, so the
    # watcher itself must hold the tick — unclaimed (fire_tick never runs), so it stays due for resume.
    estop.engage(reason="maintenance")
    try:
        await r._loop_wakeup_fire_one("sid", state, 1e12, set())
        assert (fired, r._profile_adapters["sec"][Platform.TELEGRAM].handled) == ([], [])
    finally:
        estop.disengage()

    await r._loop_wakeup_fire_one("sid", state, 1e12, set())
    await r._loop_wakeup_fire_one("sid2", SimpleNamespace(**{**vars(state), "route": {**route, "profile": "nobot"}}), 1e12, set())

    assert r._profile_adapters["sec"][Platform.TELEGRAM].handled == ["tick"]
    assert r.adapters[Platform.TELEGRAM].handled == []
    assert fired == [1]  # resume: the held tick fires once; the bot-less route never claims one


def _home_config(platform: Platform, chat_id: str, *, notify: bool = True) -> GatewayConfig:
    return GatewayConfig(platforms={platform: PlatformConfig(
        enabled=True, token="t", gateway_restart_notification=notify,
        home_channel=HomeChannel(platform=platform, chat_id=chat_id, name=chat_id))})


@pytest.mark.asyncio
async def test_shutdown_notice_reaches_every_served_profiles_home_channel(monkeypatch):
    """The home-channel shutdown broadcast covers EVERY served profile, through its own bot.

    ``self.adapters``/``self.config`` are the launch profile's alone, so a served profile's home
    channel never heard the gateway was going down (#118233). Two bots with the same positive
    Telegram id are two private conversations: both are owed a notice. A served profile's own
    ``gateway_restart_notification=false`` opt-out is honoured from ITS config.
    """
    monkeypatch.setattr("gateway.drain_control.drain_notification_suppressed", lambda: False)
    r = _runner()
    r.config = _home_config(Platform.TELEGRAM, "8776018003")
    r._profile_configs = {
        "sec": _home_config(Platform.TELEGRAM, "8776018003"),
        "quiet": _home_config(Platform.DISCORD, "-quiet", notify=False),
    }
    r._profile_adapters = {"sec": {Platform.TELEGRAM: _Adapter()}, "quiet": {Platform.DISCORD: _Adapter()}}
    r._served_profile_homes = {}
    r._snapshot_running_agents = lambda: []

    await r._notify_active_sessions_of_shutdown()

    assert r.adapters[Platform.TELEGRAM].sent == ["8776018003"]
    assert r._profile_adapters["sec"][Platform.TELEGRAM].sent == ["8776018003"], "own bot, own conversation"
    assert r._profile_adapters["quiet"][Platform.DISCORD].sent == [], "served profile's opt-out is its own"
