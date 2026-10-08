"""``schedule_wake`` deadlines are fired by the SURFACE that owns the session (#122444).

The tool only persists ``wake:<session_id>``; without a driver per surface a wake armed from a
messaging chat or a Desktop session is armed-but-dead. The gateway watcher fires chat-routed wakes
through the session's own profile bot (fail closed, estop-held); the TUI/Desktop poller fires
route-less wakes into the live session and refunds a dispatch that never started.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class _Adapter:
    def __init__(self):
        self.handled = []

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
    r._run_in_executor_with_context = asyncio.to_thread
    return r


@pytest.mark.asyncio
async def test_gateway_watcher_fires_chat_routed_wake_through_its_own_bot_and_holds_under_estop(hermes_home):
    """A wake armed inside a secondary profile's Telegram chat is injected through THAT profile's
    adapter as a rendered ``[Wake …]`` turn, exactly once; a bot-less profile never claims it; while
    ``hermes pause`` is engaged the wake stays armed instead of being consumed."""
    from agent import estop
    from gateway.run_wake_watcher import wake_fire_one
    from hermes_cli.wake import load_wake, schedule_wake

    r = _runner()
    route = {"platform": "telegram", "chat_id": "42", "chat_type": "dm", "user_id": "42", "profile": "sec"}
    schedule_wake("sid", "check the deploy", time.time() - 1, route=route)
    schedule_wake("sid2", "orphan", time.time() - 1, route={**route, "profile": "nobot"})

    estop.engage(reason="maintenance")
    try:
        assert await wake_fire_one(r, "sid", load_wake("sid"), time.time(), set()) is False
        assert load_wake("sid").armed and r._profile_adapters["sec"][Platform.TELEGRAM].handled == []
    finally:
        estop.disengage()

    assert await wake_fire_one(r, "sid", load_wake("sid"), time.time(), set()) is True
    assert await wake_fire_one(r, "sid2", load_wake("sid2"), time.time(), set()) is False
    handled = r._profile_adapters["sec"][Platform.TELEGRAM].handled
    assert len(handled) == 1 and handled[0].startswith("[Wake") and "check the deploy" in handled[0]
    assert r.adapters[Platform.TELEGRAM].handled == []  # never the default bot
    assert not load_wake("sid").armed and load_wake("sid").fire_count == 1
    assert load_wake("sid2").armed  # no adapter → unclaimed
    assert await wake_fire_one(r, "sid", load_wake("sid"), time.time(), set()) is False  # one-shot


@pytest.fixture()
def tui_server(hermes_home):
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()


def test_tui_poller_fires_route_less_wake_once_and_refunds_a_dispatch_that_never_started(tui_server):
    """The Desktop/TUI session-owner poller drives route-less wakes: claims the idle session, submits
    the rendered prompt through ``_run_prompt_submit`` once, leaves chat-routed wakes to the gateway,
    and rewinds the fire (wake stays armed) when the submit refuses the turn."""
    from hermes_cli.wake import load_wake, schedule_wake

    sid, key = "sid-wake", "tui-wake-session"
    s = {"session_key": key, "history": [], "history_lock": threading.Lock(), "history_version": 0,
         "running": False, "attached_images": [], "cols": 120, "agent": MagicMock(), "source": "desktop"}
    tui_server._sessions[sid] = s
    tui_server._get_db().create_session(key, source="desktop")
    dispatched: list[str] = []

    schedule_wake(key, "poll the build", time.time() - 1)
    with patch.object(tui_server, "_run_prompt_submit", lambda rid, sid_, sess, text, **kw: dispatched.append(text) or True), \
            patch.object(tui_server, "_emit"):
        tui_server._maybe_fire_tui_wake(sid, s)
        tui_server._maybe_fire_tui_wake(sid, s)  # claimed/running → no second dispatch
    assert len(dispatched) == 1 and dispatched[0].startswith("[Wake") and "poll the build" in dispatched[0]
    assert s["running"] is True and not load_wake(key).armed

    s["running"] = False
    schedule_wake(key, "routed elsewhere", time.time() - 1, route={"platform": "telegram", "chat_id": "42"})
    with patch.object(tui_server, "_run_prompt_submit", MagicMock(return_value=True)) as submit, \
            patch.object(tui_server, "_emit"):
        tui_server._maybe_fire_tui_wake(sid, s)
    assert not submit.called and load_wake(key).armed  # gateway-owned

    schedule_wake(key, "refused", time.time() - 1)
    with patch.object(tui_server, "_run_prompt_submit", MagicMock(return_value=False)), \
            patch.object(tui_server, "_emit"):
        tui_server._maybe_fire_tui_wake(sid, s)
    assert s["running"] is False and load_wake(key).armed and load_wake(key).fire_count == 1
