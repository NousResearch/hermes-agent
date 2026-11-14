"""Callers refuse only the event/chat whose profile is not served, never the whole pass (#154610).

The resolver raises ``ProfileRouteRejected`` for a missing or parked named profile under
multiplexing; shutdown notices, the watch-notification drain and async-delegation delivery must
skip that one item and keep serving every other profile, without a retry storm.
"""

import queue
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult
from gateway.session import SessionSource, build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.fixture
def homes(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    served = default_home / "profiles" / "served"
    served.mkdir(parents=True)
    (served / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home, served


def _bare_runner(sources):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._build_process_event_source = lambda evt: sources[evt["session_key"]]
    runner._load_background_notifications_mode = lambda: "all"
    return runner


def _src(chat_id, profile):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, profile=profile)


def test_title_cased_served_profile_still_resolves(homes):
    default_home, served = homes
    runner = _bare_runner({})
    assert runner._resolve_profile_home_for_source(_src("1", "Served")) == served
    assert Path(runner._resolve_profile_home_for_source(_src("1", "Default"))) == default_home


@pytest.mark.asyncio
async def test_shutdown_notice_skips_only_the_unserved_profile_chat(homes):
    runner, adapter = make_restart_runner()
    runner.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    ghost = make_restart_source(chat_id="ghost-chat")
    ghost.profile = "ghost"
    ok = make_restart_source(chat_id="ok-chat")
    for source in (ghost, ok):
        key = build_session_key(source)
        runner._running_agents[key] = object()
        runner.session_store._entries[key] = MagicMock(origin=source)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m"))
    runner._delivery_adapter_for = lambda _source: adapter

    await runner._notify_active_sessions_of_shutdown()

    assert [call.args[0] for call in adapter.send.await_args_list] == ["ok-chat"]


@pytest.mark.asyncio
async def test_watch_drain_drops_unserved_event_and_delivers_the_rest(homes, monkeypatch):
    import gateway.run as gateway_run
    import tools.process_registry_notifications as notifications
    runner = _bare_runner({"g": _src("g", "ghost"), "ok": _src("ok", "served")})
    injected = []

    async def inject(_text, evt):
        injected.append(evt["session_key"])
        return True

    runner._inject_watch_notification = inject
    monkeypatch.setattr(notifications, "should_surface_notification", lambda _evt: True)
    monkeypatch.setattr(gateway_run, "_format_gateway_process_notification", lambda _evt: "x")
    pending = queue.Queue()
    for key in ("g", "ok"):
        pending.put({"type": "watch_match", "session_key": key})

    await runner._drain_watch_notifications(pending)

    assert injected == ["ok"]
    assert pending.empty()  # the refused event is dropped, not requeued


@pytest.mark.asyncio
async def test_async_group_for_unserved_profile_is_terminal_not_retried(homes):
    runner = _bare_runner({"agent:ghost:telegram:dm:g": _src("g", "ghost")})
    group = [{"type": "async_delegation", "session_key": "agent:ghost:telegram:dm:g"}]

    # None = nothing deliverable here: the watcher neither requeues nor logs an error per tick.
    assert await runner._deliver_async_delegation_group(group) is None
    completion = {"type": "completion", "session_key": "agent:ghost:telegram:dm:g"}
    assert await runner._deliver_completion_notification("x", completion) is None
    assert await runner._enqueue_process_completion_notification("x", completion) is None


@pytest.mark.asyncio
async def test_process_watcher_for_unserved_profile_ends_cleanly(homes):
    runner = _bare_runner({"agent:ghost:telegram:dm:g": _src("g", "ghost")})
    runner._watch_process = AsyncMock()
    await runner._run_process_watcher({"session_key": "agent:ghost:telegram:dm:g", "session_id": "p1"})
    runner._watch_process.assert_not_awaited()
