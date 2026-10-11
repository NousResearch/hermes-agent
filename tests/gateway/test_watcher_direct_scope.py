"""Direct watcher notices resolve their owner from the keyed store, not ambient profile state."""

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_state import AsyncSessionDB


def _admitting_adapter():
    async def accept(event):
        event._gateway_accepted = True

    return SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock(side_effect=accept))


def _profile_watcher(monkeypatch, tmp_path):
    import gateway.run as gateway_run
    from hermes_cli.profiles import get_profile_dir

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_BASE_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    research_home = get_profile_dir("research")
    research_home.mkdir(parents=True)
    (research_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    runner = GatewayRunner(GatewayConfig(
        multiplex_profiles=True, sessions_dir=tmp_path / "sessions",
    ))
    adapter = _admitting_adapter()
    runner._profile_adapters = {"research": {Platform.DISCORD: adapter}}
    entry = runner.session_store.get_or_create_session(SessionSource(
        platform=Platform.DISCORD, chat_id="room", chat_type="thread",
        thread_id="topic", user_id="person", profile="research",
    ))
    owner_db = runner.session_store._db_for_key(entry.session_key)
    assert owner_db.get_session(entry.session_id) is not None
    wrong_db = runner.session_store._db_for_key("agent:main:discord:thread:other")
    assert wrong_db.get_session(entry.session_id) is None
    runner._session_db = AsyncSessionDB(wrong_db)
    watcher = {
        "session_id": "proc_old", "session_key": entry.session_key,
        "parent_session_id": entry.session_id, "platform": "discord",
        "chat_id": "room", "chat_type": "thread", "thread_id": "topic",
    }
    process = SimpleNamespace(
        parent_session_id=entry.session_id, session_key=entry.session_key,
        owner_task_id="owner", task_id="owner",
    )
    return runner, adapter, watcher, process, owner_db


@pytest.mark.asyncio
async def test_direct_watcher_uses_exact_profile_owner_despite_wrong_ambient_db(monkeypatch, tmp_path):
    runner, adapter, watcher, process, _owner_db = _profile_watcher(monkeypatch, tmp_path)

    assert await runner._watcher_message_route_owned(watcher, process)
    await runner._send_watcher_message("discord", "room", "topic", "status", watcher, process)

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("room", "status")


@pytest.mark.asyncio
async def test_direct_watcher_fails_closed_when_exact_owner_db_unreadable(monkeypatch, tmp_path):
    runner, adapter, watcher, process, owner_db = _profile_watcher(monkeypatch, tmp_path)
    monkeypatch.setattr(owner_db, "get_session", Mock(side_effect=RuntimeError("offline")))

    assert not await runner._watcher_message_route_owned(watcher, process)
    await runner._send_watcher_message("discord", "room", "topic", "status", watcher, process)

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_watch_retries_initial_routing_load_outage_without_json_mirror(
    monkeypatch, tmp_path,
):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: all\n", encoding="utf-8",
    )
    config = GatewayConfig(sessions_dir=tmp_path / "sessions", write_sessions_json=False)
    before = GatewayRunner(config)
    entry = before.session_store.get_or_create_session(SessionSource(
        platform=Platform.DISCORD, chat_id="room", chat_type="thread",
        thread_id="topic", user_id="person",
    ))
    assert not (config.sessions_dir / "sessions.json").exists()

    runner = GatewayRunner(config)
    adapter = _admitting_adapter()
    runner.adapters[Platform.DISCORD] = adapter
    routing_db = runner.session_store._routing_db
    load = routing_db.load_gateway_routing_entries
    monkeypatch.setattr(routing_db, "load_gateway_routing_entries",
                        Mock(side_effect=RuntimeError("routing DB unreadable")))
    event = {
        "type": "watch_match", "session_id": "proc_old", "session_key": entry.session_key,
        "parent_session_id": entry.session_id, "platform": "discord",
        "chat_id": "room", "chat_type": "thread", "thread_id": "topic",
        "pattern": "READY", "output": "READY", "command": "build",
    }
    events = queue.Queue()
    events.put(event)

    await runner._drain_watch_notifications(events)
    adapter.handle_message.assert_not_awaited()
    assert events.get_nowait() is event
    assert not runner.session_store._routing_db_loaded

    monkeypatch.setattr(routing_db, "load_gateway_routing_entries", load)
    events.put(event)
    await runner._drain_watch_notifications(events)
    adapter.handle_message.assert_awaited_once()
    assert events.empty()
