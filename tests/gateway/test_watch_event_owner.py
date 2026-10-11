"""A queued process watch event cannot wake a later conversation on the same chat key."""

import queue
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.process_registry import ProcessRegistry, ProcessSession


def _runner(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: all\n", encoding="utf-8",
    )
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))

    async def accept(event):
        event._gateway_accepted = True

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=accept))
    runner.adapters[Platform.DISCORD] = adapter
    entry = runner.session_store.get_or_create_session(SessionSource(
        platform=Platform.DISCORD, chat_id="room", chat_type="thread",
        thread_id="topic", user_id="person",
    ))
    return runner, adapter, entry


def _event(entry, kind):
    registry = ProcessRegistry()
    process = ProcessSession(
        id="proc_old", command="build", task_id="owner", session_key=entry.session_key,
        parent_session_id=entry.session_id, watcher_platform="discord",
        watcher_chat_id="room", watcher_thread_id="topic", watch_patterns=["READY"],
        started_at=time.time() - 10, heartbeat_seconds=60,
    )
    if kind == "watch_match":
        registry._check_watch_patterns(process, "READY\n")
    elif kind == "watch_disabled":
        registry._emit_watch_disabled(process, 1, "rate limit. ")
    else:
        process.append_output("READY\n")
        registry._emit_heartbeat(process, time.time())
    return registry.completion_queue.get_nowait()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["watch_match", "watch_disabled", "heartbeat"])
async def test_process_watch_event_from_old_session_does_not_wake_new_session(
    monkeypatch, tmp_path, kind,
):
    runner, adapter, entry = _runner(monkeypatch, tmp_path)
    event = _event(entry, kind)
    replacement = runner.session_store.reset_session(entry.session_key)
    assert replacement.session_id != entry.session_id
    events = queue.Queue()
    events.put(event)

    await runner._drain_watch_notifications(events)

    adapter.handle_message.assert_not_awaited()
    assert events.empty()  # A proved /new boundary is acknowledged, never requeued forever.
    assert event["parent_session_id"] == entry.session_id


@pytest.mark.asyncio
async def test_direct_watch_injection_rechecks_owner_before_adapter_admission(monkeypatch, tmp_path):
    runner, adapter, entry = _runner(monkeypatch, tmp_path)
    event = {**_event(entry, "watch_match"), "parent_session_id": entry.session_id}
    runner.session_store.reset_session(entry.session_key)

    assert await runner._inject_watch_notification("[SYSTEM: matched]", event) is None
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_unpinned_legacy_watch_event_fails_closed(monkeypatch, tmp_path):
    runner, adapter, entry = _runner(monkeypatch, tmp_path)
    event = _event(entry, "watch_match")
    event.pop("parent_session_id", None)
    events = queue.Queue()
    events.put(event)

    await runner._drain_watch_notifications(events)

    adapter.handle_message.assert_not_awaited()
    assert events.empty()


@pytest.mark.asyncio
async def test_watch_event_follows_verified_compression_lineage(monkeypatch, tmp_path):
    runner, adapter, entry = _runner(monkeypatch, tmp_path)
    event = {**_event(entry, "watch_match"), "parent_session_id": entry.session_id}
    db = runner.session_store._db_for_key(entry.session_key)
    db.end_session(entry.session_id, "compression")
    db.create_session("continuation", source="discord", parent_session_id=entry.session_id,
                      session_key=entry.session_key)
    assert runner.session_store.switch_session(
        entry.session_key, "continuation", expected_session_id=entry.session_id,
    ).session_id == "continuation"

    assert await runner._inject_watch_notification("[SYSTEM: matched]", event) is True
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handed_off_child_watch_event_stays_with_its_parent(monkeypatch, tmp_path):
    runner, adapter, entry = _runner(monkeypatch, tmp_path)
    runner.session_store._db_for_key(entry.session_key).create_session(
        "delegate-child", source="subagent", parent_session_id=entry.session_id,
    )
    event = {**_event(entry, "watch_match"), "parent_session_id": "delegate-child"}

    assert await runner._inject_watch_notification("[SYSTEM: matched]", event) is True
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["route_index", "session_db"])
async def test_transient_watch_route_lookup_requeues_without_waking(monkeypatch, tmp_path, failure):
    runner, adapter, entry = _runner(monkeypatch, tmp_path)
    event = {**_event(entry, "watch_match"), "parent_session_id": entry.session_id}
    events = queue.Queue()
    events.put(event)
    if failure == "route_index":
        monkeypatch.setattr(runner.async_session_store, "lookup_by_session_key",
                            AsyncMock(side_effect=RuntimeError("route index unavailable")))
    else:
        db = runner.session_store._db_for_key(entry.session_key)
        monkeypatch.setattr(db, "get_session", Mock(side_effect=RuntimeError("database unavailable")))

    await runner._drain_watch_notifications(events)

    adapter.handle_message.assert_not_awaited()
    assert events.get_nowait() is event
    assert events.empty()
