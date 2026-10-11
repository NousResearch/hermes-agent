"""Completion routing preserves the current conversation across lifecycle changes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture
def owner(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "voice.json")
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    source = SessionSource(Platform.DISCORD, chat_id="room", chat_type="thread",
                           thread_id="topic", user_id="person")
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db_for_key(entry.session_key)
    return runner, entry, db


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "idle_timeout"])
async def test_ordinary_old_owner_never_replaces_current_route(owner, reason):
    runner, original, db = owner
    if reason:
        db.end_session(original.session_id, reason)
    current = runner.session_store.reset_session(original.session_key)
    # An old incomplete reset write is not permission to undo the current mapping.
    db._write_sql("UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                  (1 if reason else None, reason, original.session_id))
    resolved = await runner._resolve_async_delegation_session(current, original.session_id)
    assert resolved is None
    assert runner.session_store.lookup_by_session_key(original.session_key).session_id == current.session_id


@pytest.mark.asyncio
async def test_idle_current_owner_and_compression_still_deliver(owner):
    runner, entry, db = owner
    db.end_session(entry.session_id, "idle_timeout")
    assert await runner._resolve_async_delegation_session(entry, entry.session_id) is entry
    db.reopen_session(entry.session_id)
    original_id = entry.session_id
    db.end_session(original_id, "compression")
    db.create_session("tip", source="discord", parent_session_id=entry.session_id,
                      session_key=entry.session_key)
    resolved = await runner._resolve_async_delegation_session(entry, entry.session_id)
    assert resolved.session_id == "tip"
    assert db.get_session(original_id)["end_reason"] == "compression"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_source", ["session_row", "route_load"])
async def test_text_final_retries_database_outage_once(owner, monkeypatch, failure_source):
    import tools.process_registry as processes

    runner, entry, db = owner
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter
    watcher = dict(session_id="process", check_interval=0, session_key=entry.session_key,
                   parent_session_id=entry.session_id, platform="discord", chat_id="room",
                   thread_id="topic", chat_type="thread", notify_on_complete=False)
    process = SimpleNamespace(session_key=entry.session_key, parent_session_id=entry.session_id,
                              owner_task_id="owner", task_id="owner", exited=True,
                              output_buffer="done", exit_code=0, command="echo done", started_at=None)
    registry = SimpleNamespace(get=Mock(side_effect=[process, process, None]),
                               is_completion_consumed=lambda _id: False)
    monkeypatch.setattr(processes, "process_registry", registry)
    read = db.get_session
    failed = False

    def once(session_id):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("temporary database read failure")
        return read(session_id)

    if failure_source == "session_row":
        monkeypatch.setattr(db, "get_session", once)
    else:
        lookup = runner.async_session_store.lookup_by_session_key

        async def unavailable_once(key):
            nonlocal failed
            if not failed:
                failed = True
                runner.session_store._routing_db_loaded = False
                return None
            runner.session_store._routing_db_loaded = True
            return await lookup(key)

        monkeypatch.setattr(runner.async_session_store, "lookup_by_session_key", unavailable_once)
    await runner._run_process_watcher(watcher)
    assert failed
    adapter.send.assert_awaited_once()
    assert registry.get.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, expected", [
    (SendResult(success=False, error="connection lost", retryable=True), False),
    (SendResult(success=False, error="Forbidden", retryable=False), None),
    (RuntimeError("403 Forbidden"), None),
])
async def test_direct_watcher_send_respects_transport_disposition(owner, failure, expected):
    runner, entry, _db = owner
    adapter = SimpleNamespace(send=AsyncMock())
    if isinstance(failure, Exception):
        adapter.send.side_effect = failure
    else:
        adapter.send.return_value = failure
    runner.adapters[Platform.DISCORD] = adapter
    watcher = dict(session_id="process", session_key=entry.session_key,
                   parent_session_id=entry.session_id, platform="discord", chat_id="room",
                   thread_id="topic", chat_type="thread")
    process = SimpleNamespace(session_key=entry.session_key, parent_session_id=entry.session_id,
                              owner_task_id="owner", task_id="owner")
    assert await runner._send_watcher_message("discord", "room", "topic", "done", watcher, process) is expected
    adapter.send.assert_awaited_once()
