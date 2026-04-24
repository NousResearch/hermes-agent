"""Tests for gateway auto-resume of interrupted sessions after restart."""

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


class DummyAdapter:
    def __init__(self):
        self.events = []

    async def handle_message(self, event: MessageEvent):
        self.events.append(event)
        return "ok"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
        user_name="Maxim",
        chat_type="dm",
    )


def _runner(tmp_path, store: SessionStore, adapter: DummyAdapter) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running = True
    runner._running_agents = {}
    runner._background_tasks = set()
    return runner


@pytest.mark.asyncio
async def test_schedule_auto_resume_injects_internal_event(tmp_path):
    store = SessionStore(tmp_path, GatewayConfig(sessions_dir=tmp_path))
    source = _source()
    entry = store.get_or_create_session(source)
    assert store.mark_resume_pending(entry.session_key, reason="restart_timeout")

    adapter = DummyAdapter()
    runner = _runner(tmp_path, store, adapter)

    runner._schedule_auto_resume_pending_sessions()
    assert len(runner._background_tasks) == 1
    await asyncio.gather(*list(runner._background_tasks))

    assert len(adapter.events) == 1
    event = adapter.events[0]
    assert event.internal is True
    assert event.source == source
    assert event.message_id == f"auto-resume:{entry.session_id}"
    assert "Automatically continue the interrupted task" in event.text


@pytest.mark.asyncio
async def test_schedule_auto_resume_skips_suspended_sessions(tmp_path):
    store = SessionStore(tmp_path, GatewayConfig(sessions_dir=tmp_path))
    source = _source()
    entry = store.get_or_create_session(source)
    assert store.mark_resume_pending(entry.session_key, reason="restart_timeout")
    entry.suspended = True
    store._save()

    adapter = DummyAdapter()
    runner = _runner(tmp_path, store, adapter)

    runner._schedule_auto_resume_pending_sessions()
    assert len(runner._background_tasks) == 0
    assert adapter.events == []


@pytest.mark.asyncio
async def test_schedule_auto_resume_skips_missing_adapter(tmp_path):
    store = SessionStore(tmp_path, GatewayConfig(sessions_dir=tmp_path))
    source = _source()
    entry = store.get_or_create_session(source)
    assert store.mark_resume_pending(entry.session_key, reason="restart_timeout")

    adapter = DummyAdapter()
    runner = _runner(tmp_path, store, adapter)
    runner.adapters = {}

    runner._schedule_auto_resume_pending_sessions()
    assert len(runner._background_tasks) == 0
    assert adapter.events == []
