"""A handoff into a Telegram DM topic never undoes a /new that finished while it was suspended."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import HomeChannel, Platform
from gateway.session import build_session_key
from tests.gateway.fixtures.storage_handles import close_storage_handles
from tests.gateway.test_telegram_topic_mode import _make_event, _make_source
from tests.gateway.test_telegram_topic_session_switch import CHAT, THREAD, _bound_session, _bound_topic, _entry


@pytest.mark.asyncio
async def test_handoff_suspended_after_its_switch_does_not_undo_a_newer_reset(tmp_path):
    """/new moves the route and writes its binding without a topic publication: a handoff that
    switched before it and binds after it must not restore the handed-off session."""
    db, runner = _bound_topic(tmp_path, "cli-session", "fresh")
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id=CHAT, name="Test DM", thread_id=THREAD)
    runner.adapters[Platform.TELEGRAM].create_handoff_thread = AsyncMock(return_value=None)
    runner._handle_message = AsyncMock(return_value="handoff ok")
    key = build_session_key(_make_source(thread_id=THREAD))
    routes = {}

    def reset(k):
        routes[k] = "fresh"
        return _entry(k, "fresh")

    peek = runner.session_store.peek_session_id.side_effect
    runner.session_store.peek_session_id.side_effect = lambda k: routes.get(k) or peek(k)
    runner.async_session_store.reset_session = AsyncMock(side_effect=reset)
    runner.session_store._entries = {}
    runner._hm_stop_owner_turn = AsyncMock()
    runner._cleanup_old_agent_for_reset = AsyncMock()
    runner._fire_session_reset_hooks = AsyncMock()
    repoint, parked, release = runner._handoff_repoint, asyncio.Event(), asyncio.Event()

    async def park_after_repoint(*args):
        entry = await repoint(*args)
        parked.set()
        await release.wait()
        return entry

    runner._handoff_repoint = park_after_repoint
    handoff = asyncio.create_task(runner._process_handoff(
        {"id": "cli-session", "title": "CLI work", "handoff_platform": "telegram"}))
    await asyncio.wait_for(parked.wait(), 10)
    await runner._handle_reset_command(_make_event("/new", thread_id=THREAD))
    assert runner.session_store.peek_session_id(key) == _bound_session(db) == "fresh"
    release.set()

    with pytest.raises(RuntimeError, match="switched to another session"):
        await asyncio.wait_for(handoff, 10)
    assert runner.session_store.peek_session_id(key) == _bound_session(db) == "fresh"
    runner._handle_message.assert_not_awaited()
