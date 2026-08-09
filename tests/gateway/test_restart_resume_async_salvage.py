"""Current-architecture regressions salvaged from PR #30030."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from tests.gateway.restart_test_helpers import make_restart_runner


def _pending_entry(*, session_id="sid", thread_id="topic"):
    now = datetime.now()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        chat_type="dm",
        thread_id=thread_id,
        message_id="stale-anchor",
    )
    return SessionEntry(
        session_key="agent:main:telegram:dm:u1",
        session_id=session_id,
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=now,
    )


@pytest.mark.asyncio
async def test_resume_uses_async_store_and_latest_user_reply_anchor():
    runner, adapter = make_restart_runner()
    entry = _pending_entry()
    runner.session_store._entries = {entry.session_key: entry}
    runner._async_session_store.load_transcript = AsyncMock(
        return_value=[
            {"role": "user", "content": "old", "message_id": "111", "timestamp": datetime.now().timestamp()},
            {"role": "assistant", "content": "working", "finish_reason": "tool_calls"},
            {"role": "user", "content": "latest", "message_id": "222", "timestamp": datetime.now().timestamp()},
        ]
    )
    captured = []

    async def _capture(event):
        captured.append(event)

    adapter.handle_message = _capture
    scheduled = await runner._schedule_resume_pending_sessions()
    await __import__("asyncio").sleep(0)

    assert scheduled == 1
    assert captured
    event = captured[0]
    assert isinstance(event, MessageEvent)
    assert event.message_id == "222"
    assert event.reply_to_message_id == "222"
    assert event.source.message_id == "222"
    runner._async_session_store.load_transcript.assert_awaited_once_with("sid")


@pytest.mark.asyncio
async def test_resume_follows_topic_binding_compression_tip():
    runner, adapter = make_restart_runner()
    entry = _pending_entry(session_id="routing-parent")
    runner.session_store._entries = {entry.session_key: entry}
    runner._session_db = MagicMock()
    runner._session_db.get_telegram_topic_binding = AsyncMock(
        return_value={"session_id": "bound-parent"}
    )
    runner._session_db.get_compression_tip = AsyncMock(return_value="bound-child")
    runner._async_session_store.load_transcript = AsyncMock(
        return_value=[
            {"role": "user", "content": "continue", "message_id": "333", "timestamp": datetime.now().timestamp()}
        ]
    )
    adapter.handle_message = AsyncMock()

    assert await runner._schedule_resume_pending_sessions() == 1
    runner._async_session_store.load_transcript.assert_awaited_once_with("bound-child")


@pytest.mark.asyncio
async def test_completed_assistant_tail_clears_marker_without_rerun():
    runner, adapter = make_restart_runner()
    entry = _pending_entry()
    runner.session_store._entries = {entry.session_key: entry}
    runner._async_session_store.load_transcript = AsyncMock(
        return_value=[
            {"role": "user", "content": "question", "message_id": "444", "timestamp": datetime.now().timestamp()},
            {"role": "assistant", "content": "done", "finish_reason": "stop", "timestamp": datetime.now().timestamp()},
        ]
    )
    adapter.handle_message = AsyncMock()

    assert await runner._schedule_resume_pending_sessions() == 0
    runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
        entry.session_key
    )
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_notification_prefers_live_reply_anchor():
    runner, adapter = make_restart_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        chat_type="dm",
        thread_id="topic",
        message_id="live-anchor",
    )
    key = "agent:main:telegram:dm:u1"
    runner._running_agents = {key: object()}
    runner._cache_session_source(key, source)
    runner.session_store._entries[key] = _pending_entry()
    adapter.send = AsyncMock(return_value=SendResult(success=True))

    await runner._notify_active_sessions_of_shutdown()

    metadata = adapter.send.await_args.kwargs["metadata"]
    assert metadata["telegram_reply_to_message_id"] == "live-anchor"
    runner._async_session_store._ensure_loaded.assert_not_awaited()
