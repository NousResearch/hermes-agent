from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key
from hermes_state import SessionDB


def _make_event(text="/sessions", platform=Platform.TELEGRAM, user_id="alice", chat_id="chat-1"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        user_name=user_id,
        chat_id=chat_id,
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(session_db=None, current_session_id="current_session_001", event=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = session_db
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._evict_cached_agent = MagicMock()
    runner._async_flush_memories = AsyncMock()

    session_key = build_session_key(event.source) if event else "agent:main:telegram:dm"
    mock_session_entry = MagicMock()
    mock_session_entry.session_id = current_session_id
    mock_session_entry.session_key = session_key

    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    mock_store.load_transcript.return_value = []
    mock_store.switch_session.return_value = mock_session_entry
    runner.session_store = mock_store
    return runner


@pytest.mark.asyncio
async def test_sessions_list_only_shows_current_users_sessions(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("alice_history_001", "telegram", user_id="alice")
        db.set_session_title("alice_history_001", "Alice History")
        db.append_message("alice_history_001", role="user", content="alice transcript")

        db.create_session("bob_history_001", "telegram", user_id="bob")
        db.set_session_title("bob_history_001", "Bob History")
        db.append_message("bob_history_001", role="user", content="bob transcript")

        event = _make_event(text="/sessions", user_id="alice")
        runner = _make_runner(session_db=db, event=event)

        result = await runner._handle_sessions_command(event)

        assert "Alice History" in result
        assert "Bob History" not in result
    finally:
        db.close()


@pytest.mark.asyncio
async def test_sessions_view_rejects_other_users_session_id(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bob_history_001", "telegram", user_id="bob")
        db.set_session_title("bob_history_001", "Bob History")
        db.append_message("bob_history_001", role="user", content="bob transcript")

        event = _make_event(text="/sessions view bob_history_001", user_id="alice")
        runner = _make_runner(session_db=db, event=event)

        result = await runner._handle_sessions_command(event)

        assert "not found" in result.lower()
        assert "Bob History" not in result
    finally:
        db.close()
