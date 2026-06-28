"""Regression tests for /resume numbered selection excluding current session (#54326).

When /sessions displays a numbered list (filtering the current session), /resume N
must resolve to the same session the user sees at position N — not N+1 because the
current session was included in the internal list.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_entry(session_id: str, source: SessionSource | None = None) -> SessionEntry:
    source = source or _make_source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=source.platform,
        chat_type=source.chat_type,
    )


def _make_runner(current_session_id: str, db_sessions: list[dict]):
    """Create a minimal GatewayRunner mock for /resume testing."""
    from gateway.run import GatewayRunner

    source = _make_source()
    session_key = build_session_key(source)
    current_entry = _make_entry(current_session_id, source)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None

    # Session store: _entries dict keyed by session_key
    runner.session_store = MagicMock()
    runner.session_store._entries = {session_key: current_entry}
    runner.session_store.get_or_create_session.return_value = current_entry
    runner.session_store.switch_session.return_value = current_entry
    runner.session_store.load_transcript.return_value = []

    # Session DB: list_sessions_rich returns all sessions (including current)
    runner._session_db = AsyncMock()
    runner._session_db.list_sessions_rich = AsyncMock(return_value=db_sessions)
    runner._session_db.resolve_resume_session_id = AsyncMock(side_effect=lambda sid: sid)
    runner._session_db.get_session = AsyncMock(return_value=None)
    runner._session_db.get_session_title = AsyncMock(return_value=None)

    return runner, session_key


# Four sessions: current is #2 (index 1) in the DB list.
# /sessions shows [A, C, D] (current B filtered). /resume 3 should pick D, not C.
_DB_SESSIONS = [
    {"id": "sess-a", "title": "Daily Food Tracking", "preview": ""},
    {"id": "sess-b", "title": "Chain-of-Thought Display", "preview": ""},  # current
    {"id": "sess-c", "title": "Hermes vs OpenClaw", "preview": ""},
    {"id": "sess-d", "title": "Hello Victor", "preview": ""},
]


@pytest.mark.asyncio
async def test_resume_numeric_excludes_current_session():
    """/resume 3 must resolve to the 3rd session in the filtered list (excluding current)."""
    runner, _ = _make_runner("sess-b", _DB_SESSIONS)

    # After the fix, _list_titled_sessions filters out sess-b, so:
    #   filtered = [sess-a, sess-c, sess-d]
    #   /resume 3 → sess-d (index 2)
    result = await runner._handle_resume_command(_make_event("/resume 3"))

    # Should switch to sess-d (Hello Victor), not sess-c (Hermes vs OpenClaw)
    runner.session_store.switch_session.assert_called_once()
    call_args = runner.session_store.switch_session.call_args
    assert call_args[0][1] == "sess-d", (
        f"/resume 3 should resolve to sess-d (Hello Victor), "
        f"but got {call_args[0][1]}"
    )


@pytest.mark.asyncio
async def test_resume_numeric_first_item_after_current_filtered():
    """/resume 1 must resolve to the first session in the filtered list."""
    runner, _ = _make_runner("sess-b", _DB_SESSIONS)

    result = await runner._handle_resume_command(_make_event("/resume 1"))

    runner.session_store.switch_session.assert_called_once()
    call_args = runner.session_store.switch_session.call_args
    assert call_args[0][1] == "sess-a", (
        f"/resume 1 should resolve to sess-a (Daily Food Tracking), "
        f"but got {call_args[0][1]}"
    )


@pytest.mark.asyncio
async def test_resume_numeric_out_of_range_after_filter():
    """/resume N where N > filtered count should report out of range."""
    runner, _ = _make_runner("sess-b", _DB_SESSIONS)

    # Filtered list has 3 items; /resume 4 should be out of range
    result = await runner._handle_resume_command(_make_event("/resume 4"))

    assert "out of range" in result.lower() or "4" in result


@pytest.mark.asyncio
async def test_resume_list_excludes_current_session():
    """/resume (no args) should list sessions excluding the current one."""
    runner, _ = _make_runner("sess-b", _DB_SESSIONS)

    result = await runner._handle_resume_command(_make_event("/resume"))

    # Current session title should NOT appear in the list
    assert "Chain-of-Thought Display" not in result
    # Other sessions should appear
    assert "Daily Food Tracking" in result
    assert "Hermes vs OpenClaw" in result
    assert "Hello Victor" in result


@pytest.mark.asyncio
async def test_resume_current_session_is_first_in_db():
    """Edge case: current session is first in DB. /resume 1 should pick DB's 2nd item."""
    db_sessions = [
        {"id": "sess-cur", "title": "Current Session", "preview": ""},  # current, first
        {"id": "sess-x", "title": "Session X", "preview": ""},
        {"id": "sess-y", "title": "Session Y", "preview": ""},
    ]
    runner, _ = _make_runner("sess-cur", db_sessions)

    result = await runner._handle_resume_command(_make_event("/resume 1"))

    runner.session_store.switch_session.assert_called_once()
    call_args = runner.session_store.switch_session.call_args
    assert call_args[0][1] == "sess-x"


@pytest.mark.asyncio
async def test_resume_current_session_is_last_in_db():
    """Edge case: current session is last in DB. /resume N picks from front of list."""
    db_sessions = [
        {"id": "sess-x", "title": "Session X", "preview": ""},
        {"id": "sess-y", "title": "Session Y", "preview": ""},
        {"id": "sess-cur", "title": "Current Session", "preview": ""},  # current, last
    ]
    runner, _ = _make_runner("sess-cur", db_sessions)

    result = await runner._handle_resume_command(_make_event("/resume 2"))

    runner.session_store.switch_session.assert_called_once()
    call_args = runner.session_store.switch_session.call_args
    assert call_args[0][1] == "sess-y"


@pytest.mark.asyncio
async def test_resume_no_current_session_entry():
    """When session_key is not in _entries, all sessions should be listed."""
    from gateway.run import GatewayRunner

    source = _make_source()
    session_key = build_session_key(source)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None

    runner.session_store = MagicMock()
    runner.session_store._entries = {}  # no entry for this session_key
    runner.session_store.get_or_create_session.return_value = _make_entry("sess-new", source)
    runner.session_store.switch_session.return_value = _make_entry("sess-a", source)
    runner.session_store.load_transcript.return_value = []

    runner._session_db = AsyncMock()
    runner._session_db.list_sessions_rich = AsyncMock(return_value=_DB_SESSIONS)
    runner._session_db.resolve_resume_session_id = AsyncMock(side_effect=lambda sid: sid)
    runner._session_db.get_session = AsyncMock(return_value=None)
    runner._session_db.get_session_title = AsyncMock(return_value=None)

    result = await runner._handle_resume_command(_make_event("/resume 1"))

    # No current session to filter, so /resume 1 picks the first DB entry
    runner.session_store.switch_session.assert_called_once()
    call_args = runner.session_store.switch_session.call_args
    assert call_args[0][1] == "sess-a"
