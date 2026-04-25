"""Tests confirmation guards for destructive gateway session commands."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
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


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-old",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    new_session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-new",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.reset_session.return_value = new_session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    return runner


@pytest.mark.asyncio
async def test_new_requires_confirmation_before_reset():
    runner = _make_runner()

    result = await runner._handle_reset_command(_make_event("/new"))

    assert "/new --yes" in result
    runner.session_store.reset_session.assert_not_called()


@pytest.mark.asyncio
async def test_reset_alias_confirmation_mentions_reset():
    runner = _make_runner()

    result = await runner._handle_reset_command(_make_event("/reset"))

    assert "/reset --yes" in result
    runner.session_store.reset_session.assert_not_called()


@pytest.mark.asyncio
async def test_undo_requires_confirmation_before_rewriting_transcript():
    runner = _make_runner()
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    result = await runner._handle_undo_command(_make_event("/undo"))

    assert "/undo --yes" in result
    runner.session_store.rewrite_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_undo_with_confirmation_rewrites_transcript():
    runner = _make_runner()
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "undo this"},
        {"role": "assistant", "content": "ok"},
    ]

    result = await runner._handle_undo_command(_make_event("/undo --yes"))

    runner.session_store.rewrite_transcript.assert_called_once_with(
        "sess-old",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )
    assert "Undid 2 message(s)." in result
