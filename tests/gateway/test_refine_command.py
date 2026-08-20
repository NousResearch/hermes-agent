"""Tests for gateway /refine cold-cache and persisted-transcript handling."""

from datetime import datetime
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key


def _make_source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/refine", *, platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(platform),
        message_id="m1",
    )


def _make_runner(session_entry: SessionEntry, *, platform: Platform = Platform.TELEGRAM):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner.hooks = SimpleNamespace(emit=MagicMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.peek_session_id.return_value = session_entry.session_id
    runner.session_store.load_transcript.return_value = []
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


def _session_entry(source=None) -> SessionEntry:
    source = source or _make_source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-refine-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=source.platform,
        chat_type=source.chat_type or "dm",
    )


@pytest.mark.asyncio
async def test_refine_cold_cache_with_persisted_turns_asks_to_resume():
    """Cold _agent_cache must not claim an empty conversation when transcript has turns."""
    entry = _session_entry()
    runner = _make_runner(entry)
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "content": "ignored for count"},
    ]

    result = await runner._handle_refine_command(_make_event("/refine"))

    assert "2 persisted user/assistant messages" in result
    assert "resume" in result.lower() or "wake the session first" in result
    assert "Nothing to refine yet" not in result
    runner.session_store.load_transcript.assert_called_once_with(
        entry.session_id, raise_on_error=True
    )


@pytest.mark.asyncio
async def test_refine_cold_cache_unknown_session_does_not_create_one():
    entry = _session_entry()
    runner = _make_runner(entry)
    runner.session_store.peek_session_id.return_value = None

    result = await runner._handle_refine_command(_make_event())

    assert result == "Nothing to refine yet — send a message first."
    runner.session_store.get_or_create_session.assert_not_called()
    runner.session_store.load_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_refine_cold_cache_truly_empty_keeps_nothing_to_refine():
    entry = _session_entry()
    runner = _make_runner(entry)
    runner.session_store.load_transcript.return_value = []

    result = await runner._handle_refine_command(_make_event())

    assert result == "Nothing to refine yet — send a message first."


@pytest.mark.asyncio
async def test_refine_cold_cache_transcript_read_failure_is_distinct():
    entry = _session_entry()
    runner = _make_runner(entry)
    runner.session_store.load_transcript.side_effect = RuntimeError("db down")

    result = await runner._handle_refine_command(_make_event())

    assert "Couldn't read the persisted conversation" in result
    assert "Nothing to refine yet" not in result


def test_refine_strict_transcript_read_propagates_store_failure():
    class _FailingDB:
        def get_compression_tip(self, _session_id):
            return None

        def get_messages_as_conversation(self, _session_id, **_kwargs):
            raise RuntimeError("db down")

    store = object.__new__(SessionStore)
    store._db = _FailingDB()
    store._transcript_reroutes = {}

    with pytest.raises(RuntimeError, match="db down"):
        store.load_transcript("sess-refine-1", raise_on_error=True)


def test_refine_strict_transcript_read_rejects_unavailable_database():
    store = object.__new__(SessionStore)
    store._db = None

    with pytest.raises(RuntimeError, match="database is unavailable"):
        store.load_transcript("sess-refine-1", raise_on_error=True)


def test_refine_strict_transcript_read_propagates_tip_lookup_failure():
    class _FailingTipDB:
        def get_compression_tip(self, _session_id):
            raise RuntimeError("tip lookup failed")

    store = object.__new__(SessionStore)
    store._db = _FailingTipDB()
    store._transcript_reroutes = {}

    with pytest.raises(RuntimeError, match="tip lookup failed"):
        store.load_transcript("sess-refine-1", raise_on_error=True)


@pytest.mark.asyncio
async def test_refine_cached_agent_uses_persisted_transcript_when_messages_empty():
    entry = _session_entry()
    runner = _make_runner(entry)
    captured = {}

    class _Agent:
        valid_tool_names = {"memory", "skill_manage"}
        _session_messages = []

        def _spawn_background_review(self, **kwargs):
            captured.update(kwargs)

    source = _make_source()
    session_key = build_session_key(source)
    runner._agent_cache[session_key] = (_Agent(), "sig")
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "persisted question"},
        {"role": "assistant", "content": "persisted answer"},
        {"role": "tool", "content": "tool output"},
    ]

    result = await runner._handle_refine_command(_make_event("/refine save workflow"))

    assert "Reviewing this conversation" in result
    assert captured["focus"] == "save workflow"
    assert captured["review_memory"] is True
    assert captured["review_skills"] is True
    assert captured["messages_snapshot"] == [
        {"role": "user", "content": "persisted question"},
        {"role": "assistant", "content": "persisted answer"},
        {"role": "tool", "content": "tool output"},
    ]


@pytest.mark.asyncio
async def test_refine_cached_agent_prefers_in_memory_messages():
    entry = _session_entry()
    runner = _make_runner(entry)
    captured = {}

    class _Agent:
        valid_tool_names = {"memory"}
        _session_messages = [
            {"role": "user", "content": "live question"},
            {"role": "assistant", "content": "live answer"},
        ]

        def _spawn_background_review(self, **kwargs):
            captured.update(kwargs)

    source = _make_source()
    runner._agent_cache[build_session_key(source)] = (_Agent(), "sig")
    runner.session_store.load_transcript.side_effect = AssertionError(
        "must not load transcript when in-memory messages exist"
    )

    result = await runner._handle_refine_command(_make_event())

    assert "Reviewing this conversation" in result
    assert captured["messages_snapshot"] == [
        {"role": "user", "content": "live question"},
        {"role": "assistant", "content": "live answer"},
    ]
    assert captured["review_skills"] is False
