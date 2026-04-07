import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _source(chat_id="123456"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
    )


@pytest.mark.asyncio
async def test_arm_self_nudge_replaces_existing_timer():
    runner = object.__new__(GatewayRunner)
    runner._self_nudge_tasks = {}
    runner._self_nudge_entries = {}
    runner._pending_hidden_turns = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}

    source = _source()
    session_key = build_session_key(source)

    first = await runner._arm_self_nudge(session_key, source, 300, "First note")
    second = await runner._arm_self_nudge(session_key, source, 120, "Second note")

    assert first["armed"] is True
    assert second["armed"] is True
    assert second["replaced_existing"] is True
    assert runner._self_nudge_entries[session_key]["note"] == "Second note"

    await runner._cancel_self_nudge(session_key, reason="test_cleanup")


@pytest.mark.asyncio
async def test_fire_self_nudge_queues_hidden_turn_when_session_busy(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._self_nudge_tasks = {}
    runner._self_nudge_entries = {}
    runner._pending_hidden_turns = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}

    source = _source()
    session_key = build_session_key(source)
    entry = {
        "session_key": session_key,
        "source": source.to_dict(),
        "delay_seconds": 5,
        "note": "Check the deploy result.",
    }
    runner._self_nudge_entries[session_key] = entry
    runner._running_agents[session_key] = object()

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    await runner._fire_self_nudge(entry)

    assert session_key not in runner._self_nudge_entries
    assert runner._pending_hidden_turns[session_key]["note"] == "Check the deploy result."


@pytest.mark.asyncio
async def test_run_self_nudge_entry_injects_hidden_turn():
    runner = object.__new__(GatewayRunner)
    runner._self_nudge_tasks = {}
    runner._self_nudge_entries = {}
    runner._pending_hidden_turns = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._handle_message_with_agent = AsyncMock(return_value=None)

    source = _source()
    session_key = build_session_key(source)
    entry = {
        "session_key": session_key,
        "source": source.to_dict(),
        "delay_seconds": 300,
        "note": "Check whether the background task finished.",
    }

    await runner._run_self_nudge_entry(entry)

    runner._handle_message_with_agent.assert_awaited_once()
    event, call_source, call_session_key = runner._handle_message_with_agent.call_args.args
    assert call_source.chat_id == source.chat_id
    assert call_session_key == session_key
    assert "self-nudge timer" in event.text.lower()
    assert "background task finished" in event.text.lower()
    assert event.persist_user_message == "[System note: a self-nudge timer fired.]"


@pytest.mark.asyncio
async def test_cancel_self_nudge_clears_pending_hidden_turn():
    runner = object.__new__(GatewayRunner)
    runner._self_nudge_tasks = {}
    runner._self_nudge_entries = {}
    runner._pending_hidden_turns = {}

    source = _source()
    session_key = build_session_key(source)
    task = asyncio.create_task(asyncio.sleep(60))
    runner._self_nudge_tasks[session_key] = task
    runner._self_nudge_entries[session_key] = {"session_key": session_key, "source": source.to_dict()}
    runner._pending_hidden_turns[session_key] = {"session_key": session_key}

    cancelled = await runner._cancel_self_nudge(session_key, reason="user_message")

    assert cancelled is True
    assert session_key not in runner._self_nudge_tasks
    assert session_key not in runner._self_nudge_entries
    assert session_key not in runner._pending_hidden_turns
    task.cancel()
