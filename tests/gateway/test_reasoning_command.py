"""Tests for gateway /reasoning command behavior."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="123", chat_id="456"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    session_entry = MagicMock()
    session_entry.session_id = "session_123"
    session_entry.session_key = "agent:main:telegram:dm"
    session_entry.reasoning_mode = "off"

    store = MagicMock()
    store.get_or_create_session.return_value = session_entry
    runner.session_store = store
    return runner, store, session_entry


@pytest.mark.asyncio
async def test_reasoning_command_shows_current_mode():
    runner, _, _ = _make_runner()

    result = await runner._handle_reasoning_command(_make_event("/reasoning"))

    assert "Current reasoning mode" in result
    assert "disable reasoning" in result


@pytest.mark.asyncio
async def test_reasoning_command_persists_stream_mode():
    runner, store, session_entry = _make_runner()

    result = await runner._handle_reasoning_command(_make_event("/reasoning stream"))

    store.set_reasoning_mode.assert_called_once_with(session_entry.session_key, "stream")
    assert "enabled and will stream live" in result


@pytest.mark.asyncio
async def test_reasoning_command_persists_on_mode_as_hidden():
    runner, store, session_entry = _make_runner()

    result = await runner._handle_reasoning_command(_make_event("/reasoning on"))

    store.set_reasoning_mode.assert_called_once_with(session_entry.session_key, "on")
    assert result == "Reasoning enabled and hidden for this chat."


def test_resolve_session_reasoning_config_disables_when_off():
    runner, _, _ = _make_runner()
    runner._reasoning_config = {"enabled": True, "effort": "high"}

    resolved = runner._resolve_session_reasoning_config("off")

    assert resolved == {"enabled": False}


def test_resolve_session_reasoning_config_enables_when_globally_disabled():
    runner, _, _ = _make_runner()
    runner._reasoning_config = {"enabled": False}

    resolved = runner._resolve_session_reasoning_config("on")

    assert resolved == {"enabled": True, "effort": "medium"}


def test_format_reasoning_preview_italicizes_each_line():
    runner, _, _ = _make_runner()

    payload = runner._format_reasoning_preview("first line\n\nsecond line")

    assert payload == "💭 **Reasoning**\n\n*first line*\n\n*second line*"


@pytest.mark.asyncio
async def test_delete_preview_message_deletes_transient_reasoning_bubble():
    runner, _, _ = _make_runner()
    adapter = SimpleNamespace(delete_message=AsyncMock(return_value=True))
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123",
        chat_id="456",
        user_name="tester",
    )

    await runner._delete_preview_message(adapter, source, "789")

    adapter.delete_message.assert_awaited_once_with(chat_id="456", message_id="789")


@pytest.mark.asyncio
async def test_reasoning_preview_does_not_send_duplicate_after_edit_failure():
    runner, _, _ = _make_runner()
    next_message_id = {"value": 1}

    async def _send(chat_id, content, metadata=None):
        message_id = f"preview-{next_message_id['value']}"
        next_message_id["value"] += 1
        return SimpleNamespace(success=True, message_id=message_id)

    async def _edit_message(chat_id, message_id, content):
        return SimpleNamespace(success=False, error="message is not modified")

    adapter = SimpleNamespace(
        send=AsyncMock(side_effect=_send),
        edit_message=AsyncMock(side_effect=_edit_message),
        send_typing=AsyncMock(return_value=None),
        delete_message=AsyncMock(return_value=True),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123",
        chat_id="456",
        user_name="tester",
    )
    reasoning_queue = asyncio.Queue()
    await reasoning_queue.put("first thought")
    await reasoning_queue.put("second thought")

    async def _reasoning_worker():
        latest_text = ""
        reasoning_msg_id = None
        can_edit = True
        try:
            while True:
                try:
                    latest_text = await asyncio.wait_for(reasoning_queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    continue

                while True:
                    try:
                        latest_text = reasoning_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                payload = runner._format_reasoning_preview(latest_text)
                if can_edit and reasoning_msg_id is not None:
                    result = await adapter.edit_message(
                        chat_id=source.chat_id,
                        message_id=reasoning_msg_id,
                        content=payload,
                    )
                    if not result.success:
                        can_edit = False
                        await runner._delete_preview_message(adapter, source, reasoning_msg_id)
                        reasoning_msg_id = None

                if reasoning_msg_id is None:
                    result = await adapter.send(
                        chat_id=source.chat_id,
                        content=payload,
                        metadata=None,
                    )
                    if result.success and result.message_id:
                        reasoning_msg_id = result.message_id

                await asyncio.sleep(0)
        except asyncio.CancelledError:
            await runner._delete_preview_message(adapter, source, reasoning_msg_id)
            raise

    task = asyncio.create_task(_reasoning_worker())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert adapter.send.await_count == 2
    assert adapter.delete_message.await_count == 2
    adapter.delete_message.assert_any_await(chat_id="456", message_id="preview-1")
    adapter.delete_message.assert_any_await(chat_id="456", message_id="preview-2")
