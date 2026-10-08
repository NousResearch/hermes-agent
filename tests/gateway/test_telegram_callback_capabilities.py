"""Security invariants for Telegram inline callback request capabilities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Button:
    def __init__(self, text, callback_data):
        self.text = text
        self.callback_data = callback_data


class _Markup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _make_adapter(monkeypatch):
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Markup)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = MagicMock()
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True
    return adapter


def test_gateway_callback_metadata_carries_telegram_requester():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        thread_id="77",
        message_id="41",
        user_id="11",
        profile=None,
    )

    threaded = runner._thread_metadata_for_source(source, "41")
    assert threaded["requester_user_id"] == "11"

    source.thread_id = None
    unthreaded = runner._thread_metadata_for_progress(source, "41", None, None)
    assert unthreaded["requester_user_id"] == "11"


@pytest.mark.asyncio
async def test_callback_capability_denies_stale_foreign_misbound_and_replayed_taps(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    sent = []

    async def send_message(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(message_id=100 + len(sent), message_thread_id=77)

    adapter._bot = SimpleNamespace(send_message=send_message)
    first_selected = AsyncMock(return_value="first")
    second_selected = AsyncMock(return_value="second")
    metadata = {"thread_id": "77", "requester_user_id": "11"}

    await adapter.send_choice_picker(
        "-100123", "First", [{"value": "old", "label": "Old"}], "session",
        first_selected, metadata=metadata,
    )
    first_data = sent[-1]["reply_markup"].inline_keyboard[0][0].callback_data
    await adapter.send_choice_picker(
        "-100123", "Second", [{"value": "new", "label": "New"}], "session",
        second_selected, metadata=metadata,
    )
    second_data = sent[-1]["reply_markup"].inline_keyboard[0][0].callback_data

    assert first_data.startswith("rq:")
    assert second_data.startswith("rq:")
    assert first_data != second_data

    async def tap(data, *, actor="11", chat_id=-100123, thread_id=77, message_id=102):
        query = SimpleNamespace(
            data=data,
            from_user=SimpleNamespace(id=actor, first_name="Tapper"),
            message=SimpleNamespace(
                chat_id=chat_id,
                chat=SimpleNamespace(type="supergroup"),
                message_thread_id=thread_id,
                message_id=message_id,
            ),
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
        return query

    denied = [
        await tap(first_data, message_id=101),  # stale generation
        await tap(second_data, actor="22"),  # foreign actor
        await tap(second_data, chat_id=-100999),  # foreign chat
        await tap(second_data, thread_id=88),  # foreign thread
        await tap(second_data, message_id=999),  # foreign message
        await tap("rq:not-issued:0"),  # unknown capability
    ]
    assert all(query.answer.await_count == 1 for query in denied)
    first_selected.assert_not_awaited()
    second_selected.assert_not_awaited()

    valid = await tap(second_data)
    valid.edit_message_text.assert_awaited_once()
    second_selected.assert_awaited_once_with("-100123", "new")

    replay = await tap(second_data)
    assert replay.answer.await_count == 1
    second_selected.assert_awaited_once()
