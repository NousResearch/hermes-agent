"""Telegram secure-message delivery tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


@pytest.fixture
def telegram_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    bot.edit_message_text = AsyncMock(return_value=True)
    bot.delete_message = AsyncMock(return_value=True)
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot
    adapter._should_attempt_rich = lambda *args, **kwargs: False
    adapter._schedule_ephemeral_delete = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_secure_message_metadata_uses_spoiler_protect_content_and_ttl(telegram_adapter):
    result = await telegram_adapter.send(
        "12345",
        "sensitive account note",
        metadata={"secure_message": True, "secure_message_ttl_seconds": 30},
    )

    assert result.success is True
    kwargs = telegram_adapter._bot.send_message.await_args.kwargs
    assert kwargs["protect_content"] is True
    assert kwargs["text"].startswith("||")
    assert kwargs["text"].endswith("||")
    assert "sensitive account note" in kwargs["text"]
    telegram_adapter._schedule_ephemeral_delete.assert_called_once_with(
        chat_id="12345",
        message_id="42",
        ttl_seconds=30,
    )


@pytest.mark.asyncio
async def test_secure_message_marker_is_stripped_and_enables_secure_send(telegram_adapter):
    await telegram_adapter.send("12345", "[[secure_message]]classified-ish")

    kwargs = telegram_adapter._bot.send_message.await_args.kwargs
    assert "[[secure_message]]" not in kwargs["text"]
    assert kwargs["protect_content"] is True
    assert kwargs["text"].startswith("||")
    assert kwargs["text"].endswith("||")
    assert "classified" in kwargs["text"]


@pytest.mark.asyncio
async def test_secure_message_options_can_disable_spoiler_and_protect_content(telegram_adapter):
    await telegram_adapter.send(
        "12345",
        "visible but temporary",
        metadata={
            "secure_message": True,
            "secure_message_spoiler": False,
            "secure_message_protect_content": False,
        },
    )

    kwargs = telegram_adapter._bot.send_message.await_args.kwargs
    assert "protect_content" not in kwargs
    assert not kwargs["text"].startswith("||")
    assert kwargs["text"] == "visible but temporary"
    telegram_adapter._schedule_ephemeral_delete.assert_not_called()


@pytest.mark.asyncio
async def test_secure_message_edit_keeps_streaming_updates_spoilered(telegram_adapter):
    result = await telegram_adapter.edit_message(
        "12345",
        "77",
        "[[secure_message]]streamed secret",
        finalize=True,
        metadata={"secure_message_ttl_seconds": 20},
    )

    kwargs = telegram_adapter._bot.send_message.await_args.kwargs
    assert "[[secure_message]]" not in kwargs["text"]
    assert kwargs["protect_content"] is True
    assert kwargs["text"].startswith("||")
    assert kwargs["text"].endswith("||")
    telegram_adapter._bot.delete_message.assert_awaited_once_with(
        chat_id=12345,
        message_id=77,
    )
    assert result.message_id == "42"
    assert result.continuation_message_ids == ("42",)
    telegram_adapter._schedule_ephemeral_delete.assert_called_once_with(
        chat_id="12345",
        message_id="42",
        ttl_seconds=20,
    )


@pytest.mark.asyncio
async def test_secure_message_metadata_edit_fresh_sends_protected_replacement(telegram_adapter):
    result = await telegram_adapter.edit_message(
        "12345",
        "77",
        "metadata secure edit",
        finalize=True,
        metadata={"secure_message": True, "secure_message_ttl_seconds": 25},
    )

    assert result.success is True
    assert result.message_id == "42"
    assert result.continuation_message_ids == ("42",)
    kwargs = telegram_adapter._bot.send_message.await_args.kwargs
    assert kwargs["protect_content"] is True
    assert kwargs["text"].startswith("||")
    telegram_adapter._bot.edit_message_text.assert_not_awaited()
    telegram_adapter._bot.delete_message.assert_awaited_once_with(
        chat_id=12345,
        message_id=77,
    )
    telegram_adapter._schedule_ephemeral_delete.assert_called_once_with(
        chat_id="12345",
        message_id="42",
        ttl_seconds=25,
    )


@pytest.mark.asyncio
async def test_secure_message_edit_without_protect_can_edit_in_place(telegram_adapter):
    result = await telegram_adapter.edit_message(
        "12345",
        "77",
        "spoiler only",
        finalize=True,
        metadata={
            "secure_message": True,
            "secure_message_protect_content": False,
            "secure_message_ttl_seconds": 15,
        },
    )

    assert result.success is True
    telegram_adapter._bot.send_message.assert_not_awaited()
    telegram_adapter._bot.edit_message_text.assert_awaited_once()
    kwargs = telegram_adapter._bot.edit_message_text.await_args.kwargs
    assert kwargs["text"].startswith("||")
    telegram_adapter._schedule_ephemeral_delete.assert_called_once_with(
        chat_id="12345",
        message_id="77",
        ttl_seconds=15,
    )


@pytest.mark.asyncio
async def test_secure_overflow_continuations_keep_protection_and_ttl(telegram_adapter):
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[SimpleNamespace(message_id=idx) for idx in range(101, 111)]
    )
    telegram_adapter._schedule_ephemeral_delete = MagicMock()
    object.__setattr__(telegram_adapter, "MAX_MESSAGE_LENGTH", 80)

    result = await telegram_adapter._edit_overflow_split(
        "12345",
        "77",
        "secret " * 40,
        finalize=True,
        metadata={"secure_message": True, "secure_message_ttl_seconds": 45},
    )

    assert result.success is True
    for call in telegram_adapter._bot.send_message.await_args_list:
        assert call.kwargs["protect_content"] is True
        assert call.kwargs["text"].startswith("||")
        assert call.kwargs["text"].endswith("||")
    telegram_adapter._schedule_ephemeral_delete.assert_any_call(
        chat_id="12345",
        message_id="77",
        ttl_seconds=45,
    )
    telegram_adapter._schedule_ephemeral_delete.assert_any_call(
        chat_id="12345",
        message_id="101",
        ttl_seconds=45,
    )
