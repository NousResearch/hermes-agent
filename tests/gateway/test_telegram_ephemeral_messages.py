"""Telegram Bot API 10.2 ephemeral group interaction tests.

These tests define the fail-closed contract before the adapter implements it:
configured group commands are registered/ingested as ephemeral, private replies
use raw Bot API 10.2 fields, and no public legacy send occurs on failure.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType, _thread_metadata_for_source
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter
from telegram.error import BadRequest


def _adapter(extra=None):
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"ephemeral_group_commands": ["status", "help"], **(extra or {})},
    )
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(
        return_value={"message_id": 0, "ephemeral_message_id": "out-eph-1"}
    )
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=999))
    bot.send_chat_action = AsyncMock()
    bot.set_my_commands = AsyncMock()
    adapter._bot = bot
    return adapter


def _message(*, text="/status", chat_type="supergroup", api_kwargs=None):
    return SimpleNamespace(
        text=text,
        message_id=0,
        api_kwargs=api_kwargs or {"ephemeral_message_id": "in-eph-1"},
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=-100123, type=chat_type),
    )


def _ephemeral_source():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_type="group",
        user_id="42",
    )
    source.delivery_metadata = {
        "telegram_ephemeral_required": True,
        "telegram_ephemeral_receiver_user_id": 42,
        "telegram_ephemeral_message_id": "in-eph-1",
        "suppress_streaming": True,
        "suppress_typing": True,
    }
    return source


def test_configured_group_command_extracts_ephemeral_delivery_metadata():
    adapter = _adapter()

    metadata = adapter._ephemeral_metadata_for_message(_message())

    assert metadata["telegram_ephemeral_required"] is True
    assert metadata["telegram_ephemeral_receiver_user_id"] == 42
    assert metadata["telegram_ephemeral_message_id"] == "in-eph-1"
    assert metadata["suppress_streaming"] is True
    assert metadata["suppress_typing"] is True
    assert time.monotonic() - metadata["telegram_ephemeral_received_at_monotonic"] < 1


def test_build_message_event_preserves_private_delivery_context():
    adapter = _adapter()
    message = _message()
    message.from_user.full_name = "Alice"
    message.chat.title = "Test group"
    message.chat.is_forum = False
    message.caption = None
    message.message_thread_id = None
    message.is_topic_message = False
    message.reply_to_message = None
    message.date = None

    event = adapter._build_message_event(message, MessageType.TEXT)

    assert event.metadata["telegram_ephemeral_required"] is True
    assert event.source.delivery_metadata["telegram_ephemeral_receiver_user_id"] == 42
    assert event.source.delivery_metadata["telegram_ephemeral_message_id"] == "in-eph-1"


@pytest.mark.asyncio
async def test_background_ephemeral_command_never_starts_typing_indicator():
    adapter = _adapter()
    message = _message()
    message.from_user.full_name = "Alice"
    message.chat.title = "Test group"
    message.chat.is_forum = False
    message.caption = None
    message.message_thread_id = None
    message.is_topic_message = False
    message.reply_to_message = None
    message.date = None
    event = adapter._build_message_event(message, MessageType.TEXT)

    async def handler(_event):
        return "private status"

    adapter.set_message_handler(handler)
    adapter._send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id=None)
    )
    keep_typing = AsyncMock()

    with patch.object(adapter, "_keep_typing", new=keep_typing):
        await adapter._process_message_background(event, "telegram:group:-100123")

    keep_typing.assert_not_awaited()
    metadata = adapter._send_with_retry.await_args.kwargs["metadata"]
    assert metadata["telegram_ephemeral_receiver_user_id"] == 42


def test_unconfigured_or_private_commands_are_not_marked_ephemeral():
    adapter = _adapter()

    assert adapter._ephemeral_metadata_for_message(_message(text="/new")) == {}
    assert adapter._ephemeral_metadata_for_message(_message(chat_type="private")) == {}


def test_ephemeral_source_metadata_reaches_base_and_runner_send_paths():
    source = _ephemeral_source()

    base_metadata = _thread_metadata_for_source(source)
    runner = object.__new__(GatewayRunner)
    runner_metadata = runner._thread_metadata_for_source(source)

    for metadata in (base_metadata, runner_metadata):
        assert metadata["telegram_ephemeral_required"] is True
        assert metadata["telegram_ephemeral_receiver_user_id"] == 42
        assert metadata["telegram_ephemeral_message_id"] == "in-eph-1"


def test_ephemeral_source_disables_streaming_to_avoid_public_preview():
    runner = object.__new__(GatewayRunner)

    assert runner._streaming_allowed_for_source(_ephemeral_source(), True) is False
    assert runner._streaming_allowed_for_source(_ephemeral_source(), False) is False
    assert (
        runner._streaming_allowed_for_source(
            SessionSource(platform=Platform.TELEGRAM, chat_id="1"), True
        )
        is True
    )


def test_group_command_objects_mark_only_configured_commands_ephemeral():
    adapter = _adapter()

    with patch("telegram.BotCommand") as command_cls:
        command_cls.side_effect = lambda name, description, api_kwargs=None: (
            SimpleNamespace(
                command=name,
                description=description,
                api_kwargs=api_kwargs or {},
            )
        )
        commands = adapter._make_menu_bot_commands(
            [("status", "Show status"), ("new", "New session")],
            group_scope=True,
        )

    assert commands[0].api_kwargs["is_ephemeral"] is True
    assert "is_ephemeral" not in commands[1].api_kwargs


@pytest.mark.asyncio
async def test_ephemeral_text_reply_uses_botapi_10_2_fields_and_not_public_send():
    adapter = _adapter()
    metadata = {
        "telegram_ephemeral_required": True,
        "telegram_ephemeral_receiver_user_id": 42,
        "telegram_ephemeral_message_id": "in-eph-1",
        "notify": True,
    }

    result = await adapter.send("-100123", "**Private status**", metadata=metadata)

    assert result.success is True
    assert result.message_id is None
    adapter._bot.send_message.assert_not_called()
    adapter._bot.do_api_request.assert_awaited_once()
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendMessage"
    payload = call.kwargs["api_kwargs"]
    assert payload["chat_id"] == -100123
    assert payload["receiver_user_id"] == 42
    assert payload["reply_parameters"] == {"ephemeral_message_id": "in-eph-1"}
    assert payload["parse_mode"] == "MarkdownV2"
    assert payload.get("disable_notification", False) is False


def test_expired_reply_window_drops_anchor_but_keeps_admin_private_target():
    adapter = _adapter()
    kwargs = adapter._ephemeral_send_api_kwargs({
        "telegram_ephemeral_required": True,
        "telegram_ephemeral_receiver_user_id": 42,
        "telegram_ephemeral_message_id": "in-eph-1",
        "telegram_ephemeral_received_at_monotonic": time.monotonic() - 16,
    })

    assert kwargs == {"receiver_user_id": 42}
    assert (
        adapter._ephemeral_send_api_kwargs({
            "telegram_ephemeral_required": True,
            "telegram_ephemeral_receiver_user_id": "not-a-user-id",
        })
        is None
    )


@pytest.mark.asyncio
async def test_ephemeral_markdown_fallback_remains_private():
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(
        side_effect=[BadRequest("can't parse entities"), {"message_id": 0}]
    )
    metadata = {
        "telegram_ephemeral_required": True,
        "telegram_ephemeral_receiver_user_id": 42,
        "telegram_ephemeral_message_id": "in-eph-1",
    }

    result = await adapter.send("-100123", "broken _ markdown", metadata=metadata)

    assert result.success is True
    assert adapter._bot.do_api_request.await_count == 2
    retry_payload = adapter._bot.do_api_request.await_args_list[1].kwargs["api_kwargs"]
    assert retry_payload["receiver_user_id"] == 42
    assert retry_payload["reply_parameters"] == {"ephemeral_message_id": "in-eph-1"}
    assert "parse_mode" not in retry_payload
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_ephemeral_text_failure_is_fail_closed_without_public_fallback():
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("not eligible"))
    metadata = {
        "telegram_ephemeral_required": True,
        "telegram_ephemeral_receiver_user_id": 42,
        "telegram_ephemeral_message_id": "in-eph-1",
    }

    result = await adapter.send("-100123", "secret", metadata=metadata)

    assert result.success is False
    assert result.retryable is False
    assert "ephemeral" in (result.error or "").lower()
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_ephemeral_image_uses_private_api_kwargs(tmp_path):
    adapter = _adapter()
    adapter._bot.send_photo = AsyncMock(
        return_value=SimpleNamespace(
            message_id=0, api_kwargs={"ephemeral_message_id": "out-eph-img"}
        )
    )
    image = tmp_path / "chart.png"
    image.write_bytes(b"not-a-real-png-but-send-is-mocked")
    metadata = {
        "telegram_ephemeral_required": True,
        "telegram_ephemeral_receiver_user_id": 42,
        "telegram_ephemeral_message_id": "in-eph-1",
    }

    result = await adapter.send_image_file("-100123", str(image), metadata=metadata)

    assert result.success is True
    kwargs = adapter._bot.send_photo.call_args.kwargs
    assert kwargs["api_kwargs"]["receiver_user_id"] == 42
    assert kwargs["api_kwargs"]["reply_parameters"] == {
        "ephemeral_message_id": "in-eph-1"
    }
    assert kwargs["reply_to_message_id"] is None


@pytest.mark.asyncio
async def test_ephemeral_media_without_receiver_fails_closed(tmp_path):
    adapter = _adapter()
    adapter._bot.send_photo = AsyncMock()
    adapter._bot.send_document = AsyncMock()
    image = tmp_path / "chart.png"
    image.write_bytes(b"png")

    result = await adapter.send_image_file(
        "-100123",
        str(image),
        metadata={"telegram_ephemeral_required": True},
    )

    assert result.success is False
    adapter._bot.send_photo.assert_not_called()
    adapter._bot.send_document.assert_not_called()
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_ephemeral_image_batch_avoids_public_send_media_group(tmp_path):
    adapter = _adapter()
    adapter.send_image_file = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter._bot.send_media_group = AsyncMock()
    p1 = tmp_path / "one.png"
    p2 = tmp_path / "two.png"
    p1.write_bytes(b"1")
    p2.write_bytes(b"2")
    metadata = {
        "telegram_ephemeral_required": True,
        "telegram_ephemeral_receiver_user_id": 42,
        "telegram_ephemeral_message_id": "in-eph-1",
    }

    await adapter.send_multiple_images(
        "-100123",
        [(f"file://{p1}", "one"), (f"file://{p2}", "two")],
        metadata=metadata,
    )

    adapter._bot.send_media_group.assert_not_called()
    assert adapter.send_image_file.await_count == 2
    for call in adapter.send_image_file.await_args_list:
        assert call.kwargs["metadata"] is metadata
