"""Tests for Telegram native reply branching."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.session import SessionSource, build_session_key
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _ensure_telegram_mock():
    import sys

    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def test_build_session_key_includes_dm_reply_branch():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        reply_branch_id="branch-1",
    )

    assert build_session_key(source) == "agent:main:telegram:dm:123:branch-1"


def test_build_message_event_uses_stable_reply_branch_root():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = SimpleNamespace(id=42)
    adapter._remember_reply_branch("123", "777", "555")

    reply_target = SimpleNamespace(
        message_id=777,
        text="chunk 2",
        caption=None,
        from_user=SimpleNamespace(id=42),
    )
    message = SimpleNamespace(
        chat=SimpleNamespace(id=123, type="private", title=None),
        from_user=SimpleNamespace(id=7, full_name="Alice"),
        message_thread_id=None,
        reply_to_message=reply_target,
        text="follow up",
        caption=None,
        message_id=888,
        date=datetime.now(),
    )

    event = adapter._build_message_event(message, MessageType.TEXT)

    assert event.reply_to_message_id == "777"
    assert event.source.reply_branch_id == "555"


@pytest.mark.asyncio
async def test_send_forced_reply_chain_threads_every_chunk():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(
        side_effect=[
            SimpleNamespace(message_id=101),
            SimpleNamespace(message_id=102),
            SimpleNamespace(message_id=103),
        ]
    )
    adapter.truncate_message = lambda content, max_len, **kw: ["chunk1", "chunk2", "chunk3"]

    await adapter.send(
        "12345",
        "test content",
        reply_to="999",
        metadata={"reply_branch_id": "555", "force_reply_chain": True},
    )

    calls = adapter._bot.send_message.call_args_list
    assert len(calls) == 3
    assert [call.kwargs.get("reply_to_message_id") for call in calls] == [999, 999, 999]
    assert adapter._resolve_reply_branch("12345", "101") == "555"
    assert adapter._resolve_reply_branch("12345", "102") == "555"
    assert adapter._resolve_reply_branch("12345", "103") == "555"


@pytest.mark.asyncio
async def test_stream_consumer_keeps_all_chunks_in_same_reply_chain():
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 610
    adapter.truncate_message = lambda content, max_len, **kw: ["chunk 1", "chunk 2"]
    adapter.send = AsyncMock(
        side_effect=[
            SimpleNamespace(success=True, message_id="msg_1"),
            SimpleNamespace(success=True, message_id="msg_2"),
        ]
    )

    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        reply_to="incoming_42",
    )
    consumer.on_delta("x" * 1000)
    consumer.finish()

    await consumer.run()

    calls = adapter.send.call_args_list
    assert len(calls) == 2
    assert [call.kwargs.get("reply_to") for call in calls] == ["incoming_42", "incoming_42"]
