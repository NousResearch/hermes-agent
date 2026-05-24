"""Regression tests for stream consumer thread/topic routing fix.

Verifies that GatewayStreamConsumer correctly passes reply_to on the first
message send, ensuring messages land in the correct topic/thread instead of
the main group chat.

Covers: #6969, #9916, #7355
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
    StreamConsumerConfig,
)


def _make_adapter(send_result=None, edit_result=None, max_length=4096):
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=send_result or SimpleNamespace(success=True, message_id="msg_1")
    )
    adapter.edit_message = AsyncMock(
        return_value=edit_result or SimpleNamespace(success=True)
    )
    adapter.MAX_MESSAGE_LENGTH = max_length
    return adapter


class TestInitialReplyToId:
    """Verify initial_reply_to_id is passed as reply_to on first send."""

    @pytest.mark.asyncio
    async def test_first_send_uses_initial_reply_to_id(self):
        """When initial_reply_to_id is set, first adapter.send() should
        include reply_to=initial_reply_to_id."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_456",
        )
        await consumer._send_or_edit("Hello world")

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["reply_to"] == "om_user_msg_456", (
            "First send should pass initial_reply_to_id as reply_to"
        )
        assert call_kwargs["chat_id"] == "chat_123"

    @pytest.mark.asyncio
    async def test_first_send_without_initial_reply_to_id(self):
        """When initial_reply_to_id is None, first send should have
        reply_to=None (backward compatible)."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
        )
        await consumer._send_or_edit("Hello world")

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs.get("reply_to") is None

    @pytest.mark.asyncio
    async def test_subsequent_edits_ignore_initial_reply_to_id(self):
        """After first send, edits should use message_id, not initial_reply_to_id."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_456",
        )

        # First send
        await consumer._send_or_edit("Hello world")
        assert adapter.send.call_count == 1

        # Second call should edit, not send
        await consumer._send_or_edit("Hello world updated")
        assert adapter.send.call_count == 1, "Should edit, not send again"
        adapter.edit_message.assert_called_once()
        edit_kwargs = adapter.edit_message.call_args[1]
        assert edit_kwargs["message_id"] == "msg_1"
        assert edit_kwargs["chat_id"] == "chat_123"

    @pytest.mark.asyncio
    async def test_metadata_passed_on_first_send(self):
        """Metadata (containing thread_id) should be forwarded on first send."""
        adapter = _make_adapter()
        metadata = {"thread_id": "omt_topic789"}
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata=metadata,
            initial_reply_to_id="om_msg_000",
        )
        await consumer._send_or_edit("Test")

        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["metadata"] == metadata


class TestOverflowFirstMessage:
    """Verify thread routing is preserved when the first message overflows."""

    @pytest.mark.asyncio
    async def test_overflow_first_send_uses_initial_reply_to_id(self):
        """When first message exceeds platform limit and is split into chunks,
        each chunk should be threaded to initial_reply_to_id, not None."""
        adapter = _make_adapter(max_length=10)
        adapter.truncate_message = MagicMock(
            return_value=["chunk_1", "chunk_2"]
        )
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_789",
        )

        # Inject oversized accumulated text to trigger overflow path
        consumer._accumulated = "A" * 100
        consumer._current_edit_interval = 999
        await consumer._send_new_chunk("chunk_1", consumer._message_id or consumer._initial_reply_to_id)

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["reply_to"] == "om_user_msg_789", (
            "Overflow first chunk should use initial_reply_to_id"
        )


class _FakeFeishuChannel:
    def __init__(self):
        self.calls = []

    async def send(self, target_id, payload, opts=None):
        self.calls.append((target_id, payload, opts))
        return SimpleNamespace(success=True, message_id="new_msg_1", error=None)


class TestFeishuSdkThreadRouting:
    """Verify FeishuAdapter routes new SDK sends to topics when available."""

    @pytest.mark.asyncio
    async def test_create_uses_thread_id_when_available(self):
        """When reply_to=None and metadata has thread_id, SDK sends to the topic."""
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "app_id": "cli_test",
                    "app_secret": "secret_test",
                    "domain": "feishu",
                },
            )
        )
        channel = _FakeFeishuChannel()
        adapter._channel = channel

        result = await adapter.send(
            chat_id="oc_main_chat",
            content="hello",
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        assert result.success is True
        assert channel.calls == [
            ("omt_topic_abc", "hello", {"receive_id_type": "thread_id"})
        ]

    @pytest.mark.asyncio
    async def test_create_uses_chat_id_when_no_thread(self):
        """When reply_to=None and metadata has no thread_id, SDK sends to the chat."""
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "app_id": "cli_test",
                    "app_secret": "secret_test",
                    "domain": "feishu",
                },
            )
        )
        channel = _FakeFeishuChannel()
        adapter._channel = channel

        result = await adapter.send(
            chat_id="oc_main_chat",
            content="hello",
            reply_to=None,
            metadata=None,
        )

        assert result.success is True
        assert channel.calls == [("oc_main_chat", "hello", None)]
