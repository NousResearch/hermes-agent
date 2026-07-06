"""Regression tests for stream consumer thread/topic routing fix.

Verifies that GatewayStreamConsumer correctly passes reply_to on the first
message send, ensuring messages land in the correct topic/thread instead of
the main group chat.

Covers: #6969, #9916, #7355
"""
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
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
        assert call_kwargs["metadata"] == {**metadata, "expect_edits": True}
        assert metadata == {"thread_id": "omt_topic789"}

    @pytest.mark.asyncio
    async def test_final_first_send_marks_metadata_notify_true(self):
        """Final streaming sends should use the existing notify=True marker."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "root_post_123"},
            initial_reply_to_id="reply_post_456",
        )

        await consumer._send_or_edit("Final answer", finalize=True)

        call_kwargs = adapter.send.call_args[1]
        metadata = call_kwargs["metadata"]
        assert metadata["thread_id"] == "root_post_123"
        assert metadata["notify"] is True
        assert "delivery_kind" not in metadata
        assert "allow_flat_fallback" not in metadata

    @pytest.mark.asyncio
    async def test_nonfinal_first_send_does_not_mark_notify(self):
        """Preview/interim streaming sends must not be notify-worthy."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "root_post_123"},
            initial_reply_to_id="reply_post_456",
        )

        await consumer._send_or_edit("Preview", finalize=False)

        metadata = adapter.send.call_args[1]["metadata"]
        assert metadata == {"thread_id": "root_post_123", "expect_edits": True}


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


class TestFeishuFallbackThreadRouting:
    """Verify FeishuAdapter._send_raw_message routes thread sends via reply API.

    Feishu's create-message API only accepts receive_id_type in
    {open_id, union_id, user_id, email, chat_id} — there is NO ``thread_id``
    receive_id_type, so a topic message can only land in a topic through the
    reply API (``reply_in_thread=true``) against a real ``om_`` message id.
    These tests assert that contract: a thread send with an anchor uses the
    reply API, and a thread send with no anchor falls back to a top-level
    chat create (never an invalid ``receive_id_type=thread_id``).
    """

    @pytest.mark.asyncio
    async def test_thread_send_with_anchor_uses_reply_api(self):
        """When reply_to is set and metadata has thread_id, the reply API is
        used with reply_in_thread=True so the message lands in the topic."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        mock_client = MagicMock()
        mock_reply_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="new_msg_1"),
        )
        mock_client.im.v1.message.reply = MagicMock(return_value=mock_reply_response)
        mock_client.im.v1.message.create = MagicMock()

        adapter = MagicMock(spec=FeishuAdapter)
        adapter._client = mock_client
        adapter._build_reply_message_body = FeishuAdapter._build_reply_message_body
        adapter._build_reply_message_request = FeishuAdapter._build_reply_message_request
        async def _run_blocking_passthrough(func, *args):
            return func(*args)
        adapter._run_blocking = _run_blocking_passthrough

        import json
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to="om_thread_root",
            metadata={"thread_id": "omt_topic_abc"},
        )

        # Reply API is the path that lands a message in a topic.
        mock_client.im.v1.message.reply.assert_called_once()
        mock_client.im.v1.message.create.assert_not_called()
        # request must target the supplied om_ message id.
        reply_request = mock_client.im.v1.message.reply.call_args[0][0]
        assert getattr(reply_request, "message_id", None) == "om_thread_root"

    @pytest.mark.asyncio
    async def test_thread_send_with_metadata_reply_to_uses_reply_api(self):
        """When reply_to is None but metadata carries reply_to_message_id
        (the Feishu status-metadata fallback), the reply API is still used."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        mock_client = MagicMock()
        mock_reply_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="new_msg_1"),
        )
        mock_client.im.v1.message.reply = MagicMock(return_value=mock_reply_response)
        mock_client.im.v1.message.create = MagicMock()

        adapter = MagicMock(spec=FeishuAdapter)
        adapter._client = mock_client
        adapter._build_reply_message_body = FeishuAdapter._build_reply_message_body
        adapter._build_reply_message_request = FeishuAdapter._build_reply_message_request
        async def _run_blocking_passthrough(func, *args):
            return func(*args)
        adapter._run_blocking = _run_blocking_passthrough

        import json
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={
                "thread_id": "omt_topic_abc",
                "reply_to_message_id": "om_thread_root",
            },
        )

        mock_client.im.v1.message.reply.assert_called_once()
        mock_client.im.v1.message.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_thread_send_without_anchor_falls_back_to_chat_create(self):
        """When reply_to is None and metadata has thread_id but no anchor,
        fall back to a top-level chat create — NOT an invalid
        receive_id_type=thread_id. The Feishu API rejects thread_id."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        mock_client = MagicMock()
        mock_create_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="new_msg_1"),
        )
        mock_client.im.v1.message.create = MagicMock(return_value=mock_create_response)
        mock_client.im.v1.message.reply = MagicMock()

        adapter = MagicMock(spec=FeishuAdapter)
        adapter._client = mock_client
        adapter._build_create_message_body = FeishuAdapter._build_create_message_body
        adapter._build_create_message_request = FeishuAdapter._build_create_message_request
        async def _run_blocking_passthrough(func, *args):
            return func(*args)
        adapter._run_blocking = _run_blocking_passthrough

        import json
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        mock_client.im.v1.message.create.assert_called_once()
        mock_client.im.v1.message.reply.assert_not_called()
        call_args = mock_client.im.v1.message.create.call_args[0][0]
        receive_id_type = getattr(call_args, "receive_id_type", None)
        assert receive_id_type != "thread_id", (
            f"receive_id_type must NOT be 'thread_id' (Feishu rejects it); "
            f"got '{receive_id_type}'"
        )
        assert receive_id_type == "chat_id", (
            f"Expected top-level fallback receive_id_type='chat_id', "
            f"got '{receive_id_type}'"
        )
        # receive_id must be the chat_id, never the omt_ thread id.
        body = getattr(call_args, "body", None) or getattr(call_args, "request_body", None)
        receive_id = getattr(body, "receive_id", None)
        if receive_id is None and isinstance(body, str):
            import json as _json
            receive_id = _json.loads(body).get("receive_id")
        assert receive_id == "oc_main_chat", (
            f"Expected receive_id='oc_main_chat', got '{receive_id}'"
        )

    @pytest.mark.asyncio
    async def test_create_uses_chat_id_when_no_thread(self):
        """When reply_to=None and metadata has no thread_id, message.create
        should use receive_id_type='chat_id' (original behavior)."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        mock_client = MagicMock()
        mock_create_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="new_msg_1"),
        )
        mock_client.im.v1.message.create = MagicMock(return_value=mock_create_response)

        adapter = MagicMock(spec=FeishuAdapter)
        adapter._client = mock_client
        adapter._build_create_message_body = FeishuAdapter._build_create_message_body
        adapter._build_create_message_request = FeishuAdapter._build_create_message_request
        async def _run_blocking_passthrough(func, *args):
            return func(*args)
        adapter._run_blocking = _run_blocking_passthrough

        import json
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata=None,
        )

        mock_client.im.v1.message.create.assert_called_once()
        call_args = mock_client.im.v1.message.create.call_args[0][0]
        receive_id_type = getattr(call_args, "receive_id_type", None)
        assert receive_id_type == "chat_id"
