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
    """Verify FeishuAdapter._send_raw_message routes to topic on fallback."""

    @pytest.mark.asyncio
    async def test_anchorless_topic_fetches_last_message_and_uses_reply(self):
        """An anchorless topic send uses a real message anchor and reply API."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        mock_client = MagicMock()
        mock_reply_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="new_msg_1"),
        )
        mock_client.im.v1.message.reply = MagicMock(return_value=mock_reply_response)

        adapter._client = mock_client
        adapter._build_reply_message_body = FeishuAdapter._build_reply_message_body
        adapter._build_reply_message_request = FeishuAdapter._build_reply_message_request
        adapter._fetch_last_message_in_thread = AsyncMock(return_value="om_last_in_topic")

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

        adapter._fetch_last_message_in_thread.assert_awaited_once_with("omt_topic_abc")
        mock_client.im.v1.message.reply.assert_called_once()
        mock_client.im.v1.message.create.assert_not_called()

        request = mock_client.im.v1.message.reply.call_args.args[0]
        assert getattr(request, "message_id", None) == "om_last_in_topic"
        body = getattr(request, "body", None) or getattr(request, "request_body", None)
        assert body is not None, "request has neither .body nor .request_body"
        assert getattr(body, "reply_in_thread", None) is True

    @pytest.mark.asyncio
    async def test_anchorless_topic_without_message_falls_back_to_chat(self):
        """If a topic has no usable anchor, deliver to the chat instead of dropping."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        mock_client = MagicMock()
        mock_client.im.v1.message.create = MagicMock(
            return_value=SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(message_id="new_msg_1"),
            )
        )

        adapter._client = mock_client
        adapter._build_create_message_body = FeishuAdapter._build_create_message_body
        adapter._build_create_message_request = FeishuAdapter._build_create_message_request
        adapter._fetch_last_message_in_thread = AsyncMock(return_value=None)

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

        adapter._fetch_last_message_in_thread.assert_awaited_once_with("omt_topic_abc")
        mock_client.im.v1.message.reply.assert_not_called()
        mock_client.im.v1.message.create.assert_called_once()

        request = mock_client.im.v1.message.create.call_args.args[0]
        assert getattr(request, "receive_id_type", None) == "chat_id"
        body = getattr(request, "body", None) or getattr(request, "request_body", None)
        assert body is not None, "request has neither .body nor .request_body"
        assert getattr(body, "receive_id", None) == "oc_main_chat"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [230011, 231003, 99992402])
    async def test_rejected_topic_reply_falls_back_to_chat(self, code):
        """A rejected topic reply still reaches the parent chat."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        mock_client = MagicMock()
        mock_client.im.v1.message.reply = MagicMock(
            return_value=SimpleNamespace(success=lambda: False, code=code)
        )
        mock_client.im.v1.message.create = MagicMock(
            return_value=SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(message_id="new_msg_1"),
            )
        )

        adapter._client = mock_client
        adapter._build_reply_message_body = FeishuAdapter._build_reply_message_body
        adapter._build_reply_message_request = FeishuAdapter._build_reply_message_request
        adapter._build_create_message_body = FeishuAdapter._build_create_message_body
        adapter._build_create_message_request = FeishuAdapter._build_create_message_request
        adapter._fetch_last_message_in_thread = AsyncMock(return_value="om_last_in_topic")
        adapter._response_succeeded = FeishuAdapter._response_succeeded

        async def _run_blocking_passthrough(func, *args):
            return func(*args)

        adapter._run_blocking = _run_blocking_passthrough

        import json
        response = await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        assert response.success()
        mock_client.im.v1.message.reply.assert_called_once()
        mock_client.im.v1.message.create.assert_called_once()
        request = mock_client.im.v1.message.create.call_args.args[0]
        assert getattr(request, "receive_id_type", None) == "chat_id"


class TestFallbackResendThreading:
    """#103068: a full fallback resend replaces the preview, so it must land in
    the originating thread; tail continuations keep their unthreaded delivery."""

    @pytest.mark.asyncio
    async def test_full_fallback_resend_threads_first_chunk_only(self):
        adapter = _make_adapter(max_length=700)
        adapter.send.side_effect = [
            SimpleNamespace(success=True, message_id=f"full_{i}") for i in range(10)
        ]
        consumer = GatewayStreamConsumer(adapter, "chat_123", initial_reply_to_id="om_user_1")
        consumer._message_id = "om_preview"
        consumer._last_sent_text = "truncated snapshot"
        consumer._already_sent = True
        consumer._fallback_final_send = True

        final = " ".join(["word"] * 400)  # not prefixed by the snapshot -> full resend
        await consumer._send_fallback_final(final)

        calls = adapter.send.await_args_list
        assert len(calls) > 1
        assert calls[0].kwargs["reply_to"] == "om_user_1"
        assert all("reply_to" not in c.kwargs for c in calls[1:])
