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

    @staticmethod
    def _topic_adapter(last_message_id, send_succeeds=True):
        """Real routing code over a mocked client; only the SDK calls are faked."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        mock_client = MagicMock()
        mock_client.im.v1.message.reply = MagicMock(return_value=SimpleNamespace(
            success=lambda: send_succeeds, code=0 if send_succeeds else 99992402,
            data=SimpleNamespace(message_id="new_msg_1"),
        ))
        mock_client.im.v1.message.create = MagicMock(return_value=SimpleNamespace(
            success=lambda: True, data=SimpleNamespace(message_id="new_msg_2"),
        ))
        adapter._client = mock_client
        adapter._build_create_message_body = FeishuAdapter._build_create_message_body
        adapter._build_create_message_request = FeishuAdapter._build_create_message_request
        adapter._build_reply_message_body = FeishuAdapter._build_reply_message_body
        adapter._build_reply_message_request = FeishuAdapter._build_reply_message_request
        adapter._response_succeeded = FeishuAdapter._response_succeeded
        adapter._fetch_last_message_in_thread = AsyncMock(return_value=last_message_id)

        async def _run_blocking_passthrough(func, *args):
            return func(*args)
        adapter._run_blocking = _run_blocking_passthrough
        return adapter, mock_client

    @pytest.mark.asyncio
    async def test_anchorless_topic_send_replies_into_the_thread(self):
        """message.create rejects receive_id_type='thread_id' with 99992402 for every
        msg_type, so a send with no reply anchor must go through the reply API."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, mock_client = self._topic_adapter("om_last_in_topic")
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload='{"text": "hello"}',
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        mock_client.im.v1.message.create.assert_not_called()
        mock_client.im.v1.message.reply.assert_called_once()
        call_args = mock_client.im.v1.message.reply.call_args[0][0]
        assert getattr(call_args, "message_id", None) == "om_last_in_topic"
        body = getattr(call_args, "request_body", None) or getattr(call_args, "body", None)
        assert getattr(body, "reply_in_thread", None) is True

    @pytest.mark.asyncio
    async def test_unresolvable_anchor_falls_back_to_the_chat(self):
        """No anchor (empty topic / lookup failure) must still deliver, to the chat."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, mock_client = self._topic_adapter(None)
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="image",
            payload='{"image_key": "img_v3_x"}',
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        mock_client.im.v1.message.reply.assert_not_called()
        mock_client.im.v1.message.create.assert_called_once()
        call_args = mock_client.im.v1.message.create.call_args[0][0]
        assert getattr(call_args, "receive_id_type", None) == "chat_id"
        body = getattr(call_args, "body", None) or getattr(call_args, "request_body", None)
        receive_id = getattr(body, "receive_id", None)
        if receive_id is None and isinstance(body, str):
            import json as _json
            receive_id = _json.loads(body).get("receive_id")
        assert receive_id == "oc_main_chat"

    @pytest.mark.asyncio
    async def test_failed_topic_reply_still_falls_back_to_the_chat(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, mock_client = self._topic_adapter("om_last_in_topic", send_succeeds=False)
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload='{"text": "hello"}',
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        mock_client.im.v1.message.reply.assert_called_once()
        mock_client.im.v1.message.create.assert_called_once()


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
