"""Post + thread_id messages rejected with 99992402 retry without thread routing.

Feishu rejects the ``receive_id_type=thread_id`` create path for post
payloads (error 99992402). The adapter must strip thread routing and retry
so the message lands in the main chat instead of being dropped. Mirrors the
existing audio fallback in ``_send_uploaded_file_message``.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def _make_adapter() -> FeishuAdapter:
    config = PlatformConfig(enabled=True)
    adapter = FeishuAdapter(config)
    adapter._client = MagicMock()
    return adapter


def _ok_response(message_id: str = "msg_001") -> SimpleNamespace:
    return SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(message_id=message_id),
        msg="",
        code=0,
    )


class TestSendMessagePostThreadFallback:
    """send_message strips thread routing on 99992402 for post messages."""

    @pytest.mark.asyncio
    async def test_exception_path_strips_thread_id_and_retries(self):
        adapter = _make_adapter()
        api_error = Exception("Feishu API error: 99992402")
        ok = _ok_response("msg_ok")

        with patch.object(
            adapter,
            "_feishu_send_with_retry",
            new_callable=AsyncMock,
            side_effect=[api_error, ok],
        ) as mock_send:
            result = await adapter.send(
                "oc_1", "**bold** content", metadata={"thread_id": "om_thread"}
            )

        assert result.success is True
        assert result.message_id == "msg_ok"
        assert mock_send.await_count == 2

        first, second = mock_send.await_args_list
        # First attempt kept thread routing; retry dropped it.
        assert first.kwargs["metadata"].get("thread_id") == "om_thread"
        assert second.kwargs["metadata"] == {}
        assert second.kwargs["msg_type"] == "post"
        assert second.kwargs["reply_to"] is None

    @pytest.mark.asyncio
    async def test_response_path_strips_thread_id_and_retries(self):
        adapter = _make_adapter()
        rejected = SimpleNamespace(
            success=lambda: False,
            msg="create message fail",
            code=99992402,
            data=None,
        )
        ok = _ok_response("msg_ok2")

        with patch.object(
            adapter,
            "_feishu_send_with_retry",
            new_callable=AsyncMock,
            side_effect=[rejected, ok],
        ) as mock_send:
            result = await adapter.send(
                "oc_1", "**bold** content", metadata={"thread_id": "om_thread"}
            )

        assert result.success is True
        assert result.message_id == "msg_ok2"
        assert mock_send.await_count == 2

        first, second = mock_send.await_args_list
        assert first.kwargs["metadata"].get("thread_id") == "om_thread"
        assert second.kwargs["metadata"] == {}
        assert second.kwargs["msg_type"] == "post"
        assert second.kwargs["reply_to"] is None

    @pytest.mark.asyncio
    async def test_non_thread_post_error_still_raises(self):
        """Unrelated post errors propagate unchanged (no fallback).

        ``send()`` catches exceptions and returns ``SendResult(success=False,
        error=...)``; assert the error surfaces and only one attempt was made
        (no thread-stripping retry).
        """
        adapter = _make_adapter()
        api_error = Exception("Feishu API error: 99992402")
        with patch.object(
            adapter,
            "_feishu_send_with_retry",
            new_callable=AsyncMock,
            side_effect=[api_error],
        ) as mock_send:
            result = await adapter.send("oc_1", "**bold** content")

        assert result.success is False
        assert "99992402" in result.error
        assert mock_send.await_count == 1

    @pytest.mark.asyncio
    async def test_plain_text_with_thread_id_unaffected(self):
        """Text messages keep thread routing (only post payloads are rejected)."""
        adapter = _make_adapter()
        ok = _ok_response("msg_txt")
        with patch.object(
            adapter,
            "_feishu_send_with_retry",
            new_callable=AsyncMock,
            return_value=ok,
        ) as mock_send:
            result = await adapter.send(
                "oc_1", "plain text", metadata={"thread_id": "om_thread"}
            )

        assert result.success is True
        assert mock_send.await_count == 1
        assert mock_send.await_args.kwargs["msg_type"] == "text"
        assert mock_send.await_args.kwargs["metadata"].get("thread_id") == "om_thread"
