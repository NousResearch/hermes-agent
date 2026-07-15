"""Streaming obeys only the completed structured delivery decision."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter(*, supports_delete: bool = True) -> MagicMock:
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="preview_1")
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="preview_1")
    )
    if supports_delete:
        adapter.delete_message = AsyncMock(return_value=True)
    else:
        del adapter.delete_message  # type: ignore[attr-defined]
    return adapter


def _delivered_text(adapter) -> str:
    texts = [call.kwargs.get("content", "") for call in adapter.send.call_args_list]
    texts.extend(
        call.kwargs.get("content", "")
        for call in adapter.edit_message.call_args_list
    )
    return "".join(texts)


@pytest.mark.asyncio
async def test_text_that_looks_like_an_old_marker_streams_normally():
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_1",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
    )
    consumer.on_delta("NO")
    consumer.on_delta("_REPLY")
    consumer.finish()
    await consumer.run()

    assert "NO_REPLY" in _delivered_text(adapter)
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_structured_suppress_retracts_streamed_preview():
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_1",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
    )
    consumer._message_id = "preview_1"
    consumer._preview_message_ids = {"preview_1"}
    consumer._already_sent = True
    consumer.on_delta("Nothing changed in this run.")
    consumer.finish(suppress_delivery=True)
    await consumer.run()

    adapter.delete_message.assert_awaited_once_with("chat_1", "preview_1")
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False
    assert consumer.already_sent is False


@pytest.mark.asyncio
async def test_structured_deliver_does_not_hold_partial_text():
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_1",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
    )
    consumer.on_delta("NO")
    consumer.on_delta(" report is missing; here is the result.")
    consumer.finish(suppress_delivery=False)
    await consumer.run()

    assert "here is the result" in _delivered_text(adapter)
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_structured_suppress_without_delete_support_is_fail_safe():
    adapter = _make_adapter(supports_delete=False)
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_1",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
    )
    consumer.on_delta("Nothing worth delivering.")
    consumer.finish(suppress_delivery=True)
    await consumer.run()

    assert consumer.final_content_delivered is False
    assert consumer.already_sent is False
