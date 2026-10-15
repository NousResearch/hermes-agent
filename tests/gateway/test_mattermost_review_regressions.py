"""Regressions for source-span and acknowledgement reconciliation in Mattermost."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _Tick
from plugins.platforms.mattermost.adapter import MattermostAdapter


def partial(source, prefix):
    return SendResult(success=False, message_id="head", error="rejected",
                      raw_response={"source_confirmed_prefix": prefix,
                                    "source_attempted_prefix": prefix,
                                    "message_ids": ["head"]})


def rejected():
    return SendResult(success=False, error="rejected", raw_response={
        "source_confirmed_prefix": "", "source_attempted_prefix": ""})


def consumer_with_outcomes(*outcomes):
    adapter = SimpleNamespace(send=AsyncMock(side_effect=outcomes),
                              edit_message=AsyncMock())
    consumer = GatewayStreamConsumer(adapter, "channel", config=StreamConsumerConfig(cursor=""))
    return consumer, adapter


@pytest.mark.asyncio
async def test_segment_break_after_confirmed_head_sends_only_rejected_tail():
    source = "confirmed head and a definite rejected tail"
    prefix = "confirmed head "
    consumer, adapter = consumer_with_outcomes(partial(source, prefix),
                                                SendResult(success=True, message_id="tail", raw_response={
                                                    "source_confirmed_prefix": source[len(prefix):],
                                                    "source_attempted_prefix": source[len(prefix):]}))
    consumer._accumulated = source
    assert not await consumer._send_or_edit(source, finalize=False)
    await consumer._end_segment(_Tick(got_segment_break=True))
    assert [call.kwargs["content"] for call in adapter.send.await_args_list] == [source, source[len(prefix):]]


@pytest.mark.asyncio
async def test_rejected_edit_keeps_previously_visible_prefix_at_segment_break():
    source = "Already shown unseen tail"
    consumer, adapter = consumer_with_outcomes(SendResult(success=True, message_id="tail"))
    adapter.edit_message.return_value = rejected()
    consumer._message_id = "preview"
    consumer._last_sent_text = "Already shown "
    consumer._accumulated = source
    assert not await consumer._send_or_edit(source, finalize=False)
    await consumer._end_segment(_Tick(got_segment_break=True))
    assert [call.kwargs["content"] for call in adapter.send.await_args_list] == ["unseen tail"]


@pytest.mark.asyncio
async def test_rejected_interim_tail_receipt_survives_boundary_until_final():
    source = "confirmed interim prefix and a still missing tail"
    prefix = "confirmed interim prefix "
    consumer, adapter = consumer_with_outcomes(partial(source, prefix), rejected(),
                                                SendResult(success=True, message_id="tail", raw_response={
                                                    "source_confirmed_prefix": source[len(prefix):],
                                                    "source_attempted_prefix": source[len(prefix):]}))
    consumer._accumulated = source
    assert not await consumer._send_or_edit(source, finalize=False)
    await consumer._end_segment(_Tick(got_segment_break=True))
    assert consumer.source_delivery_pending
    assert await consumer.reconcile_source_final(source)
    assert [call.kwargs["content"] for call in adapter.send.await_args_list] == [
        source, source[len(prefix):], source[len(prefix):]]


@pytest.mark.asyncio
async def test_definite_final_suffix_rejections_are_reported_without_replaying_head():
    source = "confirmed head followed by missing final tail"
    prefix = "confirmed head "
    consumer, adapter = consumer_with_outcomes(partial(source, prefix), rejected(), rejected())
    assert not await consumer._send_or_edit(source, finalize=False)
    assert await consumer.reconcile_source_final(source)
    assert consumer.source_delivery_error
    from gateway.run_turn import GatewayTurnMixin
    response = {"final_response": source}
    turn = SimpleNamespace(stream_consumer_holder=[consumer],
                           source=SimpleNamespace(platform="mattermost"), session_key="session")
    await GatewayTurnMixin()._run_agent_mark_streamed_delivery(response, turn)
    assert response.get("already_sent") is True  # do not replay confirmed head
    assert response.get("delivery_error") == consumer.source_delivery_error
    assert [call.kwargs["content"] for call in adapter.send.await_args_list] == [
        source, source[len(prefix):], source[len(prefix):]]


@pytest.mark.asyncio
async def test_thread_lookup_timeout_preserves_original_reply_anchor():
    adapter = MattermostAdapter(PlatformConfig(extra={"reply_mode": "thread"}))
    adapter._api_get = AsyncMock(side_effect=TimeoutError())
    payloads = []

    async def post(path, payload):
        payloads.append(payload)
        return {"id": "posted"}

    adapter._api_post = post
    result = await adapter.send("channel", "reply content", reply_to="original-post")
    assert result.success
    assert len(payloads) == 1 and payloads[0]["root_id"] == "original-post"


@pytest.mark.asyncio
async def test_later_image_batch_lost_ack_is_not_aggregate_success():
    adapter = MattermostAdapter(PlatformConfig(extra={"reply_mode": "off"}))
    adapter._load_batch_image = AsyncMock(return_value=(b"image", "image.png", "image/png"))
    adapter._upload_file = AsyncMock(side_effect=[f"file-{i}" for i in range(6)])
    posts = []

    async def post(chat_id, message, reply_to, metadata, file_ids=None):
        posts.append(list(file_ids))
        if len(posts) == 1:
            return {"id": "first"}
        return {"_delivery_uncertain": True, "content_attempted": True, "content_uncertain": True}

    adapter._post_message = post
    result = await adapter.send_multiple_images("channel", [("file://unused", "caption") for _ in range(6)])
    assert posts == [[f"file-{i}" for i in range(5)], ["file-5"]]
    assert not result.success and result.message_id == "first"
    assert result.raw_response["_delivery_uncertain"]


@pytest.mark.asyncio
async def test_confirmed_first_post_later_rejection_marks_partial_for_ledger():
    adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500, "reply_mode": "off"}))
    payloads = []

    async def post(path, payload):
        payloads.append(payload)
        return {"id": "first"} if len(payloads) == 1 else {"_post_rejection_status": 400}

    adapter._api_post = post
    source = "a" * 500 + "tail never acknowledged"
    result = await adapter.send("channel", source)
    assert not result.success and result.message_id == "first"
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["delivered_prefix"] == source[:500]
    assert result.raw_response["source_attempted_prefix"] == source[:500]
    assert [p["message"] for p in payloads] == [source[:500], source[500:]]


@pytest.mark.asyncio
async def test_suppressed_warning_does_not_block_flat_recovery_of_long_final():
    adapter = MattermostAdapter(PlatformConfig(extra={"reply_mode": "thread", "max_post_length": 500}))
    adapter.warning_text = lambda text: " "
    posts = []

    async def post(path, payload):
        posts.append(payload)
        return {"_post_broken_thread_root": True} if len(posts) == 1 else {"id": "flat"}

    adapter._api_post = post
    body = "f" * 500
    result = await adapter._post_preserving_thread(
        "channel", {"channel_id": "channel", "root_id": "bad", "message": body}, {"notify": True})
    assert result["id"] == "flat"
    assert len(posts) == 2 and posts[1]["message"] == body and "root_id" not in posts[1]
    assert all(post["message"] for post in posts)
