"""Regressions for source-span and acknowledgement reconciliation in Mattermost."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base import SendResult
from gateway.session import SessionSource
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
    assert response.get("already_sent") is not True
    assert response.get("delivery_error") == consumer.source_delivery_error
    correction = response["delivery_fallback_text"]
    assert correction.startswith("Correction:")
    assert correction.endswith(source[len(prefix):])
    assert prefix not in correction
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    gateway = GatewayTurnPersistenceMixin()
    gateway._delivery_adapter_for = lambda _source: adapter
    gateway._should_send_voice_reply = lambda *args, **kwargs: False
    delivered = await gateway._hmwa_deliver_turn_response(
        SimpleNamespace(), SimpleNamespace(chat_id="channel"), SimpleNamespace(session_id="s"),
        "session", 1, response, [], source, None, False)
    assert delivered == correction
    assert response["final_response"] == source
    repeated = {"final_response": source}
    await GatewayTurnMixin()._run_agent_mark_streamed_delivery(repeated, turn)
    assert repeated.get("already_sent") is True
    assert "delivery_fallback_text" not in repeated
    assert adapter.send.await_count == 3
    assert [call.kwargs["content"] for call in adapter.send.await_args_list] == [
        source, source[len(prefix):], source[len(prefix):]]


@pytest.mark.asyncio
async def test_rejected_stream_suffix_correction_still_delivers_requested_files_in_original_thread(
    tmp_path, monkeypatch,
):
    from gateway.platforms import base
    from gateway.run_notifications import GatewayNotificationsMixin
    from gateway.run_turn import GatewayTurnMixin

    class Runner(GatewayTurnMixin, GatewayNotificationsMixin):
        def _thread_metadata_for_source(self, source, reply_to_message_id=None):
            return {"thread_id": source.thread_id}

        def _reply_anchor_for_event(self, event):
            return event.message_id

        def _delivery_adapter_for(self, source):
            return self.adapter

        def _should_send_voice_reply(self, *args, **kwargs):
            return False

    media_root = tmp_path / "media"
    media_root.mkdir()
    files = [media_root / "report.pdf", media_root / "data.csv"]
    for file in files:
        file.write_bytes(file.name.encode())
    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (media_root,))

    adapter = MattermostAdapter(PlatformConfig(extra={
        "reply_mode": "thread", "max_post_length": 500,
    }))
    adapter._api_get = AsyncMock(return_value={"id": "root", "channel_id": "original-channel"})
    adapter._upload_file = AsyncMock(side_effect=["file-1", "file-2"])
    posts = []

    async def post(path, payload):
        assert path == "posts"
        posts.append(payload)
        if len(posts) in (2, 3, 4):
            return {"_post_rejection_status": 400}
        return {"id": f"post-{len(posts)}"}

    adapter._api_post = post
    source = SessionSource(platform=adapter.platform, chat_id="original-channel",
                           chat_type="group", thread_id="root")
    event = MessageEvent(text="send both files", message_type=MessageType.TEXT,
                         source=source, message_id="inbound")
    runner = Runner()
    runner.adapter = adapter
    final = "Reply: " + "x" * 493 + " remaining text.\n" + "\n".join(f"MEDIA:{p}" for p in files)
    consumer = GatewayStreamConsumer(
        adapter, source.chat_id, config=StreamConsumerConfig(cursor=""),
        metadata=runner._thread_metadata_for_source(source),
    )
    assert not await consumer._send_or_edit(final, finalize=False)
    assert consumer.source_delivery_pending
    result = {"final_response": final}
    turn = SimpleNamespace(stream_consumer_holder=[consumer], source=source, session_key="s")
    await runner._run_agent_mark_streamed_delivery(result, turn)
    assert result.get("already_sent") is not True
    assert result["delivery_fallback_text"].startswith("Correction:")
    assert "MEDIA:" not in result["delivery_fallback_text"]

    correction = await runner._hmwa_deliver_turn_response(
        event, source, SimpleNamespace(session_id="s"), "s", 1,
        result, [], final, None, False,
    )
    assert correction == result["delivery_fallback_text"]
    await adapter.send(source.chat_id, correction, metadata=runner._event_thread_metadata(event, source))

    visible = consumer._clean_for_display(final)
    assert [p["message"] for p in posts[:4]] == [visible[:500]] + [visible[500:]] * 3
    assert len(posts) == 7
    assert [p.get("file_ids") for p in posts] == [None] * 4 + [["file-1"], ["file-2"], None]
    assert posts[-1]["message"] == correction
    assert all(p["channel_id"] == "original-channel" and p["root_id"] == "root" for p in posts)
    assert all(p["props"]["disable_mentions"] for p in posts)
    assert [call.args[:3] for call in adapter._upload_file.await_args_list] == [
        ("original-channel", file.name.encode(), file.name) for file in files
    ]


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
    adapter._upload_image_batch = AsyncMock(side_effect=[
        ([f"file-{i}" for i in range(5)], ["image.png"] * 5, None),
        (["file-5"], ["image.png"], None)])
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
    assert result.raw_response["partial_failure"] and result.raw_response["failed_media"] == 1


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
async def test_disabled_warning_does_not_block_flat_recovery_of_long_final():
    adapter = MattermostAdapter(PlatformConfig(extra={"reply_mode": "thread", "max_post_length": 500}))
    adapter.warning_text = lambda text: " "  # warning policy emitted only invisible whitespace
    posts = []

    async def post(path, payload):
        posts.append(payload)
        return {"_post_rejection_status": 400, "_post_broken_thread_root": True} if len(posts) == 1 else {"id": "flat"}

    adapter._api_post = post
    body = "f" * 500
    result = await adapter._post_preserving_thread("channel", {"channel_id": "channel", "root_id": "bad", "message": body}, {"notify": True})
    assert result["id"] == "flat"
    assert len(posts) == 2 and posts[1]["message"] == body and "root_id" not in posts[1]
    assert all(post["message"] for post in posts)
