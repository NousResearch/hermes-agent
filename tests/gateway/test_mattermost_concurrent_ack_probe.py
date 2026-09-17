"""Regression tests for concurrent Mattermost acknowledgement classification."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _Tick
from plugins.platforms.mattermost.adapter import MattermostAdapter


@pytest.mark.parametrize("other_rejected", [False, True])
@pytest.mark.parametrize("ack_error", [aiohttp.ServerDisconnectedError, asyncio.TimeoutError])
@pytest.mark.asyncio
async def test_other_post_cannot_classify_an_inflight_acknowledgement(other_rejected, ack_error):
    adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500, "reply_mode": "thread"}))
    posts, other_posts = [], []

    async def interleaved_ack_failure():
        # A second real send finishes while the first response body is pending.
        await asyncio.create_task(adapter.send("other-channel", "unrelated request"))
        raise ack_error()

    def post(url, **kwargs):
        assert url.endswith("/api/v4/posts")
        payload = kwargs["json"]
        response = AsyncMock()
        response.__aenter__.return_value = response
        if payload["channel_id"] == "other-channel":
            other_posts.append(payload)
            response.status = 400 if other_rejected else 201
            response.text.return_value = "invalid message"
            response.json.return_value = {"id": "other-id"}
            return response
        posts.append(payload)
        response.status = 201
        response.json.return_value = {"id": f"post-{len(posts)}"}
        if len(posts) == 3:
            response.json.side_effect = interleaved_ack_failure
        return response

    def put(url, **kwargs):
        assert url.endswith("/api/v4/posts/post-1/patch")
        response = AsyncMock()
        response.__aenter__.return_value = response
        response.status = 200
        response.json.return_value = {"id": "post-1"}
        return response

    adapter._session = MagicMock()
    adapter._session.post.side_effect = post
    adapter._session.put.side_effect = put
    adapter._api_get = AsyncMock(return_value={"root_id": "root"})
    consumer = GatewayStreamConsumer(adapter, "channel", metadata={"thread_id": "root"},
                                     config=StreamConsumerConfig(cursor=""))
    assert await consumer._send_or_edit("preview")
    consumer._accumulated = "a" * 490 + " (1/2)  \n" + "b" * 490 + " (2/2)  \n" + "remaining tail"
    await consumer._finalize_edit_path(_Tick(got_done=True))

    assert len(other_posts) == 1
    assert len(posts) == 3, "another request's rejection enabled replay of an unacknowledged continuation"
    assert consumer._preview_message_ids == {"post-1", "post-2"}
    assert consumer._delivery_ambiguous
    assert consumer._message_id == "post-2"

    await consumer._finalize_edit_path(_Tick(got_done=True))
    await consumer._send_or_edit(consumer._accumulated)
    assert len(posts) == 3
    assert consumer._preview_message_ids == {"post-1", "post-2"}
    assert consumer._message_id == "post-2"


@pytest.mark.asyncio
async def test_known_rejection_body_timeout_returns_failed_receipt():
    adapter = MattermostAdapter(PlatformConfig(extra={"reply_mode": "off"}))
    response = AsyncMock()
    response.__aenter__.return_value = response
    response.status = 400
    response.text.side_effect = asyncio.TimeoutError()
    adapter._session = MagicMock()
    adapter._session.post.return_value = response

    result = await adapter.send("channel", "rejected content")

    assert not result.success
    assert not result.message_id
    assert not result.raw_response.get("_delivery_uncertain")
    assert adapter._session.post.call_count == 1


@pytest.mark.asyncio
async def test_ambiguous_continuation_releases_dequeued_flush_waiter():
    import threading
    from gateway.stream_consumer import _FLUSH

    adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500, "reply_mode": "off"}))
    posts = []

    def post(url, **kwargs):
        posts.append(kwargs["json"])
        response = AsyncMock()
        response.__aenter__.return_value = response
        response.status = 201
        response.json.return_value = {"id": f"post-{len(posts)}"}
        if len(posts) == 2:
            response.json.side_effect = asyncio.TimeoutError()
        return response

    response = AsyncMock()
    response.__aenter__.return_value = response
    response.status = 200
    response.json.return_value = {"id": "post-1"}
    adapter._session = MagicMock()
    adapter._session.post.side_effect = post
    adapter._session.put.return_value = response
    adapter._api_get = AsyncMock(return_value={"root_id": ""})
    consumer = GatewayStreamConsumer(adapter, "channel", config=StreamConsumerConfig(cursor=""))
    assert await consumer._send_or_edit("preview")
    consumer.on_delta("a" * 350 + "\n" + "b" * 350 + "\n" + "c" * 100)
    flush_event = threading.Event()
    consumer._queue.put((_FLUSH, flush_event))

    consumer.finish()
    await asyncio.wait_for(consumer.run(), timeout=2)

    assert consumer._source_receipt_segments[-1]["uncertain"]
    assert len(posts) == 2
    assert flush_event.is_set(), "the consumed flush barrier must not wait for its timeout"


@pytest.mark.parametrize("scenario", [
    "later", "during", "overflow", "flush", "segment", "decorated",
    "changed", "changed_lost", "changed_rejected", "changed_long_lost",
    "flush_final", "long_tail", "tail_lost", "media", "overflow_media", "overflow_changed",
    "overflow_media_whitespace", "overflow_media_newlines", "overflow_media_mixed",
])
@pytest.mark.asyncio
async def test_interim_receipt_preserves_later_final_without_replay(scenario):
    import threading
    from types import SimpleNamespace
    from gateway.session import SessionSource
    from gateway.stream_consumer import _FLUSH
    from tests.gateway.test_stale_finalize_suppression import _make_runner

    adapter = MattermostAdapter(PlatformConfig(extra={"max_post_length": 500, "reply_mode": "thread"}))
    posts, edits = [], []
    pending, release, overflow_acked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    correction_pending, correction_release = asyncio.Event(), asyncio.Event()
    changed = "changed" in scenario
    media_whitespace = scenario in {"overflow_media_whitespace", "overflow_media_newlines", "overflow_media_mixed"}
    whitespace = {
        "overflow_media_whitespace": " " * 10,
        "overflow_media_newlines": "\n" * 3,
        "overflow_media_mixed": "\n" * 6 + " \t",
    }.get(scenario, "")
    loss_at = 2 if media_whitespace else (3 if scenario.startswith("overflow") else 1)
    async def lost_ack():
        pending.set()
        await release.wait()
        raise asyncio.TimeoutError()
    def post(url, **kwargs):
        assert url.endswith("/api/v4/posts")
        posts.append(kwargs["json"])
        response = AsyncMock()
        response.__aenter__.return_value = response
        response.status = 201
        response.json.return_value = {"id": f"post-{len(posts)}"}
        if len(posts) == loss_at:
            response.json.side_effect = lost_ack
        elif scenario.startswith("overflow") and len(posts) == 2:
            async def overflow_ack():
                overflow_acked.set()
                return {"id": "post-2"}
            response.json.side_effect = overflow_ack
        elif scenario == "tail_lost" and len(posts) == 2:
            response.json.side_effect = asyncio.TimeoutError()
        elif posts[-1]["message"].startswith("Correction:"):
            if scenario == "changed_lost":
                async def correction_ack():
                    correction_pending.set()
                    await correction_release.wait()
                    raise asyncio.TimeoutError()
                response.json.side_effect = correction_ack
            elif scenario == "changed_long_lost":
                response.json.side_effect = asyncio.TimeoutError()
            elif scenario == "changed_rejected":
                response.status = 400
                response.text.return_value = "invalid message"
        return response
    def put(url, **kwargs):
        edits.append((url, kwargs["json"]))
        response = AsyncMock()
        response.__aenter__.return_value = response
        response.status = 200
        response.json.return_value = {"id": url.split("/")[-2]}
        return response
    adapter._session = MagicMock()
    adapter._session.post.side_effect = post
    adapter._session.put.side_effect = put
    adapter._api_get = AsyncMock(return_value={"root_id": "root"})
    adapter.delete_message = AsyncMock()
    cursor = "▉" if scenario == "decorated" else ""
    consumer = GatewayStreamConsumer(adapter, "channel", metadata={"thread_id": "root"},
        config=StreamConsumerConfig(cursor=cursor, edit_interval=0.01, buffer_threshold=1))
    prefix = "a" * 200
    if scenario == "decorated":
        prefix = "  ```text\n![image](https://example.com/image) (1/2) \t\n" + "a" * 100
    if scenario in {"media", "overflow_media"}:
        prefix = "MEDIA:/tmp/receipt-image.png\n\n" + prefix
    if media_whitespace:
        prefix = "[[audio_as_voice]]" + prefix
    consumer.on_delta(prefix)
    task = asyncio.create_task(consumer.run())
    if scenario.startswith("overflow"):
        # A confirmed preview AND a successful overflow precede the lost ack.
        while not posts:
            await asyncio.sleep(0)
        if media_whitespace:
            addition = "c" * 900 + whitespace
            consumer.on_delta(addition)
        else:
            addition = "c" * 150 + "\n\n" + "d" * 200
            consumer.on_delta(addition)
            await asyncio.wait_for(overflow_acked.wait(), 2)
            addition += "e" * 300
            consumer.on_delta("e" * 300)
        prefix += addition
    await asyncio.wait_for(pending.wait(), 2)
    tail = "b" * (1200 if scenario in {"long_tail", "tail_lost", "changed_long_lost"} else 100)
    full = ("replacement " + "z" * 180 + tail) if changed else prefix + tail
    if scenario == "during":
        consumer.on_delta(tail)
        consumer.finish(full)
    release.set()
    async def classified():
        while not consumer._delivery_ambiguous and not task.done():
            await asyncio.sleep(0)
    await asyncio.wait_for(classified(), 2)
    interim_confirmed = consumer.final_content_delivered
    receipt = consumer._source_receipt
    assert receipt["source"] == consumer._clean_for_display(prefix)
    if media_whitespace:
        assert consumer._source_offset == 400
        assert len(consumer._accumulated) == 718 + len(whitespace)
        assert edits[0][1]["message"] == "a" * 200 + "c" * 182
        assert posts[1]["message"] == "c" * 500
    else:
        assert receipt["attempted_end"] == len(consumer._clean_for_display(prefix))
        assert receipt["confirmed_end"] == (receipt["attempted_end"] - 100 if scenario.startswith("overflow") else 0)
    if scenario != "during":
        consumer.on_delta(tail)
        if scenario in {"flush", "segment", "flush_final"}:
            flush_event = threading.Event()
            consumer.on_delta(None) if scenario == "segment" else consumer._queue.put((_FLUSH, flush_event))
            # Flush both forms before introducing the next independent segment.
            if scenario == "segment":
                consumer._queue.put((_FLUSH, flush_event))
            async def flushed():
                while not flush_event.is_set() and not task.done():
                    await asyncio.sleep(0)
            await asyncio.wait_for(flushed(), 2)
            assert flush_event.is_set()
            assert posts[-1]["message"] == tail
            if scenario != "flush_final":
                full = "new segment answer"
                consumer.on_delta(full)
        consumer.finish(full)
    await asyncio.wait_for(task, 2)
    edit_count = len(edits)
    runner = _make_runner(adapter)
    source = SessionSource(platform=adapter.platform, chat_id="channel", chat_type="group", thread_id="root")
    turn = SimpleNamespace(stream_consumer_holder=[consumer], source=source, session_key="receipt-turn")
    response = {"final_response": full, "messages": [{"role": "assistant", "content": full}],
                "response_transformed": changed}
    if scenario == "changed_lost":
        reconcile = asyncio.create_task(runner._run_agent_mark_streamed_delivery(response, turn))
        await asyncio.wait_for(correction_pending.wait(), 2)
        concurrent_response = {"final_response": full}
        await runner._run_agent_mark_streamed_delivery(concurrent_response, turn)
        assert concurrent_response["already_sent"] and len(posts) == loss_at + 1
        correction_release.set()
        await asyncio.wait_for(reconcile, 2)
    else:
        await runner._run_agent_mark_streamed_delivery(response, turn)
    count = len(posts)
    await runner._run_agent_mark_streamed_delivery(response, turn)
    assert len(posts) == count and len(edits) == edit_count
    assert response["already_sent"]
    if changed:
        assert "".join(p["message"] for p in posts[loss_at:]) == (
            "Correction: Earlier text may appear again because delivery was not confirmed.\n\n" + full)
    elif media_whitespace:
        # Observe the complete run + HTTP-201 lost-ack + gateway reconciliation,
        # not just a replica of the receipt-offset calculation.
        remaining = "".join(p["message"] for p in posts[loss_at:])
        expected_whitespace = {
            "overflow_media_whitespace": " " * 10,
            "overflow_media_newlines": "\n\n",
            "overflow_media_mixed": "\n\n \t",
        }[scenario]
        expected = "c" * 218 + expected_whitespace + tail
        assert remaining == expected, (
            f"R1: remaining={len(remaining)}, expected={len(expected)}, "
            f"confirmed_end={receipt['confirmed_end']}, attempted_end={receipt['attempted_end']}"
        )
        assert edits[0][1]["message"] + posts[1]["message"] + remaining == consumer._clean_for_display(full)
        assert receipt["confirmed_end"] == 382 and receipt["attempted_end"] == 882
    elif scenario not in {"flush", "segment"}:
        assert "".join(p["message"] for p in posts[loss_at:]) == tail
    assert not interim_confirmed, "an interim lost ack must not confirm final content"
    assert consumer._queue.empty()
    if scenario.startswith("overflow"):
        assert len(edits) == (1 if media_whitespace else 2)
        assert consumer._preview_message_ids == ({"post-1", "post-3"} if media_whitespace else {"post-1", "post-2", "post-4"})
        assert consumer.message_id == ("post-3" if media_whitespace else "post-4")
    adapter.delete_message.assert_not_called()
    assert all(len(p["message"]) <= 500 and p["root_id"] == "root" and
               p["props"]["disable_mentions"] for p in posts)
    runner._should_send_voice_reply = MagicMock(return_value=False)
    runner._deliver_media_from_response = AsyncMock()
    event = SimpleNamespace(message_id="inbound", source=source)
    returned = await runner._hmwa_deliver_turn_response(
        event, source, SimpleNamespace(session_id="receipt-session"), "receipt-turn", 1,
        response, response["messages"], full, "footer", False)
    assert returned is None and event._streamed_final_response == full
    assert response["final_response"] == response["messages"][0]["content"] == full
    runner._deliver_media_from_response.assert_awaited_once_with(full, event, adapter)
    assert posts[-1]["message"] == "footer" and len(posts) == count + 1
