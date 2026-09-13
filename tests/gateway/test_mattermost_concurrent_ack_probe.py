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

    await asyncio.wait_for(consumer.run(), timeout=2)

    assert consumer._delivery_ambiguous
    assert len(posts) == 2
    assert flush_event.is_set(), "the consumed flush barrier must not wait for its timeout"
