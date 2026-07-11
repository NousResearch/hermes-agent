"""Regression: a transient failure after the dedup mark must not drop the
platform redelivery.

``MessageDeduplicator.is_duplicate()`` marks a message seen at check time,
before the adapter has handed it to the gateway.  WeCom redelivers a callback
when our ACK is lost (#47573), and an exception escaping ``_on_message`` tears
down the websocket via ``_listen_loop`` — same process, same adapter instance,
same dedup cache — so the redelivery arrives right after reconnect.  Without
releasing the dedup claim on failure, that redelivery is classified as a
duplicate and the user's message is silently lost for the dedup TTL (300s).

The failure is injected at ``handle_message`` — the dispatch boundary, where
exceptions really do propagate (unlike media downloads, which ``_cache_media``
degrades to "no media" by design).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _payload(msgid: str) -> dict:
    return {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": f"req-{msgid}"},
        "body": {
            "msgid": msgid,
            "from": {"userid": "user-1"},
            "chatid": "",
            "chattype": "single",
            "msgtype": "text",
            "text": {"content": "hello"},
        },
    }


def _adapter():
    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True, extra={"dm_policy": "pairing"}))
    # Disable text-batch aggregation so dispatch is synchronous and the
    # assertions below observe handle_message directly.
    adapter._text_batch_delay_seconds = 0
    return adapter


@pytest.mark.asyncio
async def test_redelivery_processed_after_transient_failure():
    adapter = _adapter()
    # First attempt: dispatch fails transiently after the dedup mark.
    # Redelivery: succeeds.
    adapter.handle_message = AsyncMock(side_effect=[RuntimeError("dispatch failed"), None])

    with pytest.raises(RuntimeError):
        await adapter._on_message(_payload("m-1"))
    assert adapter.handle_message.call_count == 1

    # WeCom redelivers the same msgid after the lost ACK — it must be
    # reprocessed, not swallowed as a duplicate.
    await adapter._on_message(_payload("m-1"))
    assert adapter.handle_message.call_count == 2


@pytest.mark.asyncio
async def test_successful_message_still_deduplicated():
    adapter = _adapter()
    adapter.handle_message = AsyncMock()

    await adapter._on_message(_payload("m-2"))
    adapter.handle_message.assert_called_once()

    # A true duplicate of a successfully processed message stays suppressed.
    await adapter._on_message(_payload("m-2"))
    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_batched_text_flush_failure_releases_claim():
    """TEXT messages dispatch from the background batch-flush task; a failure
    there must release the claim too, not just the synchronous path."""
    adapter = _adapter()
    adapter._text_batch_delay_seconds = 0.1  # keep batching enabled
    adapter.handle_message = AsyncMock(side_effect=[RuntimeError("dispatch failed"), None])

    await adapter._on_message(_payload("m-4"))
    key = next(iter(adapter._pending_text_batch_tasks))
    with pytest.raises(RuntimeError):
        await adapter._pending_text_batch_tasks[key]
    assert adapter.handle_message.call_count == 1

    # Redelivery must be re-enqueued and reprocessed.
    await adapter._on_message(_payload("m-4"))
    await adapter._pending_text_batch_tasks[key]
    assert adapter.handle_message.call_count == 2


@pytest.mark.asyncio
async def test_flush_failure_releases_all_merged_chunk_claims():
    """A failed flush of a merged batch must release every chunk's claim,
    not just the batch head's."""
    adapter = _adapter()
    adapter._text_batch_delay_seconds = 0.1
    adapter.handle_message = AsyncMock(side_effect=[RuntimeError("dispatch failed"), None])

    await adapter._on_message(_payload("m-5a"))
    await adapter._on_message(_payload("m-5b"))  # merged into the same batch
    key = next(iter(adapter._pending_text_batch_tasks))
    with pytest.raises(RuntimeError):
        await adapter._pending_text_batch_tasks[key]

    # Redelivery of the SECOND chunk (not the batch head) must be reprocessed.
    await adapter._on_message(_payload("m-5b"))
    await adapter._pending_text_batch_tasks[key]
    assert adapter.handle_message.call_count == 2


@pytest.mark.asyncio
async def test_cancellation_releases_claim():
    adapter = _adapter()
    adapter.handle_message = AsyncMock(side_effect=[asyncio.CancelledError(), None])

    with pytest.raises(asyncio.CancelledError):
        await adapter._on_message(_payload("m-3"))

    # A message cancelled mid-processing never reached the gateway; its
    # redelivery must be reprocessed.
    await adapter._on_message(_payload("m-3"))
    assert adapter.handle_message.call_count == 2
