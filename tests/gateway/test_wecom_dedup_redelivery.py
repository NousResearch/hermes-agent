"""Regression: a failure after the dedup mark must not drop the platform
redelivery — and a completed turn must never be re-executed by one.

``MessageDeduplicator.is_duplicate()`` marks a message seen at check time,
before the adapter has handed it to the gateway.  WeCom redelivers a callback
when no reply frame acks it (#47573), so the claim lifecycle has three
disjoint owners (see ``WeComAdapter.on_handler_failure`` for the contract):
the ``_on_message`` except (synchronous intake), the ``_flush_text_batch``
except (batched dispatch), and the ``on_handler_failure`` lifecycle hook
(background pipeline, fired by base only when the handler did not complete).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


def _payload(msgid: str, text: str = "hello") -> dict:
    return {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": f"req-{msgid}"},
        "body": {
            "msgid": msgid,
            "from": {"userid": "user-1"},
            "chatid": "",
            "chattype": "single",
            "msgtype": "text",
            "text": {"content": text},
        },
    }


def _adapter(batch_delay: float = 0.0):
    """Build a WeCom adapter; batching is off unless a delay is given."""
    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True, extra={"dm_policy": "pairing"}))
    adapter._text_batch_delay_seconds = batch_delay
    return adapter


async def _drain_background(adapter):
    """Await the real background processing tasks (and their follow-ups)."""
    for _ in range(10):
        tasks = list(adapter._background_tasks)
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)
    raise AssertionError("background processing did not settle")


@pytest.mark.asyncio
async def test_redelivery_processed_after_transient_failure():
    adapter = _adapter()
    # First attempt: dispatch fails synchronously after the dedup mark.
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
async def test_cancellation_releases_claim():
    adapter = _adapter()
    adapter.handle_message = AsyncMock(side_effect=[asyncio.CancelledError(), None])

    with pytest.raises(asyncio.CancelledError):
        await adapter._on_message(_payload("m-3"))

    # A message cancelled mid-intake never reached the gateway; its
    # redelivery must be reprocessed.
    await adapter._on_message(_payload("m-3"))
    assert adapter.handle_message.call_count == 2


@pytest.mark.asyncio
async def test_flush_failure_releases_all_merged_chunk_claims():
    """A failed flush of a merged batch must release every chunk's claim —
    the head's and the appended one's."""
    adapter = _adapter(batch_delay=0.1)
    adapter.handle_message = AsyncMock(
        side_effect=[RuntimeError("dispatch failed"), None, None]
    )

    await adapter._on_message(_payload("m-5a"))
    await adapter._on_message(_payload("m-5b"))  # merged into the same batch
    key = next(iter(adapter._pending_text_batch_tasks))
    with pytest.raises(RuntimeError):
        await adapter._pending_text_batch_tasks[key]

    # Redelivery of the batch HEAD must be reprocessed...
    await adapter._on_message(_payload("m-5a"))
    await adapter._pending_text_batch_tasks[key]
    assert adapter.handle_message.call_count == 2

    # ...and so must the redelivery of the SECOND (appended) chunk.
    await adapter._on_message(_payload("m-5b"))
    await adapter._pending_text_batch_tasks[key]
    assert adapter.handle_message.call_count == 3


@pytest.mark.asyncio
async def test_supersede_cancel_neither_loses_nor_duplicates():
    """A flush cancelled because a newer chunk superseded it must let the
    in-flight dispatch run to completion (shielded): the popped batch is
    neither lost nor double-processed, and its claims stay held."""
    adapter = _adapter(batch_delay=0.05)
    dispatch_started = asyncio.Event()
    dispatch_blocker = asyncio.Event()
    dispatched = []

    async def slow_handle(event):
        dispatched.append(event.text)
        if len(dispatched) == 1:
            dispatch_started.set()
            await dispatch_blocker.wait()  # supersede-cancel lands on the flush here

    adapter.handle_message = slow_handle

    await adapter._on_message(_payload("m-6", text="first"))
    await asyncio.wait_for(dispatch_started.wait(), timeout=2)

    # A newer message supersedes the in-flight flush; the shielded dispatch
    # of m-6 must survive the cancel and complete.
    await adapter._on_message(_payload("m-7", text="second"))
    key = next(iter(adapter._pending_text_batch_tasks))
    dispatch_blocker.set()
    await adapter._pending_text_batch_tasks[key]
    for _ in range(10):
        if len(dispatched) == 2:
            break
        await asyncio.sleep(0)
    assert dispatched == ["first", "second"]

    # m-6 was processed exactly once: its claim is held, so a redelivery is
    # suppressed instead of double-processed.
    await adapter._on_message(_payload("m-6", text="first"))
    assert key not in adapter._pending_text_batches
    assert dispatched == ["first", "second"]


@pytest.mark.asyncio
async def test_background_processing_failure_releases_claim():
    """Failures of the real message handler surface in the background task
    (handle_message returns after scheduling it) — base fires
    on_handler_failure and the claim must be released there."""
    adapter = _adapter()
    adapter.send = AsyncMock()  # error-notification send; no websocket in tests
    handler = AsyncMock(side_effect=[RuntimeError("agent failed"), None])
    adapter.set_message_handler(handler)

    await adapter._on_message(_payload("m-8"))
    await _drain_background(adapter)
    assert handler.await_count == 1

    # WeCom redelivers (no reply frame was sent) — must be reprocessed.
    await adapter._on_message(_payload("m-8"))
    await _drain_background(adapter)
    assert handler.await_count == 2

    # Successful processing keeps the claim: a true duplicate stays dropped.
    await adapter._on_message(_payload("m-8"))
    await _drain_background(adapter)
    assert handler.await_count == 2


@pytest.mark.asyncio
async def test_delivery_failure_after_completed_handler_keeps_claim():
    """A completed turn whose reply SEND fails is a delivery problem, not a
    handler failure: re-processing it would duplicate tool side effects, so
    the claim must stay held and the redelivery must stay suppressed."""
    adapter = _adapter()
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="boom"))
    handler = AsyncMock(return_value="reply text")
    adapter.set_message_handler(handler)

    await adapter._on_message(_payload("m-9"))
    await _drain_background(adapter)
    assert handler.await_count == 1

    # base classifies this run FAILURE (delivery heuristic), but the handler
    # completed — the redelivery must NOT re-run the turn.
    await adapter._on_message(_payload("m-9"))
    await _drain_background(adapter)
    assert handler.await_count == 1


@pytest.mark.asyncio
async def test_busy_inline_dispatch_failure_releases_claim():
    """On a busy session, bypass commands dispatch inline inside base's
    handle_message where failures are swallowed — base must still report
    them via on_handler_failure so the claim is released."""
    adapter = _adapter()
    adapter.send = AsyncMock()
    started = asyncio.Event()
    blocker = asyncio.Event()
    calls = []

    async def handler(event):
        calls.append(event.text)
        if len(calls) == 1:
            started.set()
            await blocker.wait()
            return None
        raise RuntimeError("inline dispatch failed")

    adapter.set_message_handler(handler)

    # Occupy the session with a long-running turn.
    await adapter._on_message(_payload("m-10", text="long task"))
    await asyncio.wait_for(started.wait(), timeout=2)

    # /status bypasses the active-session guard and dispatches inline; the
    # handler raises and base swallows it — but the claim must be released.
    await adapter._on_message(_payload("m-11", text="/status"))
    assert calls[-1] == "/status"

    # WeCom redelivers the failed command — it must be reprocessed.
    await adapter._on_message(_payload("m-11", text="/status"))
    assert calls.count("/status") == 2

    blocker.set()
    await _drain_background(adapter)
