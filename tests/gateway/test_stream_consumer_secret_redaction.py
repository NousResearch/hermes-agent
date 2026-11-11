"""Streamed chat egress is secret-redacted like the non-streamed final send (#56039).

Every message the consumer puts on a platform — previews, sealed overflow heads, the split
tail — must carry no credential, including one the overflow split would otherwise cut in two
(each half alone matches no redaction pattern).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until

_KEY = "sk-proj-" + "Q7mZ2xW9rT4kL8vB3nH6pD1sF5gJ0cY"


def _adapter():
    cls = type("ChatAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 1000})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    ids = iter(f"m{i}" for i in range(100))
    adapter.send = AsyncMock(side_effect=lambda **kw: SimpleNamespace(success=True, message_id=next(ids)))
    adapter.edit_message = AsyncMock(
        side_effect=lambda **kw: SimpleNamespace(success=True, message_id=kw["message_id"]))
    return adapter


def _delivered(adapter) -> list[str]:
    return [c.kwargs["content"] for c in adapter.send.call_args_list + adapter.edit_message.call_args_list]


def _leaked_fragments(texts: list[str]) -> list[str]:
    """Any 8-char run of the key body (masks show at most its last 4 chars)."""
    body = _KEY[len("sk-proj-"):]
    runs = {body[i:i + 8] for i in range(len(body) - 7)}
    return [r for r in runs for t in texts if r in t]


async def _stream(adapter, *deltas: str) -> None:
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=5, cursor=""))
    task = asyncio.create_task(consumer.run())
    for delta in deltas:
        # One delta per consumer tick: wait until this one has reached the wire.
        before = len(_delivered(adapter))
        consumer.on_delta(delta)
        await async_wait_until(lambda: consumer._queue.empty() and len(_delivered(adapter)) > before,
                               timeout=5, what="delta delivered")
    consumer.finish()
    await task


@pytest.mark.asyncio
async def test_short_streamed_reply_is_redacted():
    adapter = _adapter()
    await _stream(adapter, f"Your key is {_KEY} - keep it safe.")
    texts = _delivered(adapter)
    assert texts and all(_KEY not in t for t in texts)
    assert not _leaked_fragments(texts)


@pytest.mark.asyncio
async def test_first_send_overflow_split_never_cuts_a_secret_in_two():
    # No whitespace before the key: the adapter's splitter has to hard-cut, and the cut lands
    # inside the key.
    adapter = _adapter()
    await _stream(adapter, "/" * 880 + _KEY + "\nTail line.")
    texts = _delivered(adapter)
    assert len(adapter.send.call_args_list) >= 2  # really split
    assert not _leaked_fragments(texts)


@pytest.mark.asyncio
async def test_sealed_overflow_head_never_cuts_a_secret_in_two():
    adapter = _adapter()
    # An open preview first, so overflow goes through the seal-the-existing-message path.
    await _stream(adapter, "Starting the report.", "/" * 860 + _KEY + "\nTail line.")
    texts = _delivered(adapter)
    assert len(adapter.send.call_args_list) >= 2  # really sealed + continued
    assert not _leaked_fragments(texts)


@pytest.mark.asyncio
async def test_secret_still_arriving_at_the_split_is_redacted_once_complete():
    # The overflow tick sees only the first half of the key at the end of the buffer; the
    # second half arrives on a later tick.
    adapter = _adapter()
    await _stream(adapter, "Starting the report.", " word" * 174 + " " + _KEY[:20],
                  _KEY[20:] + " done.")
    texts = _delivered(adapter)
    assert len(adapter.send.call_args_list) >= 2  # really sealed + continued
    assert not _leaked_fragments(texts)
