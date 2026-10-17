"""A seal must not leave an oversized leftover on the non-final send lane.

``_seal_overflow_heads`` clears the edit target MID-ITERATION, so the overflow
gate evaluated at the top of ``run()`` is already stale by the time the leftover
is pushed. When that leftover still exceeds one message, it reaches
``_push_update`` -> ``_first_send`` with no message to edit, on a NON-final tick.
``adapter.send()`` then chunks and caps the payload itself, numbering the pieces
``(i/n)``; the turn-final lane afterwards publishes the SAME text with its own
denominator.

Observed in production on Discord: one inbound message, one API call, no tool
turns, a 36642-char answer - and two interleaved sequences on screen, ``(i/10)``
and ``(i/9)``, whose first chunks were byte-identical once the indicator was
stripped, and whose concatenations shared a prefix. The gateway's normal final
send was correctly suppressed, which is what places the duplicate inside the
consumer rather than in the gateway's delivery ledger.

Contract pinned here: on a non-final lane the CONSUMER owns chunking, so no line
of the answer reaches the channel twice as a new message, and none is lost when a
boundary or a failed tail send lands in the sealing tick.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

# The consumer floors its own length budget at 500 chars, so a toy adapter limit
# below that floor would make it legitimately emit chunks larger than the limit -
# a measurement artifact, not a leak. Use a realistic limit instead.
LIMIT = 2000
HEAD = "head line\n" * 60
TAIL = "".join(f"tail line {i:03d} filler filler filler filler filler\n" for i in range(240))


def _make_plain_adapter():
    """No rich tail at all: ``preferred_final_chunks`` declines, so an oversized
    payload has no atomic-chunk excuse and must be split by the consumer."""
    from gateway.platforms.base import BasePlatformAdapter

    cls = type(
        "PlainAdapter",
        (BasePlatformAdapter,),
        {
            "MAX_MESSAGE_LENGTH": LIMIT,
            "preferred_final_chunks": lambda self, text, budget: None,
            "preferred_final_split_index": lambda self, text, budget: None,
        },
    )
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    return adapter


def _wire(adapter, fail_first_send_of=None):
    """Record delivered sends/edits. ``fail_first_send_of``: the first send whose content
    contains that marker fails (success=False) and is not delivered."""
    sends, edits = [], []
    failed = []

    async def fake_send(**kw):
        content = kw.get("content", "")
        if fail_first_send_of and not failed and fail_first_send_of in content:
            failed.append(content)
            return SimpleNamespace(success=False, error="timeout")
        sends.append((content, (kw.get("metadata") or {}).get("notify")))
        return SimpleNamespace(success=True, message_id=f"m{len(sends)}")

    async def fake_edit(**kw):
        edits.append(kw.get("content", ""))
        return SimpleNamespace(success=True, message_id=kw.get("message_id", "m1"))

    adapter.send = AsyncMock(side_effect=fake_send)
    adapter.edit_message = AsyncMock(side_effect=fake_edit)
    return sends, edits, failed


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["none", "commentary", "segment_break", "tail_send_fails"])
async def test_every_tail_line_reaches_the_channel_exactly_once(boundary):
    """Live shape: a streamed head is already on screen when a long tail pushes the buffer far
    past the limit, so the seal leaves a leftover that STILL overflows. A commentary or tool
    boundary may be drained in that same tick, and the tail send carrying the boundary may fail.
    Every tail line must reach the channel, never twice as a new message, and post-boundary
    text must start a new message instead of being glued onto the pre-boundary preview."""
    adapter = _make_plain_adapter()
    sends, edits, failed = _wire(
        adapter, fail_first_send_of="tail line 239" if boundary == "tail_send_fails" else None)
    config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="")
    consumer = GatewayStreamConsumer(adapter, "chat_plain", config)
    consumer.on_delta(HEAD)
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.06)
    # Enqueued synchronously, so one drain folds TAIL and stops at the boundary.
    consumer.on_delta(TAIL)
    if boundary == "commentary":
        consumer.on_commentary("COMMENTARY-MARKER")
    if boundary != "none":
        consumer.on_segment_break()
        consumer.on_delta("POST-TOOL-MARKER")
    await asyncio.sleep(0.12)
    consumer.finish()
    # asyncio.wait_for, never a bare await: a hold branch that fails to yield
    # spins hot and would hang the suite instead of failing it.
    await asyncio.wait_for(task, timeout=10)

    if boundary == "tail_send_fails":
        assert failed, "the harness never failed the tail send"
    joined = "".join(c for c, _ in sends)
    repeated = [f"tail line {i:03d}" for i in range(240) if joined.count(f"tail line {i:03d}") > 1]
    assert not repeated, (
        f"{len(repeated)} tail line(s) reached the channel more than once as NEW "
        f"messages (first: {repeated[:3]}) - that is the interleaved duplicate"
    )
    texts = [c for c, _ in sends] + edits
    lost = [f"tail line {i:03d}" for i in range(240)
            if not any(f"tail line {i:03d}" in t for t in texts)]
    assert not lost, f"{len(lost)} tail line(s) never reached the channel (first: {lost[:3]})"
    if boundary == "commentary":
        assert any("COMMENTARY-MARKER" in t for t in texts), "commentary was dropped"
    if boundary != "none":
        assert any("POST-TOOL-MARKER" in t for t in texts)
        glued = [t for t in texts if "POST-TOOL-MARKER" in t and "tail line" in t]
        assert not glued, "post-boundary text was glued onto the pre-boundary preview"
