"""Regression (PR #96942, reviewer finding #2): the native delivery ledger must
not record post-stream AUGMENTATION as delivered before it has reached the wire.

Native non-split turn: body "Answer" is streamed, then ``finish()`` adopts the
authoritative final "Answer\\nVerified footer" — a verifier footer the streaming
accumulator never saw. The native finalize path only frames ``_accumulated``
("Answer") to the wire, so the footer is never transmitted. Before the fix
``_record_turn_final_payload`` substituted the healed ledger (final_raw, WITH the
footer) as ``_delivered_final_text``, so ``delivered_final_matches`` reported the
augmented payload as delivered and the gateway SUPPRESSED its corrective send —
the footer was silently lost.

Invariant: until the augmentation is actually sent, ``delivered_final_matches``
of the augmented final must be False so the gateway's corrective send fires. A
pure re-expression of the already-delivered body must STILL match (no double
bubble) — the tension this fix must respect.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

from tests.gateway.test_stream_consumer_wecom_native import (
    _make_native_streaming_adapter,
)


@pytest.mark.asyncio
async def test_native_augmented_final_not_recorded_until_sent():
    """Caller-boundary drive (finish() -> run()): the footer is never framed, so
    the augmented final must NOT be reported as delivered."""
    adapter = _make_native_streaming_adapter()
    cfg = StreamConsumerConfig(
        chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
    )
    consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

    consumer.on_delta("Answer")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    # Authoritative final carries a post-stream verifier footer the stream never saw.
    consumer.finish("Answer\nVerified footer")
    await task

    # Native non-split kept _accumulated as the framed body (footer never framed).
    assert consumer._use_native_streaming is True
    assert consumer._accumulated == "Answer"

    # The wire received ONLY "Answer" — no frame ever carried the footer.
    assert not any("footer" in f["text"] for f in adapter.frames), adapter.frames
    finalize_frames = [f for f in adapter.frames if f["finalize"]]
    assert len(finalize_frames) == 1, adapter.frames
    assert finalize_frames[0]["text"] == "Answer"

    # THE INVARIANT: the augmented final must NOT be recorded as delivered until it
    # is actually sent, so the gateway's corrective send is NOT suppressed.
    assert consumer.delivered_final_matches("Answer\nVerified footer") is False
    # The delivered body itself, of course, still matches.
    assert consumer.delivered_final_matches("Answer") is True


def test_augmentation_ledger_not_recorded_as_delivered():
    """Record-level distinction: a ledger that STRICTLY EXTENDS the framed body
    (augmentation) records only the delivered body."""
    adapter = _make_native_streaming_adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(chat_type="dm"),
    )
    consumer._use_native_streaming = True
    consumer._turn_split_delivery = False
    consumer._accumulated = "Full delivered body."
    consumer._last_sent_text = "Full delivered body."
    consumer._stream_ledger = "Full delivered body.\nVerified footer."  # augmentation

    consumer._record_turn_final_payload("Full delivered body.")

    assert consumer.delivered_final_matches(
        "Full delivered body.\nVerified footer.") is False
    assert consumer.delivered_final_matches("Full delivered body.") is True


def test_reexpression_ledger_still_recorded_as_delivered():
    """Double-bubble non-regression pin: when the healed ledger RE-EXPRESSES the
    framed body (no net-new content), it MUST still be recorded so
    delivered_final_matches is True and the gateway does not resend."""
    adapter = _make_native_streaming_adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(chat_type="dm"),
    )
    consumer._use_native_streaming = True
    consumer._turn_split_delivery = False
    consumer._accumulated = "Full delivered body."
    consumer._last_sent_text = "Full delivered body."
    consumer._stream_ledger = "Full delivered body."  # adopt healed to the same

    consumer._record_turn_final_payload("Full delivered body.")

    assert consumer.delivered_final_matches("Full delivered body.") is True


def test_tail_record_substitutes_full_ledger_double_bubble():
    """Double-bubble non-regression pin (rotation/tool-bearing shape): finalize
    passes only the tail while _accumulated is unset here; the full healed ledger
    must be substituted so the complete final still matches (mirrors
    test_native_finalize_records_ledger_so_no_duplicate_resend)."""
    adapter = _make_native_streaming_adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(chat_type="dm"),
    )
    full_final = "".join(f"segment-{i:04d} line of streamed native content.\n"
                         for i in range(80))
    tail_only = full_final[1054:]

    consumer._use_native_streaming = True
    consumer._turn_split_delivery = False
    consumer._stream_ledger = full_final  # healed by _adopt_final_text
    # _accumulated left empty (the finalize-tail reconstruction case).

    consumer._record_turn_final_payload(tail_only)

    assert consumer.delivered_final_matches(full_final) is True
