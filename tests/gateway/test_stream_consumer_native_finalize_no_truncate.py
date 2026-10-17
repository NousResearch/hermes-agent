"""Behavior contract for _adopt_final_text on native streaming (rotation-safe).

A native-streaming turn keeps ``_accumulated`` as the FULL cumulative body and
renders each frame sliced at ``_native_split_offset`` (see stream_consumer.py
~203-208, 270). The authoritative turn-final (``final_raw``, the LAST API call's
text) is much shorter than the rotation-accumulated body. Regression #1: the
rebase collapsed _adopt_final_text to a single non-split branch that overwrote
``_accumulated = final_raw`` for native too, so after a Layer-2 rotation split at
offset N the fresh bubble's ``_accumulated[N:]`` slice went EMPTY — the rotated
bubble lost its body. The fix restores the pre-rebase three-branch shape: on
native non-split finalize, heal the LEDGER ONLY and never touch ``_accumulated``.
"""

from __future__ import annotations

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

from tests.gateway.test_stream_consumer_wecom_native import (
    _make_native_streaming_adapter,
)


def _native_consumer() -> GatewayStreamConsumer:
    adapter = _make_native_streaming_adapter()
    cfg = StreamConsumerConfig(chat_type="dm")
    return GatewayStreamConsumer(adapter, "chat-1", cfg)


def test_native_finalize_leaves_accumulated_full_and_heals_ledger():
    """Native non-split: _accumulated stays full (rotation slice non-empty),
    ledger is healed to the authoritative final."""
    consumer = _native_consumer()

    # A long cumulative body (mirrors streamed frames across a rotation).
    long_body = "".join(f"segment-{i:04d} line of streamed native content.\n"
                        for i in range(80))
    assert len(long_body) > 2000

    consumer._use_native_streaming = True
    consumer._turn_split_delivery = False
    consumer._accumulated = long_body
    consumer._stream_ledger = long_body
    consumer._message_id = "m"
    consumer._native_split_offset = 1054  # a rotation seal point inside the body

    short_final = "Final answer from the last API call only."
    assert len(short_final) < len(long_body)

    consumer._adopt_final_text(short_final)

    # _accumulated must NOT be truncated to the short final.
    assert consumer._accumulated == long_body
    # The fresh-bubble slice at the rotation offset stays non-empty.
    assert consumer._accumulated[consumer._native_split_offset:] != ""
    # The ledger IS healed to the authoritative final (for reconciliation).
    assert consumer._stream_ledger == short_final


def test_non_native_finalize_still_adopts_wholesale():
    """Toggle arm: non-native non-split path is unchanged — _accumulated is
    overwritten with the authoritative final."""
    consumer = _native_consumer()

    consumer._use_native_streaming = False
    consumer._turn_split_delivery = False
    consumer._accumulated = "streamed body that differs from the final"
    consumer._stream_ledger = consumer._accumulated
    consumer._message_id = "m"

    differing_final = "Authoritative final with a post-stream footer."
    consumer._adopt_final_text(differing_final)

    assert consumer._accumulated == differing_final
    assert consumer._stream_ledger == differing_final


def test_native_finalize_records_ledger_so_no_duplicate_resend():
    """Regression #2 (double bubble): on native non-split finalize,
    _record_turn_final_payload must record the un-truncated ledger — not the
    tail-only raw text — so delivered_final_matches(full_final) returns True and
    the gateway does NOT resend the tail as a fresh second bubble.

    The rebase collapsed the recording guard to split-only, dropping the native
    disjunct; a rotated/tool-bearing native turn then recorded a tail-only payload
    and the gateway issued a corrective plain-send (agent.log: 'Sending response
    (231 chars)') → duplicate bubble.
    """
    consumer = _native_consumer()

    full_final = "".join(f"segment-{i:04d} line of streamed native content.\n"
                         for i in range(80))
    tail_only = full_final[1054:]  # what finalize passes post-rotation (a slice)
    assert tail_only and tail_only != full_final

    consumer._use_native_streaming = True
    consumer._turn_split_delivery = False
    consumer._stream_ledger = full_final  # healed by _adopt_final_text
    consumer._message_id = "m"

    # Finalize records the tail — but the guard must substitute the full ledger.
    consumer._record_turn_final_payload(tail_only)

    # The recorded payload is the FULL final, so reconciliation matches and the
    # gateway suppresses the corrective duplicate send.
    assert consumer.delivered_final_matches(full_final) is True


def test_non_native_finalize_records_raw_text():
    """Toggle arm: non-native path records the raw text unchanged (no ledger
    substitution), so behavior for the non-native case is untouched."""
    consumer = _native_consumer()

    consumer._use_native_streaming = False
    consumer._turn_split_delivery = False
    consumer._stream_ledger = "a long ledger that should NOT be substituted here"
    consumer._message_id = "m"

    raw = "the actual delivered final text"
    consumer._record_turn_final_payload(raw)

    assert consumer.delivered_final_matches(raw) is True
