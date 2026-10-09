"""Local integration trace: REAL StreamConsumer + REAL RelayAdapter + stub
transport. Reproduces the multi-segment live-cards turn to reveal the exact
op sequence the connector sees (finding #4/#5 forensics — Alice canary).

Run: python -m pytest tests/gateway/relay/test_live_cards_flow_trace.py -q -s
"""
import asyncio
import sys
import types

import pytest

from gateway.relay.adapter import RelayAdapter


class TraceTransport:
    """Stub connector transport recording every outbound op."""

    def __init__(self, fail_ops=None):
        self.ops = []
        self.fail_ops = set(fail_ops or ())
        self._ts = 1000

    async def send_outbound(self, payload, platform=None):
        op = payload.get("op")
        self.ops.append(dict(payload))
        if op in self.fail_ops:
            return {"success": False, "error": f"stub-forced {op} failure"}
        self._ts += 1
        return {"success": True, "message_id": f"{self._ts}.000"}


def _mk_adapter(supported_ops=("send", "edit", "typing", "draft", "task_card", "task_card_stop")):
    # Reuse the live-cards test helper wiring (descriptor + config stubs).
    from tests.gateway.relay.test_relay_live_cards import _connected_adapter
    adapter, _ = _connected_adapter(supported_ops=supported_ops)
    t = TraceTransport()
    adapter._transport = t
    return adapter, t


def test_trace_multisegment_draft_flow():
    """Simulate the consumer's actual call pattern for a 3-segment turn:
    seg1 drafts -> segment-break finalize (send) -> seg2 drafts ->
    segment-break finalize (send) -> seg3 drafts -> turn-final send.
    Prints the op timeline; asserts the invariant we EXPECT enterprise-wise:
    at most ONE final message identity visible to the user.
    """
    adapter, t = _mk_adapter()
    loop = asyncio.new_event_loop()
    md = {"thread_ts": "1700.100"}

    async def turn():
        # segment 1 streaming
        await adapter.send_draft("C1", 7, "seg1 partial", metadata=md)
        await adapter.send_draft("C1", 7, "seg1 complete.", metadata=md)
        # tool boundary (fix #5): consumer now emits a cumulative draft
        # frame instead of a finalize send for stream-is-the-message
        # adapters — simulate that call shape.
        await adapter.send_draft("C1", 7, "seg1 complete.", metadata=md)
        # segment 2 streaming (fix #4: same draft_id)
        await adapter.send_draft("C1", 7, "seg1 complete.\nseg2 partial", metadata=md)
        await adapter.send_draft("C1", 7, "seg1 complete.\nseg2 complete.", metadata=md)
        # segment 3 + the ONE turn-final send (seal-intercepted)
        await adapter.send_draft(
            "C1", 7, "seg1 complete.\nseg2 complete.\nfinal answer partial", metadata=md
        )
        r3 = await adapter.send(
            "C1", "seg1 complete.\nseg2 complete.\nfinal answer complete.", metadata=md
        )
        return r3

    r3 = loop.run_until_complete(turn())
    print("\n--- OP TIMELINE ---")
    for i, op in enumerate(t.ops):
        print(f"{i:2d} {op['op']:<16} final={op.get('final')} draft_id={op.get('draft_id')} "
              f"content={str(op.get('content'))[:40]!r}")
    finals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
    plain_sends = [o for o in t.ops if o["op"] == "send"]
    print(f"seal frames: {len(finals)}, plain sends: {len(plain_sends)}")
    # The user-visible message count = seals + plain sends (each seal ends a
    # visible stream message; each plain send posts a message).
    visible = len(finals) + len(plain_sends)
    print(f"user-visible messages this turn: {visible}")
    assert visible == 1, (
        f"turn produced {visible} user-visible messages (expected 1): "
        f"each segment-break send() gets converted to draft(final=true) by the "
        f"adapter's seal-interception, sealing a stream PER SEGMENT"
    )


def test_trace_parallel_turns_do_not_collide():
    """Finding #10 (live): three concurrent turns in ONE flat DM must keep
    fully independent stream + card identities. Per-chat keying merged
    turn B's card into turn A's and clobbered seal state (3x duplicates)."""
    adapter, t = _mk_adapter()
    loop = asyncio.new_event_loop()
    # Each turn's metadata carries its own thread anchor (inbound stamps
    # thread_ts = event.thread_ts or ts on every top-level message).
    md_a = {"thread_ts": "100.1"}
    md_b = {"thread_ts": "200.2"}

    async def interleaved():
        # A and B stream interleaved on the SAME chat with different anchors
        await adapter.send_draft("C1", 11, "A partial", metadata=md_a)
        await adapter.send_draft("C1", 12, "B partial", metadata=md_b)
        # A's card and B's card must be distinct card_ids
        await adapter.send_native_task_card_progress(
            "C1", [{"id": "t1", "title": "x", "status": "in_progress"}],
            reply_to=None, metadata=md_a)
        await adapter.send_native_task_card_progress(
            "C1", [{"id": "t2", "title": "y", "status": "in_progress"}],
            reply_to=None, metadata=md_b)
        # A seals; B keeps streaming — B's state must survive A's seal
        ra = await adapter.send("C1", "A final.", metadata=md_a)
        await adapter.send_draft("C1", 12, "B partial more", metadata=md_b)
        rb = await adapter.send("C1", "B final.", metadata=md_b)
        return ra, rb

    ra, rb = loop.run_until_complete(interleaved())
    drafts = [o for o in t.ops if o["op"] == "draft"]
    seals = [o for o in drafts if o.get("final")]
    cards = [o for o in t.ops if o["op"] == "task_card"]
    plain = [o for o in t.ops if o["op"] == "send"]
    # Distinct card identities per turn:
    assert len({c["card_id"] for c in cards}) == 2, cards
    # Each turn sealed its OWN stream (2 seals, matching draft_ids 11/12):
    assert sorted(s["draft_id"] for s in seals) == [11, 12], seals
    # No leaked plain send: both finals absorbed by their own seals:
    assert not plain, plain
    # B's post-A-seal frame was NOT dropped by A's tombstone:
    b_frames = [d for d in drafts if d["draft_id"] == 12 and not d.get("final")]
    assert len(b_frames) == 2, b_frames
