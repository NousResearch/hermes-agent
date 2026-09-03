"""Regression: per-TURN stream/card identity (PR 85796 review, B2).

Keying coordination state on the thread anchor alone is simultaneously:

  - too coarse: two parallel turns replying INSIDE ONE Slack thread share
    thread_ts — turn A's final sealed turn B's stream with A's content
    while A's own stream stayed open (live probe on the review branch);
  - too fragile: a flat DM with no thread metadata degraded to the bare
    chat id, re-creating the original finding-#10 collision.

The key now prefers the triggering inbound message id (message_id /
reply_to_message_id — per-turn by construction), falling back to the
thread anchor, then the bare chat. Legacy callers with placement-only
metadata still seal via _match_open_draft's single-open-stream fallback,
but NEVER when multiple streams are open (a duplicate message is
recoverable; sealing someone else's stream is not).
"""

import asyncio

import pytest

from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class RecordingTransport:
    def __init__(self):
        self.ops = []
        self._n = 0

    async def send_outbound(self, payload, platform=None):
        self.ops.append(dict(payload))
        self._n += 1
        return {"success": True, "message_id": f"ts.{self._n}"}


def _adapter():
    adapter, _ = _connected_adapter()
    t = RecordingTransport()
    adapter._transport = t
    return adapter, t


class TestSameThreadParallelTurns:
    @pytest.mark.asyncio
    async def test_two_turns_in_one_thread_do_not_collide(self):
        """The exact B2 probe: same thread_ts, distinct triggering message
        ids -> each turn seals its OWN stream with its OWN content."""
        adapter, t = _adapter()
        md_a = {"thread_ts": "1700.100", "message_id": "1700.111"}
        md_b = {"thread_ts": "1700.100", "message_id": "1700.222"}
        await adapter.send_draft("C1", 11, "A partial", metadata=md_a)
        await adapter.send_draft("C1", 12, "B partial", metadata=md_b)
        ra = await adapter.send("C1", "A final", metadata=dict(md_a))
        rb = await adapter.send("C1", "B final", metadata=dict(md_b))
        assert ra.success and rb.success
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert {(s["draft_id"], s["content"]) for s in seals} == {
            (11, "A final"),
            (12, "B final"),
        }, seals
        assert not [o for o in t.ops if o["op"] == "send"]

    @pytest.mark.asyncio
    async def test_card_ids_distinct_per_turn_in_one_thread(self):
        adapter, t = _adapter()
        md_a = {"thread_ts": "1700.100", "message_id": "1700.111"}
        md_b = {"thread_ts": "1700.100", "message_id": "1700.222"}
        tasks = [{"id": "t1", "title": "x", "status": "in_progress"}]
        await adapter.send_native_task_card_progress(
            "C1", tasks, reply_to=None, metadata=md_a
        )
        await adapter.send_native_task_card_progress(
            "C1", tasks, reply_to=None, metadata=md_b
        )
        cards = [o for o in t.ops if o["op"] == "task_card"]
        assert len({c["card_id"] for c in cards}) == 2, cards

    @pytest.mark.asyncio
    async def test_card_stop_uses_same_key_as_send(self):
        adapter, t = _adapter()
        md = {"thread_ts": "1700.100", "message_id": "1700.111"}
        tasks = [{"id": "t1", "title": "x", "status": "in_progress"}]
        await adapter.send_native_task_card_progress(
            "C1", tasks, reply_to=None, metadata=md
        )
        await adapter.stop_native_task_card_progress("C1", metadata=md)
        card_ids = {
            o["card_id"]
            for o in t.ops
            if o["op"] in ("task_card", "task_card_stop")
        }
        assert len(card_ids) == 1, t.ops


class TestFlatDmKeying:
    @pytest.mark.asyncio
    async def test_flat_dm_without_thread_metadata_still_seals(self):
        """Flat DM: no thread anchor anywhere, but the consumer stamps
        reply_to_message_id on frames and the final alike."""
        adapter, t = _adapter()
        md = {"reply_to_message_id": "evt.1"}
        await adapter.send_draft("D1", 5, "partial", metadata=md)
        r = await adapter.send("D1", "final", metadata=dict(md))
        assert r.success
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert len(seals) == 1 and seals[0]["content"] == "final"

    @pytest.mark.asyncio
    async def test_parallel_flat_dm_turns_keyed_by_message_id(self):
        adapter, t = _adapter()
        md_a = {"reply_to_message_id": "evt.a"}
        md_b = {"reply_to_message_id": "evt.b"}
        await adapter.send_draft("D1", 21, "A partial", metadata=md_a)
        await adapter.send_draft("D1", 22, "B partial", metadata=md_b)
        await adapter.send("D1", "A final", metadata=dict(md_a))
        await adapter.send("D1", "B final", metadata=dict(md_b))
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert {(s["draft_id"], s["content"]) for s in seals} == {
            (21, "A final"),
            (22, "B final"),
        }


class TestLegacyPlacementOnlyCallers:
    @pytest.mark.asyncio
    async def test_single_open_stream_absorbs_identityless_final(self):
        """A resolver-lane send with NO turn identity still seals when the
        chat has exactly one open stream (legacy compatibility)."""
        adapter, t = _adapter()
        md = {"thread_ts": "1700.9", "message_id": "1700.910"}
        await adapter.send_draft("C1", 31, "partial", metadata=md)
        r = await adapter.send("C1", "final", metadata=None)
        assert r.success
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert len(seals) == 1 and seals[0]["draft_id"] == 31

    @pytest.mark.asyncio
    async def test_thread_anchored_placement_only_final_seals(self):
        """Review r2, finding 5: metadata carrying ONLY a thread anchor is
        placement info, not turn identity — with one open turn-keyed
        stream in the chat, the final must absorb into it, not post a
        plain duplicate beside the live stream."""
        adapter, t = _adapter()
        md_frames = {"thread_ts": "1700.9", "message_id": "1700.910"}
        await adapter.send_draft("C1", 32, "partial", metadata=md_frames)
        r = await adapter.send("C1", "final", metadata={"thread_ts": "1700.9"})
        assert r.success
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert len(seals) == 1 and seals[0]["draft_id"] == 32, (
            "placement-only final must seal the single open stream"
        )
        assert not [o for o in t.ops if o["op"] == "send"]
        assert not adapter._open_draft_by_chat

    @pytest.mark.asyncio
    async def test_multiple_open_streams_identityless_send_stays_plain(self):
        """With several open streams, an identity-less send must NOT guess:
        plain send (recoverable) over sealing the wrong stream (not)."""
        adapter, t = _adapter()
        await adapter.send_draft(
            "C1", 41, "A", metadata={"message_id": "m.1"}
        )
        await adapter.send_draft(
            "C1", 42, "B", metadata={"message_id": "m.2"}
        )
        r = await adapter.send("C1", "ambiguous final", metadata=None)
        assert r.success
        assert [o["op"] for o in t.ops][-1] == "send"
        assert not [o for o in t.ops if o["op"] == "draft" and o.get("final")]

    @pytest.mark.asyncio
    async def test_placement_only_with_multiple_open_streams_stays_plain(self):
        adapter, t = _adapter()
        await adapter.send_draft(
            "C1", 43, "A", metadata={"thread_ts": "1700.9", "message_id": "m.3"}
        )
        await adapter.send_draft(
            "C1", 44, "B", metadata={"thread_ts": "1700.9", "message_id": "m.4"}
        )
        r = await adapter.send("C1", "ambiguous", metadata={"thread_ts": "1700.9"})
        assert r.success
        assert [o["op"] for o in t.ops][-1] == "send"

    @pytest.mark.asyncio
    async def test_message_id_mismatch_never_falls_back(self):
        """A caller WITH a message id whose key misses must not steal the
        one open stream — its identity is authoritative (different turn)."""
        adapter, t = _adapter()
        await adapter.send_draft("C1", 45, "A", metadata={"message_id": "m.5"})
        r = await adapter.send("C1", "other turn", metadata={"message_id": "m.OTHER"})
        assert r.success
        assert [o["op"] for o in t.ops][-1] == "send"
        assert adapter._open_draft_by_chat, "stream must survive the mismatch"
