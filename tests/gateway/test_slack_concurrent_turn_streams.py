"""Concurrent turns in one Slack channel must not share a streaming slot.

``_active_streams`` used to be keyed on ``chat_id`` alone, so two turns in the
same channel competed for one slot. Each time a turn found the slot holding the
other turn's stream it sealed that stream and opened a fresh one, so a single
answer arrived as several separate Slack messages — the duplicate-message
symptom in #95430 (cause C).

Keying per ``(chat_id, draft_id)`` gives each turn its own stream. The
same-turn segment hand-off (transports that bump ``draft_id`` at a tool
boundary) must still seal its own predecessor, which is what distinguishes
this from simply never sealing.

Behaviour contract asserted here:
  * two concurrent turns → exactly one startStream each, no cross-sealing
  * each turn's final send seals its OWN stream, no chat.postMessage
  * a same-turn draft_id bump still seals the superseded segment
  * a sibling turn's stream survives that hand-off, including one sharing
    the same thread anchor that started later (higher ``draft_id``)
  * a turn that never finalizes is swept by age, so per-turn keys cannot
    accumulate stale entries or leave a live-typing indicator up
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _make_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra={})
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    client = AsyncMock()
    counter = {"n": 0}

    async def _start(**_kwargs):
        # Real await boundary: the slot bookkeeping spans this call, which is
        # exactly the window the two turns used to interleave in.
        await asyncio.sleep(0)
        counter["n"] += 1
        return {"ok": True, "ts": f"ts-{counter['n']}"}

    client.chat_startStream = AsyncMock(side_effect=_start)
    client.chat_appendStream = AsyncMock(return_value={"ok": True})
    client.chat_stopStream = AsyncMock(return_value={"ok": True})
    client.chat_postMessage = AsyncMock(return_value={"ts": "999.000"})
    client.chat_update = AsyncMock(return_value={"ts": "999.000"})
    adapter._get_client = MagicMock(return_value=client)
    adapter.stop_typing = AsyncMock()
    adapter._running = True
    return adapter, client


# Distinct thread anchors: two people asking in the same channel each get
# their own thread, which is what the gateway stamps on every frame.
TURN_A = {"thread_id": "111.000", "user_id": "U1"}
TURN_B = {"thread_id": "222.000", "user_id": "U2"}


class TestConcurrentTurns:
    @pytest.mark.asyncio
    async def test_each_turn_opens_exactly_one_stream(self):
        adapter, client = _make_adapter()

        await asyncio.gather(
            adapter.send_draft("C1", 1, "turn one ", metadata=TURN_A),
            adapter.send_draft("C1", 2, "turn two ", metadata=TURN_B),
        )
        await asyncio.gather(
            adapter.send_draft("C1", 1, "turn one continued", metadata=TURN_A),
            adapter.send_draft("C1", 2, "turn two continued", metadata=TURN_B),
        )

        assert client.chat_startStream.await_count == 2, (
            "each turn must open exactly one stream; extra startStream calls "
            "mean the turns are taking the slot from each other"
        )
        assert client.chat_stopStream.await_count == 0, (
            "no stream should be sealed while both turns are still streaming"
        )
        assert ("C1", 1) in adapter._active_streams
        assert ("C1", 2) in adapter._active_streams
        assert adapter._active_streams[("C1", 1)]["ts"] != (
            adapter._active_streams[("C1", 2)]["ts"]
        )

    @pytest.mark.asyncio
    async def test_each_turn_finalizes_into_its_own_stream(self):
        adapter, client = _make_adapter()

        await adapter.send_draft("C1", 1, "answer one", metadata=TURN_A)
        await adapter.send_draft("C1", 2, "answer two", metadata=TURN_B)

        first = await adapter.send("C1", "answer one, done.", metadata=TURN_A)
        second = await adapter.send("C1", "answer two, done.", metadata=TURN_B)

        assert first.success and second.success
        assert first.message_id != second.message_id, (
            "both turns finalized into the same stream"
        )
        client.chat_postMessage.assert_not_awaited()
        assert not adapter._active_streams

    @pytest.mark.asyncio
    async def test_interleaved_frames_keep_their_own_deltas(self):
        """Appends must be computed against the sending turn's own text."""
        adapter, client = _make_adapter()

        await adapter.send_draft("C1", 1, "AAA", metadata=TURN_A)
        await adapter.send_draft("C1", 2, "BBB", metadata=TURN_B)
        await adapter.send_draft("C1", 1, "AAA111", metadata=TURN_A)
        await adapter.send_draft("C1", 2, "BBB222", metadata=TURN_B)

        deltas = [
            (c.kwargs["ts"], c.kwargs["markdown_text"])
            for c in client.chat_appendStream.await_args_list
        ]
        ts_a = adapter._active_streams[("C1", 1)]["ts"]
        ts_b = adapter._active_streams[("C1", 2)]["ts"]
        assert (ts_a, "111") in deltas
        assert (ts_b, "222") in deltas


class TestSameTurnSegmentHandoff:
    @pytest.mark.asyncio
    async def test_draft_id_bump_seals_the_superseded_segment(self):
        """Transports that bump draft_id per tool boundary still hand off."""
        adapter, client = _make_adapter()

        await adapter.send_draft("C1", 7, "segment one", metadata=TURN_A)
        await adapter.send_draft("C1", 8, "segment two", metadata=TURN_A)

        client.chat_stopStream.assert_awaited_once()
        assert ("C1", 7) not in adapter._active_streams
        assert ("C1", 8) in adapter._active_streams

    @pytest.mark.asyncio
    async def test_handoff_does_not_seal_a_sibling_turn(self):
        """The other turn's stream must survive a same-turn segment bump."""
        adapter, client = _make_adapter()

        await adapter.send_draft("C1", 1, "sibling turn", metadata=TURN_B)
        await adapter.send_draft("C1", 7, "segment one", metadata=TURN_A)
        sibling_ts = adapter._active_streams[("C1", 1)]["ts"]

        await adapter.send_draft("C1", 8, "segment two", metadata=TURN_A)

        assert ("C1", 1) in adapter._active_streams, (
            "a same-turn segment hand-off sealed another turn's stream"
        )
        sealed = [c.kwargs["ts"] for c in client.chat_stopStream.await_args_list]
        assert sibling_ts not in sealed

    @pytest.mark.asyncio
    async def test_handoff_does_not_seal_a_newer_same_thread_turn(self):
        """A same-thread sibling that started LATER must survive the hand-off.

        The thread anchor alone cannot separate two turns under one parent
        (an interactive reply and a scheduled turn), so the anchor match used
        to seal the sibling and reproduce the split-answer symptom narrowly.
        ``draft_id`` ordering closes that direction: only a provably older
        segment can be this turn's predecessor.
        """
        adapter, client = _make_adapter()

        # Same thread anchor, higher draft_id => a turn that started after us.
        await adapter.send_draft("C1", 9, "newer same-thread turn", metadata=TURN_A)
        newer_ts = adapter._active_streams[("C1", 9)]["ts"]

        await adapter.send_draft("C1", 7, "our segment one", metadata=TURN_A)
        await adapter.send_draft("C1", 8, "our segment two", metadata=TURN_A)

        assert ("C1", 9) in adapter._active_streams, (
            "a hand-off sealed a same-thread turn that started later"
        )
        sealed = [c.kwargs["ts"] for c in client.chat_stopStream.await_args_list]
        assert newer_ts not in sealed
        # Our own predecessor is still handed off.
        assert ("C1", 7) not in adapter._active_streams


class TestAbandonedStreamReaper:
    """Per-turn keys are no longer displaced by the next turn, so a turn that
    never finalizes would hold a live-typing indicator open forever."""

    @pytest.mark.asyncio
    async def test_abandoned_stream_is_sealed_and_evicted(self):
        adapter, client = _make_adapter()

        await adapter.send_draft("C1", 1, "turn that dies here", metadata=TURN_B)
        stale = adapter._active_streams[("C1", 1)]
        stale_ts = stale["ts"]
        # Age it past the threshold rather than sleeping.
        stale["started"] -= adapter._STREAM_ABANDON_SECONDS + 1

        await adapter.send_draft("C1", 2, "a later turn", metadata=TURN_A)

        assert ("C1", 1) not in adapter._active_streams, (
            "abandoned stream survived the sweep; the map is unbounded"
        )
        sealed = [c.kwargs["ts"] for c in client.chat_stopStream.await_args_list]
        assert stale_ts in sealed, "abandoned stream evicted without sealing"

    @pytest.mark.asyncio
    async def test_reaper_leaves_live_streams_alone(self):
        adapter, client = _make_adapter()

        await adapter.send_draft("C1", 1, "live sibling", metadata=TURN_B)
        await adapter.send_draft("C1", 2, "our turn", metadata=TURN_A)

        assert ("C1", 1) in adapter._active_streams
        assert ("C1", 2) in adapter._active_streams
        client.chat_stopStream.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sweep_precedes_the_slot_read_so_no_append_hits_a_dead_ts(self):
        """The sweep runs before this turn's slot is read.

        Order matters: reading the slot first would hand back an entry the
        sweep then seals, and the append would target a stopped ``ts``.  A
        turn aged past the threshold is instead reopened as a fresh stream.
        """
        adapter, client = _make_adapter()

        await adapter.send_draft("C1", 1, "long turn", metadata=TURN_A)
        adapter._active_streams[("C1", 1)]["started"] -= (
            adapter._STREAM_ABANDON_SECONDS + 1
        )

        result = await adapter.send_draft("C1", 1, "long turn continues", metadata=TURN_A)

        assert result.success
        # Reaped, then reopened as a fresh stream for the same turn — the
        # invariant that matters is that no append targets a dead ts.
        assert ("C1", 1) in adapter._active_streams
        client.chat_appendStream.assert_not_awaited()
