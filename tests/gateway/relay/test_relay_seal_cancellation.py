"""Regression: cancellation during the final seal must not orphan the
remote stream (PR 85796 review round 2, finding 4).

_seal_open_draft pops the open entry and writes the local tombstone
BEFORE awaiting transport I/O. CancelledError bypasses the failure
handling (it is not an Exception), so a cancel mid-seal left:

    remote stream: OPEN (live indicator until connector eviction)
    adapter._open_draft_by_chat: {}     <- abandon finds nothing
    adapter._sealed_draft_by_chat: {key: id}  <- premature tombstone

The seal now restores the open entry and drops its premature tombstone
on CancelledError before re-raising, so the consumer's abandon pass can
seal the stream in place.
"""

import asyncio

import pytest

from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class HangOnSealTransport:
    def __init__(self):
        self.ops = []
        self.hang_seal = True

    async def send_outbound(self, payload, platform=None):
        final = bool(payload.get("final"))
        self.ops.append((payload.get("op"), final, str(payload.get("content"))[:25]))
        if payload.get("op") == "draft" and final and self.hang_seal:
            await asyncio.sleep(30)
        return {"success": True, "message_id": "ts.1"}


class TestCancelDuringSeal:
    @pytest.mark.asyncio
    async def test_cancel_mid_seal_restores_open_state(self):
        adapter, _ = _connected_adapter()
        t = HangOnSealTransport()
        adapter._transport = t
        md = {"message_id": "m.1"}
        await adapter.send_draft("C1", 11, "partial", metadata=md)
        key = adapter._draft_key("C1", md)

        task = asyncio.create_task(adapter.send("C1", "complete", metadata=dict(md)))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert adapter._open_draft_by_chat.get(key) == 11, (
            "cancelled seal must restore the open entry so abandon can close it"
        )
        assert key not in adapter._sealed_draft_by_chat, (
            "premature tombstone must not survive a cancelled seal"
        )

    @pytest.mark.asyncio
    async def test_abandon_after_cancelled_seal_closes_remote_stream(self):
        """End-to-end: cancel mid-seal, then the abandon pass (what the
        consumer's CancelledError handler runs) seals the stream."""
        adapter, _ = _connected_adapter()
        t = HangOnSealTransport()
        adapter._transport = t
        md = {"message_id": "m.2"}
        await adapter.send_draft("C1", 12, "partial on screen", metadata=md)

        task = asyncio.create_task(adapter.send("C1", "complete", metadata=dict(md)))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        t.hang_seal = False  # transport recovers for the abandon pass
        r = await adapter.abandon_open_draft(
            "C1", "partial on screen", metadata=dict(md)
        )
        assert r.success
        seals = [o for o in t.ops if o[0] == "draft" and o[1]]
        # First seal attempt hung (cancelled); the abandon's seal landed.
        assert seals[-1][2] == "partial on screen"
        assert not adapter._open_draft_by_chat
