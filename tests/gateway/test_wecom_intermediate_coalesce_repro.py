"""Reproduction + contract: intermediate stream frames must COALESCE (not drop)
while a prior frame's ack is pending.

The fork's golden ``_send_stream_reply`` used serial-with-coalesce for
intermediate (``finish=False``) frames: when an ack was still pending for a
prior frame, the new (cumulative) content was buffered and flushed the moment
the pending ack resolved — so the latest content ALWAYS reached the wire.

The upstream ``skip_if_pending`` machinery the migration kept instead DROPS the
frame (``{"skipped": True}``) with no buffer and no flush-on-ack.  Under any ack
lag — routine on long replies (server queue lag, WS jitter, concurrent replies)
— a burst of cumulative frames all land while the first ack is still pending, so
every one after the first is silently dropped and the visible bubble FREEZES at
stale content until the next frame that happens to catch an idle ack (screenshot
1: "旧气泡停在 … 空白").  When a Layer 2 rotation then seals that bubble, whatever
was frozen is what the user is left with.

These tests fake only the ``_send_json``/ack seam so the REAL
``_send_stream_reply`` intermediate path runs, and drive frames faster than the
simulated ack lag so the pending-ack pileup is real (not a synchronous mock).
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wecom.adapter import WeComAdapter


CHAT = "chat-coalesce"
REQ = "req-coalesce"
TURN_ID = "turn-coalesce"


class _AckControlWS:
    """Fake websocket that records frames and lets the test resolve acks on
    demand, so intermediate frames pile up as pending (the drop/coalesce path)."""

    def __init__(self) -> None:
        self.closed = False
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True

    def stream_contents(self) -> list[str]:
        """Non-seed stream-frame contents that reached the wire, in order."""
        out = []
        for p in self.sent:
            stream = p.get("body", {}).get("stream", {})
            content = stream.get("content", "")
            if content and content != "<think></think>":
                out.append(content)
        return out


def _make_adapter() -> tuple[WeComAdapter, _AckControlWS]:
    adapter = WeComAdapter(PlatformConfig(enabled=True, extra={"stream_keepalive_enabled": False}))
    adapter._last_chat_req_ids[CHAT] = REQ
    ws = _AckControlWS()
    adapter._ws = ws
    return adapter, ws


@pytest.mark.asyncio
async def test_pending_ack_intermediate_coalesces_latest_to_wire():
    """A cumulative frame that arrives while a prior ack is pending must reach
    the wire once that ack resolves — not be silently dropped."""
    adapter, ws = _make_adapter()
    try:
        # Seed the turn (registers the seed frame's pending ack — unresolved).
        await adapter._send_stream_frame_inner("", chat=CHAT, finalize=False, turn_id=TURN_ID)
        # Two cumulative body frames arrive while the seed ack is still pending.
        await adapter._send_stream_frame_inner("AAAA", chat=CHAT, finalize=False, turn_id=TURN_ID)
        await adapter._send_stream_frame_inner("AAAABBBB", chat=CHAT, finalize=False,
                                               turn_id=TURN_ID)
        # The pending (seed) ack now resolves — the buffered latest content must
        # flush to the wire.
        adapter._resolve_reply_ack(REQ, {"body": {"errcode": 0}})
        await asyncio.sleep(0)  # let any scheduled flush task run
        await asyncio.sleep(0.01)

        contents = ws.stream_contents()
        assert "AAAABBBB" in contents, (
            "the latest cumulative content must reach the wire after the pending "
            f"ack resolves (coalesce), not be dropped; wire={contents!r}"
        )
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_bubble_does_not_freeze_under_ack_lag():
    """Under continuous ack lag, the visible bubble must keep advancing: the
    LAST content on the wire must be the full cumulative body, never a stale
    early prefix frozen by dropped intermediates."""
    adapter, ws = _make_adapter()
    try:
        await adapter._send_stream_frame_inner("", chat=CHAT, finalize=False, turn_id=TURN_ID)
        # A burst of cumulative frames, each arriving before any ack resolves.
        cumulative = ""
        for word in ("The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ",
                     "dog."):
            cumulative += word
            await adapter._send_stream_frame_inner(cumulative, chat=CHAT, finalize=False,
                                                   turn_id=TURN_ID)
        # Acks drain (resolve repeatedly until the queue is idle), flushing the
        # coalesced tail.
        for _ in range(20):
            if not adapter._resolve_reply_ack(REQ, {"body": {"errcode": 0}}):
                break
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)

        contents = ws.stream_contents()
        assert contents, "at least one body frame must reach the wire"
        assert contents[-1] == cumulative, (
            "the bubble froze on a stale prefix — the final cumulative body never "
            f"reached the wire under ack lag; last wire content={contents[-1]!r}"
        )
    finally:
        await adapter.disconnect()
