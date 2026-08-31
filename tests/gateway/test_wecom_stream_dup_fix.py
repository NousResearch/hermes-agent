"""Behavior-contract tests for the WeCom native-streaming "重复气泡" fix.

Two production bug classes are covered, each toggle-validated so the test
reproduces the duplicate/mis-decline when the fix is disabled and passes when
it is enabled — proving the assertions track behavior, not a frozen snapshot.

Fix A — Layer 2 clock fallback must NOT decline a finalize frame while Layer 1
keep-alive is enabled.  Keep-alive refreshes the stream window every ~2min, so
a large ``stream_age`` does not mean the stream is dead; declining it on a blind
clock read forces the consumer's send() fallback to re-deliver content the
intermediate frames already put on screen (the duplicate bubble).  Toggle =
``adapter._stream_keepalive_enabled``.

Fix B — an intermediate frame (finalize=False) failing/expiring must be
fire-and-forget (return True, turn stays live, no consumer fallback), because a
later cumulative frame overwrites it.  Only a FINAL frame (finalize=True)
failure means the screen is genuinely missing content and must return False to
trip the consumer's send() fallback.  Toggle = the ``finalize`` argument.

These drive the REAL ``WeComAdapter._send_stream_frame_inner`` with only the
byte-level ``_send_stream_reply`` seam faked, so the actual finalize / except
control flow runs.  Assertions read observable adapter state: the return value
(what the consumer keys its fallback on), whether the turn survived in
``_stream_turns``, whether the chat was marked expired, and how many finalize
frames actually reached ``_send_stream_reply``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wecom.adapter import (
    WeComAdapter,
    WeComStreamExpiredError,
    STREAM_EXPIRED_ERRCODE,
)


CHAT_ID = "chat-dup"
REQ_ID = "req-dup"
TURN_ID = "turn-dup"

# Fire-and-forget intermediate frames are pushed as soon as the cumulative
# text differs from the last sent frame (pure identity-dedup, no chunker /
# min-chars gate). A non-trivial body is used so the intermediate-failure
# tests exercise a real content frame rather than only the seed frame.
BLOCK_TEXT = (
    "This is a complete sentence used to fill the block chunker past its "
    "minimum character threshold so it actually drains a content frame. "
    "Here is a second sentence to be safely over the limit."
)


def _make_adapter(*, keepalive_enabled: bool) -> WeComAdapter:
    """Real adapter with only the stream byte-writer faked.

    ``_send_stream_reply`` is the seam between per-turn logic and the wire.
    Faking it here lets each test dictate per-frame success/expiry while the
    real finalize / except branches run.
    """
    extra = {"stream_keepalive_enabled": keepalive_enabled}
    adapter = WeComAdapter(PlatformConfig(enabled=True, extra=extra))
    adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
    return adapter


def _finalize_calls(mock: AsyncMock) -> list:
    """finish=True calls that reached ``_send_stream_reply``."""
    return [c for c in mock.await_args_list if c.kwargs.get("finish") is True]


# ===========================================================================
# Fix A — keep-alive suppresses the Layer 2 clock decline
# ===========================================================================


class TestKeepaliveSuppressesClockDecline:
    """A long-lived, keep-alive-refreshed stream must still finalize natively;
    the clock fallback must not decline it and force a duplicate send()."""

    @pytest.mark.asyncio
    async def test_keepalive_on_old_stream_finalizes_natively(self):
        """FIX ENABLED: keep-alive on + stream_age >> safe_duration + finalize.

        Post-fix contract: the finalize frame is sent on the wire (finish=True),
        finalize returns True, the turn is finalized+cleaned, and the chat is
        NOT marked expired — so the consumer suppresses its send() fallback and
        no duplicate bubble is produced.
        """
        adapter = _make_adapter(keepalive_enabled=True)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            # Open the turn (seed + first intermediate).
            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]

            # Age the stream far beyond the Layer 2 safe duration.
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            ok = await adapter._send_stream_frame_inner(
                "the complete final answer",
                chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )

            # Native finalize succeeded — consumer will suppress fallback.
            assert ok is True
            assert len(_finalize_calls(reply)) == 1, (
                "keep-alive on: finalize frame must reach the wire, not be "
                "declined by the Layer 2 clock fallback"
            )
            assert CHAT_ID not in adapter._stream_expired_chats
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_keepalive_off_old_stream_declines_finalize(self):
        """FIX DISABLED (toggle): keep-alive off preserves original Layer 2.

        With keep-alive off there is nothing refreshing the window, so a
        stream older than safe_duration is genuinely doomed — the original
        clock decline must still fire: no finalize frame on the wire, return
        False, turn retired, chat marked expired (consumer takes over via
        send()).  This proves Fix A is gated on the toggle, not unconditional.
        """
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            ok = await adapter._send_stream_frame_inner(
                "the complete final answer",
                chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )

            assert ok is False, "keep-alive off: old stream must decline finalize"
            assert len(_finalize_calls(reply)) == 0, (
                "keep-alive off: no finalize frame should reach the wire"
            )
            assert CHAT_ID in adapter._stream_expired_chats
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_keepalive_on_truly_expired_stream_falls_back(self):
        """Even with the clock decline skipped, a genuinely dead stream is safe.

        Keep-alive on skips the blind clock check, but if the stream really has
        expired the finalize ``_send_stream_reply(finish=True)`` hits 846608 and
        raises ``WeComStreamExpiredError`` — the existing except path then
        retires the turn and returns False for the (correct, finalize-only)
        fallback.  This shows Fix A does not lose the real-expiry safety net.
        """
        adapter = _make_adapter(keepalive_enabled=True)
        try:
            async def _reply(req_id, stream_id, content, finish=False):
                if finish:
                    raise WeComStreamExpiredError(errcode=STREAM_EXPIRED_ERRCODE)
                return {"errcode": 0}

            adapter._send_stream_reply = AsyncMock(side_effect=_reply)

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            ok = await adapter._send_stream_frame_inner(
                "the complete final answer",
                chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )

            assert ok is False, "real 846608 on finalize must fall back"
            assert CHAT_ID in adapter._stream_expired_chats
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()


# ===========================================================================
# Fix B — intermediate failures are fire-and-forget; only final falls back
# ===========================================================================


class TestIntermediateFrameFailureIsFireAndForget:
    """A single intermediate frame failing must not trip the consumer fallback
    or kill the turn; only a final-frame failure does."""

    @pytest.mark.asyncio
    async def test_intermediate_expired_returns_true_keeps_turn(self):
        """FIX ENABLED: intermediate finish=False hits 846608.

        Contract: return True (no fallback), turn survives in the registry
        (keep-alive keeps refreshing), chat NOT marked expired.  A later
        cumulative frame will overwrite the dropped one.
        """
        adapter = _make_adapter(keepalive_enabled=True)
        try:
            # Seed succeeds; the next intermediate content frame expires.
            calls = {"n": 0}

            async def _reply(req_id, stream_id, content, finish=False):
                calls["n"] += 1
                # First call is the seed ("<think></think>"); let it succeed so
                # the turn opens, then expire the real content frame.
                if calls["n"] >= 2 and not finish:
                    raise WeComStreamExpiredError(errcode=STREAM_EXPIRED_ERRCODE)
                return {"errcode": 0}

            adapter._send_stream_reply = AsyncMock(side_effect=_reply)

            # Send a non-trivial body so a real content frame is drained
            # (fire-and-forget: any content differing from the last frame).
            ok = await adapter._send_stream_frame_inner(
                BLOCK_TEXT,
                chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )

            assert ok is True, "intermediate expiry must be fire-and-forget"
            assert f"{CHAT_ID}:{TURN_ID}" in adapter._stream_turns, (
                "intermediate failure must NOT retire the turn — keep-alive is "
                "still refreshing the live stream"
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            assert turn.expired is False
            assert CHAT_ID not in adapter._stream_expired_chats
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_intermediate_generic_exception_returns_true_keeps_turn(self):
        """Same fire-and-forget contract for a generic (non-expiry) exception on
        an intermediate frame — the whole except class is fixed, not just the
        WeComStreamExpiredError path."""
        adapter = _make_adapter(keepalive_enabled=True)
        try:
            calls = {"n": 0}

            async def _reply(req_id, stream_id, content, finish=False):
                calls["n"] += 1
                if calls["n"] >= 2 and not finish:
                    raise RuntimeError("transient wire error")
                return {"errcode": 0}

            adapter._send_stream_reply = AsyncMock(side_effect=_reply)

            ok = await adapter._send_stream_frame_inner(
                BLOCK_TEXT,
                chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )

            assert ok is True
            assert f"{CHAT_ID}:{TURN_ID}" in adapter._stream_turns
            assert CHAT_ID not in adapter._stream_expired_chats
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_final_expired_returns_false_falls_back(self):
        """FIX-INVARIANT (toggle via finalize flag): a final finish=True frame
        that expires MUST return False, retire the turn, and mark the chat
        expired — the screen is genuinely missing this content, so the consumer
        must run its send() fallback.  Contrast with the intermediate case above
        proves the fix discriminates on finalize, not blanket-swallows."""
        adapter = _make_adapter(keepalive_enabled=True)
        try:
            async def _reply(req_id, stream_id, content, finish=False):
                if finish:
                    raise WeComStreamExpiredError(errcode=STREAM_EXPIRED_ERRCODE)
                return {"errcode": 0}

            adapter._send_stream_reply = AsyncMock(side_effect=_reply)

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )

            ok = await adapter._send_stream_frame_inner(
                "the complete final answer",
                chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )

            assert ok is False, "final-frame expiry MUST fall back (return False)"
            assert CHAT_ID in adapter._stream_expired_chats
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_final_generic_exception_returns_false_retires(self):
        """A generic exception on the final frame also returns False + retires
        the turn (consumer fallback)."""
        adapter = _make_adapter(keepalive_enabled=True)
        try:
            async def _reply(req_id, stream_id, content, finish=False):
                if finish:
                    raise RuntimeError("wire down on finalize")
                return {"errcode": 0}

            adapter._send_stream_reply = AsyncMock(side_effect=_reply)

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )

            ok = await adapter._send_stream_frame_inner(
                "the complete final answer",
                chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )

            assert ok is False
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()
