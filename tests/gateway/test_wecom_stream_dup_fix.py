"""Behavior-contract tests for the WeCom native-streaming "重复气泡" fix.

Two production bug classes are covered, each toggle-validated so the test
reproduces the duplicate/mis-decline when the fix is disabled and passes when
it is enabled — proving the assertions track behavior, not a frozen snapshot.

Fix A — Layer 2 must NOT decline/rotate a finalize frame while Layer 1
keep-alive is enabled.  Keep-alive keeps the bubble live, so a large
``stream_age`` is trusted and the finalize lands natively on the SAME stream;
with keep-alive OFF the stream is instead ROTATED near the 10-min wall (old
bubble sealed with finish=true, a fresh stream_id opened on the same req_id,
finalize landed on the new bubble) so the answer is never dropped.  Toggle =
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
    ROTATION_CONTINUATION_SUFFIX,
)
from plugins.platforms.wecom.stream_types import StreamSendOutcome


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

        Post-fix contract (rotation×keep-alive now coexist): an over-age stream
        can no longer be finalized in place because it may be past the 10-min
        wall.  Layer 2 rotation seals the old bubble ("partial answer…⏬⏬⏬",
        finish=True) and finalizes the real answer on a FRESH stream ("the
        complete final answer", finish=True) — two finish frames, both on the
        wire, crossing the wall.  finalize still returns True, the turn is
        finalized+cleaned, and the chat is NOT marked expired, so the consumer
        suppresses its send() fallback and no duplicate bubble is produced.
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
            assert ok
            # Rotation×keep-alive: over-age stream is sealed on its old bubble,
            # then the answer is finalized on a fresh stream — two finish frames.
            fcalls = _finalize_calls(reply)
            assert len(fcalls) == 2, (
                "keep-alive on + over-age: rotation must seal the old bubble and "
                "finalize on a fresh stream (2 finish frames), crossing the wall"
            )
            # First finish frame: the sealed old bubble (partial + continuation).
            assert fcalls[0].args[2].endswith(ROTATION_CONTINUATION_SUFFIX)
            assert "partial answer" in fcalls[0].args[2]
            # Second finish frame: the real answer on a DIFFERENT (fresh) stream.
            assert fcalls[1].args[2] == "the complete final answer"
            assert fcalls[0].args[1] != fcalls[1].args[1]  # distinct stream ids
            assert CHAT_ID not in adapter._stream_expired_chats
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_keepalive_off_old_stream_rotates_finalize(self):
        """FIX DISABLED (toggle): keep-alive off triggers Layer 2 ROTATION.

        With keep-alive off there is nothing keeping the bubble live, so a
        stream approaching the 10-min wall is rotated rather than declined:
        the old bubble is sealed with finish=true (still inside the window),
        a fresh stream_id is opened on the same req_id, and the finalize lands
        on the NEW bubble.  Post-rotation contract: finalize returns True, the
        chat is NOT marked expired, the turn is finalized+cleaned, and TWO
        finish=true frames reached the wire (the rotation close + the finalize).
        This proves rotation replaced the old blind-decline behavior while
        staying gated on the keep-alive toggle.
        """
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            old_stream_id = turn.stream_id
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            ok = await adapter._send_stream_frame_inner(
                "the complete final answer",
                chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )

            assert ok, "keep-alive off: old stream must rotate + finalize, not decline"
            finish_calls = _finalize_calls(reply)
            assert len(finish_calls) == 2, (
                "rotation must seal the old bubble (finish=true) AND land the "
                "finalize on the new bubble (finish=true) — two wire frames"
            )
            # The rotation close targets the OLD stream_id; the finalize targets
            # the NEW one.  Confirm the two finish frames used different streams.
            finish_stream_ids = {c.args[1] for c in finish_calls}
            assert old_stream_id in finish_stream_ids
            assert len(finish_stream_ids) == 2, "finalize must land on a fresh stream_id"
            assert CHAT_ID not in adapter._stream_expired_chats
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_keepalive_off_rotation_close_expired_falls_back(self):
        """If the old stream is ALREADY dead when rotation tries to seal it, the
        rotation close hits 846608 → the turn is retired and finalize returns
        False so the consumer's send() fallback delivers the content.  This is
        the safety net for the case where the 9.5-min trigger lost the race to
        the 10-min wall."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            async def _reply(req_id, stream_id, content, finish=False):
                # Seed (finish=False) succeeds; the first finish=true (the
                # rotation close) hits the wall.
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

            assert not ok, "rotation close hitting the wall must fall back"
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

            assert not ok, "real 846608 on finalize must fall back"
            assert CHAT_ID in adapter._stream_expired_chats
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()


# ===========================================================================
# Layer 2 rotation on an INTERMEDIATE frame (mid-stream, not finalize)
# ===========================================================================


class TestIntermediateFrameRotation:
    """A mid-stream (finalize=False) frame that crosses the safe duration must
    rotate transparently: seal the old bubble, open a fresh one, and push the
    content there — the turn stays live so streaming (and the tool timer)
    continues in the new bubble."""

    @pytest.mark.asyncio
    async def test_intermediate_rotates_to_fresh_bubble_and_streams_on(self):
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            # Open the turn (seed + first intermediate).
            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            old_stream_id = turn.stream_id
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            # A further intermediate frame crosses the wall → rotation.
            ok = await adapter._send_stream_frame_inner(
                "partial answer, now longer",
                chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )

            assert ok, "intermediate rotation must succeed and keep streaming"
            # Turn survived (still live) but on a NEW stream_id.
            assert f"{CHAT_ID}:{TURN_ID}" in adapter._stream_turns
            assert turn.stream_id != old_stream_id, "must be a fresh bubble"
            assert turn.expired is False
            assert CHAT_ID not in adapter._stream_expired_chats
            # Exactly one finish=true reached the wire — the rotation close on
            # the OLD stream; the intermediate content itself is finish=false.
            finish_calls = _finalize_calls(reply)
            assert len(finish_calls) == 1
            assert finish_calls[0].args[1] == old_stream_id
            # The clock reset, so the new bubble is young again.
            assert (
                __import__("time").monotonic() - turn.start_time
                < adapter._stream_safe_duration_seconds
            )
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_first_frame_never_rotates(self):
        """The very first frame (unseeded, age ~0) must never trip rotation —
        rotation is gated on ``turn.seeded`` so a brand-new turn just seeds."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            ok = await adapter._send_stream_frame_inner(
                "hello", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )

            assert ok
            # Only the seed + content frame — no finish=true rotation close.
            assert len(_finalize_calls(reply)) == 0
        finally:
            await adapter.disconnect()


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

            assert ok, "intermediate expiry must be fire-and-forget"
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

            assert ok
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

            assert not ok, "final-frame expiry MUST fall back (return False)"
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

            assert not ok
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()


class TestActiveRotationTimer:
    """Layer 2 ACTIVE timer: rotation must fire from the periodic clock check
    even when NO frames are flowing (long tool calls), lead the deadline
    instead of lagging it, protect every new bubble, and be race-safe against
    the frame-send path."""

    @pytest.mark.asyncio
    async def test_active_timer_rotates_without_frames(self):
        """Defect B/core: _rotation_check_execute seals the old bubble and
        rotates to a fresh one with no intervening frame push, and the new
        bubble is protected (rotation check re-armed on next seed)."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            old_stream_id = turn.stream_id
            # Age past the safe wall so the active timer decides to rotate.
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            # Drive the active-timer coroutine directly (no frames pushing).
            await adapter._rotation_check_execute(turn, TURN_ID)

            assert turn.stream_id != old_stream_id, "active timer must rotate"
            assert turn.seeded is False, "new bubble not seeded until next frame"
            assert turn.expired is False
            assert CHAT_ID not in adapter._stream_expired_chats
            finish_calls = _finalize_calls(reply)
            assert len(finish_calls) == 1
            assert finish_calls[0].args[1] == old_stream_id
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_pure_tool_tail_finalize_reseeds_and_protects_new_bubble(self):
        """Defect B: after an ACTIVE rotation the tail is a pure finalize with
        no intermediate text frame.  The finalize must re-seed the fresh
        (un-seeded) bubble AND arm the active rotation check before sealing —
        i.e. the new bubble is opened and closed, never stranded."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            old_stream_id = turn.stream_id
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            # Active rotation with NO following intermediate frame.
            await adapter._rotation_check_execute(turn, TURN_ID)
            new_stream_id = turn.stream_id
            assert new_stream_id != old_stream_id
            assert turn.seeded is False

            # Pure-tool-tail: the very next event is the finalize.
            reply.reset_mock()
            armed_seen = {"v": False}
            orig_arm = adapter._arm_rotation_check

            def _spy_arm(t, *, turn_id):
                armed_seen["v"] = True
                return orig_arm(t, turn_id=turn_id)

            adapter._arm_rotation_check = _spy_arm

            ok = await adapter._send_stream_frame_inner(
                "the complete final answer",
                chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )

            assert ok, "finalize on the rotated fresh bubble must succeed"
            # Finalize re-seeded the new bubble (finish=false seed) AND armed
            # the active rotation check for it, then sealed it (finish=true).
            assert armed_seen["v"], (
                "new bubble must be armed for rotation before finalize seals it"
            )
            seed_frames = [
                c for c in reply.await_args_list
                if c.kwargs.get("finish") is False
                and c.args[1] == new_stream_id
            ]
            assert seed_frames, "finalize must re-seed the fresh rotated bubble"
            finish_frames = [
                c for c in reply.await_args_list
                if c.kwargs.get("finish") is True
                and c.args[1] == new_stream_id
            ]
            assert finish_frames, "finalize must seal the fresh rotated bubble"
            assert f"{CHAT_ID}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_arm_delay_leads_deadline_never_lags(self):
        """Defect A: the armed timer must wake BEFORE the rotation threshold
        (safe_duration - lead), never after it — no +epsilon overshoot."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]

            import time as _t
            rotate_at = (
                adapter._stream_safe_duration_seconds
                - adapter._rotation_lead_seconds
            )
            # Put the stream 10s before the rotation threshold.
            turn.start_time = _t.monotonic() - (rotate_at - 10.0)

            captured = {}
            real_call_later = asyncio.get_running_loop().call_later

            def _spy_call_later(delay, cb, *args):
                captured["delay"] = delay
                return real_call_later(delay, cb, *args)

            asyncio.get_running_loop().call_later = _spy_call_later
            try:
                adapter._cancel_rotation_check(turn)
                adapter._arm_rotation_check(turn, turn_id=TURN_ID)
            finally:
                asyncio.get_running_loop().call_later = real_call_later
                adapter._cancel_rotation_check(turn)

            # Wake must land AT or BEFORE the threshold (delay <= remaining=10),
            # not after it (the old bug scheduled remaining+0.5 = 10.5).
            assert captured["delay"] <= 10.0 + 1e-9, (
                f"timer must lead the deadline; got delay={captured['delay']}"
            )
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_concurrent_rotation_and_framesend_no_double_loss(self):
        """Defect C: the active-timer rotation and a concurrent frame-send are
        serialized by the per-turn rotation lock, so exactly ONE rotation runs
        and no old-bubble-unsealed + new-bubble-uncreated double-loss occurs."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                "partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            old_stream_id = turn.stream_id
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            # Fire the active timer coroutine and a frame-send at the same time.
            await asyncio.gather(
                adapter._rotation_check_execute(turn, TURN_ID),
                adapter._send_stream_frame_inner(
                    "more content", chat=CHAT_ID, finalize=False,
                    turn_id=TURN_ID,
                ),
            )

            # The lock must have serialized them: exactly ONE seal (finish=true)
            # of the ORIGINAL bubble reached the wire — not zero (both saw a
            # stale seeded=True and neither sealed) and not two (double seal).
            seals_of_old = [
                c for c in _finalize_calls(reply)
                if c.args[1] == old_stream_id
            ]
            assert len(seals_of_old) == 1, (
                f"exactly one rotation of the old bubble expected, got "
                f"{len(seals_of_old)} — lock failed to serialize"
            )
            assert turn.stream_id != old_stream_id
            assert turn.expired is False
            assert CHAT_ID not in adapter._stream_expired_chats
        finally:
            await adapter.disconnect()


class TestRotationSplitSignal:
    """Layer 2 rotation must (1) seal the OLD bubble with a continuation divider
    and (2) report the rotation back to the gateway via StreamSendOutcome so the
    fresh bubble can carry only incremental text (no prefix repeat)."""

    @pytest.mark.asyncio
    async def test_seal_appends_continuation_divider(self):
        """The finish=true seal of the old bubble ends with the continuation
        suffix, so the sealed bubble reads as 'to be continued'."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                "AAAA", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            old_stream_id = turn.stream_id
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            # Next frame crosses the wall → passive rotation seals the old bubble.
            await adapter._send_stream_frame_inner(
                "AAAABBBB", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )

            seals = [c for c in _finalize_calls(reply) if c.args[1] == old_stream_id]
            assert len(seals) == 1, "exactly one seal of the old bubble"
            close_text = seals[0].args[2]
            assert close_text.endswith(ROTATION_CONTINUATION_SUFFIX), (
                f"seal must end with continuation divider; got {close_text!r}"
            )
            # The sealed content still contains the old bubble's body.
            assert "AAAA" in close_text
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_passive_rotation_reports_rotated_in_outcome(self):
        """A frame that triggers passive rotation returns StreamSendOutcome
        with rotated=True (synchronous, same call)."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            first = await adapter._send_stream_frame_inner(
                "AAAA", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            assert isinstance(first, StreamSendOutcome)
            assert first.rotated is False, "no rotation on the opening frame"

            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            out = await adapter._send_stream_frame_inner(
                "AAAABBBB", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            assert isinstance(out, StreamSendOutcome)
            assert out.rotated is True, "passive rotation must report rotated=True"
            assert bool(out) is True, "outcome must stay truthy (frame delivered)"
            # Flag is drained — not re-reported on the next frame.
            assert turn.pending_rotation_signal is False
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_active_rotation_reports_rotated_on_next_frame(self):
        """An ACTIVE-timer rotation (no frame in flight) leaves the signal on
        the turn; the NEXT frame's outcome reports rotated=True (deferred)."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                "AAAA", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            # Active timer rotates with NO frame-send in flight.
            await adapter._rotation_check_execute(turn, TURN_ID)
            assert turn.pending_rotation_signal is True, (
                "active rotation leaves the signal pending for the next frame"
            )

            # The next frame carries the deferred signal.
            out = await adapter._send_stream_frame_inner(
                "AAAABBBB", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            assert out.rotated is True, "deferred rotation reported on next frame"
            assert turn.pending_rotation_signal is False, "drained after report"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_midflight_seal_reports_seal_len_of_advanced_body(self):
        """When the active timer seals DURING a frame's body send — AFTER
        ``turn.accumulated_text`` was advanced to that frame's grown body — the
        reported outcome's ``seal_len`` must equal ``len(accumulated_text)`` at
        seal (the grown body), NOT the previous body length.  This is the relative
        seal point the gateway needs to split at.

        Deterministic: the fake ``_send_stream_reply`` performs the seal itself on
        the target content-frame's finish=False invocation (the ``sealed`` guard
        stops the re-entrant finish=true close from re-sealing)."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            state = {"sealed": False, "seal_len_at_seal": None}

            async def _reply(req_id, stream_id, content, finish=False, **kw):
                if not finish and content == "AAAABBBB" and not state["sealed"]:
                    state["sealed"] = True
                    t = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
                    # accumulated_text was advanced to the grown body before this send.
                    assert t.accumulated_text == "AAAABBBB"
                    await adapter._rotate_stream_locked(t, TURN_ID)
                    # _rotate_stream_locked records the seal length off accumulated_text.
                    state["seal_len_at_seal"] = t.rotation_seal_body_len
                return {"errcode": 0}
            adapter._send_stream_reply = AsyncMock(side_effect=_reply)

            await adapter._send_stream_frame_inner(
                "AAAA", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            # Active band [555, 570): passive up-front check (>= 570) must not fire.
            turn.start_time = __import__("time").monotonic() - 560.0

            out = await adapter._send_stream_frame_inner(
                "AAAABBBB", chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )

            assert state["sealed"], "the mid-flight seal must have fired"
            assert state["seal_len_at_seal"] == 8, (
                "the seal recorded the grown body length (len('AAAABBBB'))"
            )
            assert out.rotated is True, "the same frame reports the rotation"
            assert out.seal_len == 8, (
                "outcome.seal_len must be the relative body length the rotation "
                f"sealed with (== len(accumulated_text) at seal = 8); got {out.seal_len}"
            )
        finally:
            await adapter.disconnect()


class TestRotationNoLossEndToEnd:
    """The check every earlier attempt missed: concatenating each bubble's final
    frame (minus the continuation divider) MUST equal the complete _accumulated —
    no dropped segment, no repeated prefix.  Drives the gateway's real
    _send_or_edit into the REAL adapter (only _send_stream_reply faked), for both
    the passive and active-timer rotation paths.  These are the guards that would
    have caught the silent-loss regression.
    """

    def _make_gateway_consumer(self, adapter):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
        cfg = StreamConsumerConfig(chat_type="dm", cursor="")
        c = GatewayStreamConsumer(adapter, CHAT_ID, cfg)
        c._use_native_streaming = True
        c._native_stream_opened = True
        c._turn_id = TURN_ID
        c._initial_reply_to_id = None
        return c

    def _concat_bubbles(self, calls):
        """Each stream_id's final non-seed frame, divider stripped, concatenated
        in bubble order — i.e. what the user ultimately sees across bubbles."""
        from collections import OrderedDict
        last = OrderedDict()
        for stream_id, finish, content in calls:
            if content == "<think></think>":
                continue
            last[stream_id] = content
        return "".join(
            ct.split(ROTATION_CONTINUATION_SUFFIX)[0].replace("​", "")
            for ct in last.values()
        )

    @pytest.mark.asyncio
    async def test_passive_rotation_concat_no_loss_no_repeat(self):
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            calls = []

            async def rec(req_id, stream_id, content, finish=False, **kw):
                calls.append((stream_id, finish, content))
                return {"errcode": 0}
            adapter._send_stream_reply = rec

            c = self._make_gateway_consumer(adapter)
            c._accumulated = "AAAA"
            await c._send_or_edit("AAAA", finalize=False)
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            turn.start_time -= adapter._stream_safe_duration_seconds + 500
            c._accumulated = "AAAABBBB"        # this frame trips rotation
            await c._send_or_edit("AAAABBBB", finalize=False)
            c._accumulated = "AAAABBBBCCCC"    # next delta on the fresh bubble
            await c._send_or_edit("AAAABBBBCCCC", finalize=False)

            seen = self._concat_bubbles(calls)
            assert seen == "AAAABBBBCCCC", (
                f"concat across bubbles must equal the full response (no loss, "
                f"no repeat); got {seen!r}"
            )
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_active_rotation_concat_no_loss_no_repeat(self):
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            calls = []

            async def rec(req_id, stream_id, content, finish=False, **kw):
                calls.append((stream_id, finish, content))
                return {"errcode": 0}
            adapter._send_stream_reply = rec

            c = self._make_gateway_consumer(adapter)
            c._accumulated = "AAAA"
            await c._send_or_edit("AAAA", finalize=False)
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            turn.start_time -= adapter._stream_safe_duration_seconds + 500
            # Active-timer rotation with no frame in flight.
            await adapter._rotation_check_execute(turn, TURN_ID)
            c._accumulated = "AAAABBBB"        # first frame re-seeds fresh bubble
            await c._send_or_edit("AAAABBBB", finalize=False)
            c._accumulated = "AAAABBBBCCCC"
            await c._send_or_edit("AAAABBBBCCCC", finalize=False)

            seen = self._concat_bubbles(calls)
            assert seen == "AAAABBBBCCCC", (
                f"active-path concat must equal the full response; got {seen!r}"
            )
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_active_midflight_seal_advances_offset_to_real_seal_point(self):
        """Regression for the mid-flight active-timer seal (bug #96942-followup).

        The active timer fires in the band [safe-lead, safe) — 555s <= age < 570s
        — i.e. BELOW the passive up-front check (>= 570 at streaming.py L716), so
        the frame does NOT passive-rotate up front.  It proceeds to set
        ``turn.accumulated_text`` to THIS frame's grown body (L2), and then, DURING
        the finish=False body send, the active timer seals the OLD bubble with that
        already-advanced body — recording ``rotation_seal_body_len = L2`` and
        ``pending_rotation_signal = True``.  The SAME frame then reports
        ``rotated=True`` to the gateway.

        The bug: the gateway assumed the seal point equals the PREVIOUS committed
        length (L1) and advanced its split offset to L1 — so every later frame
        re-sliced ``_accumulated[L1:]`` and re-emitted the sealed tail
        ``_accumulated[L1:L2]`` on the fresh bubble (repeated "BBBB").

        The fix surfaces the adapter's real relative ``seal_len`` (== L2) so the
        gateway advances the offset to L2 — the fresh bubble carries only the
        post-seal tail, no repeat.

        Deterministic (no wall clock): the fake ``_send_stream_reply`` performs the
        seal itself on the target content frame's finish=False invocation, exactly
        mimicking the active timer sealing mid-await after accumulated_text was set.
        """
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            calls = []
            state = {"sealed": False}

            async def rec(req_id, stream_id, content, finish=False, **kw):
                calls.append((stream_id, finish, content))
                # On the target intermediate body send (finish=False, the grown
                # "AAAABBBB" body), seal mid-await — like the active timer running
                # concurrently AFTER turn.accumulated_text was advanced to "AAAABBBB".
                # _rotate_stream_locked itself calls _send_stream_reply(finish=True)
                # for the seal close; the `sealed` guard keeps that re-entrant call
                # from recursing into another seal.
                if (not finish and content == "AAAABBBB" and not state["sealed"]):
                    state["sealed"] = True
                    turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
                    await adapter._rotate_stream_locked(turn, TURN_ID)
                return {"errcode": 0}
            adapter._send_stream_reply = rec

            c = self._make_gateway_consumer(adapter)

            # Frame 1: seed + "AAAA".  offset=0, committed=4.
            c._accumulated = "AAAA"
            await c._send_or_edit("AAAA", finalize=False)
            assert c._native_committed_len == 4
            assert c._native_split_offset == 0

            # Age into the active band [555, 570): the passive up-front check
            # (>= 570) must NOT fire, so the next frame advances accumulated_text
            # to "AAAABBBB" BEFORE the mid-flight seal happens.
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            turn.start_time = __import__("time").monotonic() - 560.0

            # Frame 2: "AAAABBBB" — the mid-flight seal fires inside its body send.
            c._accumulated = "AAAABBBB"
            await c._send_or_edit("AAAABBBB", finalize=False)

            # The offset must advance to the REAL seal point L2 (=8), not the stale
            # previous committed length L1 (=4).
            assert c._native_split_offset == 8, (
                "mid-flight active seal advanced accumulated_text to the grown body "
                "before sealing — the gateway must split at that real seal point (8), "
                f"not the stale previous committed length (4); got "
                f"{c._native_split_offset}"
            )

            # Frame 3: next delta on the fresh bubble — must carry only "CCCC".
            c._accumulated = "AAAABBBBCCCC"
            await c._send_or_edit("AAAABBBBCCCC", finalize=False)

            seen = self._concat_bubbles(calls)
            assert seen == "AAAABBBBCCCC", (
                f"mid-flight-seal concat across bubbles must equal the full response "
                f"(no repeat, no loss); got {seen!r}"
            )
            # The sealed old-bubble frame carried the full grown body "AAAABBBB".
            seal_frames = [c2 for c2 in calls if c2[1] is True]
            assert seal_frames, "expected a finish=true seal of the old bubble"
            assert "AAAABBBB" in seal_frames[0][2], (
                f"sealed old bubble must carry the grown body; got {seal_frames[0][2]!r}"
            )
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_toggle_wrong_offset_would_lose_segment(self):
        """Toggle guard: with the OLD buggy offset (len(_accumulated) instead of
        the seal point), 'BBBB' is dropped — proving these tests track behavior,
        not a frozen snapshot."""
        adapter = _make_adapter(keepalive_enabled=False)
        try:
            calls = []

            async def rec(req_id, stream_id, content, finish=False, **kw):
                calls.append((stream_id, finish, content))
                return {"errcode": 0}
            adapter._send_stream_reply = rec

            c = self._make_gateway_consumer(adapter)
            c._accumulated = "AAAA"
            await c._send_or_edit("AAAA", finalize=False)
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            turn.start_time -= adapter._stream_safe_duration_seconds + 500
            c._accumulated = "AAAABBBB"
            await c._send_or_edit("AAAABBBB", finalize=False)
            # Simulate the OLD bug: offset = full length instead of seal point.
            c._native_split_offset = len(c._accumulated)
            c._accumulated = "AAAABBBBCCCC"
            await c._send_or_edit("AAAABBBBCCCC", finalize=False)

            seen = self._concat_bubbles(calls)
            assert "BBBB" not in seen, (
                "sanity: the old offset MUST drop BBBB (else the no-loss tests "
                "above are not actually exercising the fix)"
            )
        finally:
            await adapter.disconnect()
