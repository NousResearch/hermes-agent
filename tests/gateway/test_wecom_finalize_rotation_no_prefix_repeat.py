"""Regression: a FINALIZE frame that itself crosses (or lands right after) a
Layer 2 rotation must NOT repeat the sealed prefix on the fresh bubble.

Reviewer "unsupportedpastels" finding #3 (PR #96942): when the finalize frame
crosses the rotation threshold, the adapter rotates up-front INSIDE the finalize
call and seals the old bubble with the current body ("PREFIX…⏬⏬⏬"), but
``_finalize_turn`` was EXEMPT from the intermediate body-defer, so it sent the
finalize ``text`` (the gateway's full cumulative slice, still cut at the STALE
pre-rotation offset — i.e. it still starts with "PREFIX") verbatim onto the
fresh bubble → the new bubble repeats "PREFIX".

Unlike an intermediate frame, a finalize has no subsequent frame to correct its
slice, so the fix must establish the seal offset BEFORE the final body is
composed and slice the finalize body by that seal length (offset accounting),
leaving only the post-seal TAIL on the fresh bubble.

These drive the REAL ``WeComAdapter._send_stream_frame_inner`` with only the
byte-level ``_send_stream_reply`` seam faked (same harness as
test_wecom_stream_dup_fix.py), so the actual rotation / finalize branches run.

The realistic gateway contract is modelled explicitly: the gateway slices
``_accumulated`` at its ``_native_split_offset`` before the wire, but on the
finalize-crosses-rotation case the offset has NOT yet advanced (it learns
``rotated=True`` only from THIS call's return), so the finalize ``text`` the
adapter receives is the FULL cumulative body — it starts with the very prefix
the old bubble just sealed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wecom.adapter import (
    WeComAdapter,
    ROTATION_CONTINUATION_SUFFIX,
)


CHAT_ID = "chat-fin-rot"
REQ_ID = "req-fin-rot"
TURN_ID = "turn-fin-rot"

PREFIX = "PREFIX BODY that the old bubble already showed before the wall."
TAIL = " TAIL that only the fresh bubble should carry."
FULL = PREFIX + TAIL


def _make_adapter() -> WeComAdapter:
    adapter = WeComAdapter(PlatformConfig(enabled=True, extra={"stream_keepalive_enabled": True}))
    adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
    return adapter


def _finish_frames(mock: AsyncMock) -> list:
    """content strings of every finish=True (_send_stream_reply) call, in order."""
    return [c.args[2] for c in mock.await_args_list if c.kwargs.get("finish") is True]


def _all_stream_ids(mock: AsyncMock) -> list:
    return [c.args[1] for c in mock.await_args_list]


class TestFinalizeCrossingRotationNoPrefixRepeat:
    @pytest.mark.asyncio
    async def test_finalize_frame_itself_crosses_rotation(self):
        """Scenario (a): the finalize frame ITSELF crosses the rotation threshold.

        Up-front rotation happens inside the finalize call. The fresh bubble's
        finish=true frame must carry ONLY the post-seal TAIL, not the sealed
        PREFIX.
        """
        adapter = _make_adapter()
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            # Seed + first intermediate: the old bubble accumulates PREFIX.
            await adapter._send_stream_frame_inner(
                PREFIX, chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            old_stream_id = turn.stream_id
            assert turn.accumulated_text == PREFIX  # sealed body-of-record

            # Age the stream past the Layer 2 safe duration so THIS finalize
            # rotates up-front.
            turn.start_time -= adapter._stream_safe_duration_seconds + 500

            # Gateway contract on the finalize-crosses-rotation case: the split
            # offset has NOT advanced yet, so the finalize text is the FULL
            # cumulative body (starts with PREFIX).
            ok = await adapter._send_stream_frame_inner(
                FULL, chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )
            assert ok

            finishes = _finish_frames(reply)
            # Two finish frames: the sealed old bubble + the fresh finalize.
            assert len(finishes) == 2, (
                f"expected seal + fresh finalize (2 finish frames), got {finishes!r}"
            )
            # First finish frame = sealed old bubble (PREFIX + continuation marker).
            assert finishes[0].endswith(ROTATION_CONTINUATION_SUFFIX)
            assert PREFIX in finishes[0]

            # Second finish frame = the FRESH bubble finalize. It must be the
            # TAIL only, with NO repeated PREFIX.
            fresh_final = finishes[1]
            assert PREFIX not in fresh_final, (
                f"fresh-bubble finalize repeats the sealed prefix: {fresh_final!r}"
            )
            assert fresh_final == TAIL, (
                f"fresh-bubble finalize must carry only the post-seal tail; got "
                f"{fresh_final!r}"
            )
            # And the fresh finalize landed on a DIFFERENT (rotated) stream.
            final_call = [c for c in reply.await_args_list if c.kwargs.get("finish") is True][-1]
            assert final_call.args[1] != old_stream_id
        finally:
            for t in list(adapter._stream_turns.values()):
                adapter._cancel_keepalive(t)
                adapter._cancel_rotation_check(t)

    @pytest.mark.asyncio
    async def test_active_timer_rotation_then_finalize_is_next_frame(self):
        """Scenario (b): an active-timer rotation seals the bubble while NO frame
        flows, and the very NEXT frame is the finalize.

        The timer's rotation set ``pending_rotation_signal`` and sealed the old
        bubble with PREFIX; the up-front age-check on the finalize may not
        re-trigger (already rotated). The finalize must STILL slice by the seal
        so the fresh bubble carries only the TAIL.
        """
        adapter = _make_adapter()
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner(
                PREFIX, chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
            )
            turn = adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"]
            old_stream_id = turn.stream_id
            assert turn.accumulated_text == PREFIX

            # Simulate the ACTIVE rotation timer firing while no frame is in
            # flight: seal + rotate directly (leaves pending_rotation_signal set,
            # to be reported on the next frame — here, the finalize).
            async with turn.rotation_lock():
                rotated = await adapter._rotate_stream_locked(turn, TURN_ID)
            assert rotated
            assert turn.pending_rotation_signal is True
            assert turn.stream_id != old_stream_id

            # The NEXT frame is the finalize, carrying the full cumulative body
            # (gateway offset still stale — active-timer rotation is reported one
            # frame late).
            ok = await adapter._send_stream_frame_inner(
                FULL, chat=CHAT_ID, finalize=True, turn_id=TURN_ID,
            )
            assert ok

            finishes = _finish_frames(reply)
            # seal (from _rotate_stream_locked) + fresh finalize.
            assert len(finishes) == 2, (
                f"expected seal + fresh finalize (2 finish frames), got {finishes!r}"
            )
            assert finishes[0].endswith(ROTATION_CONTINUATION_SUFFIX)
            assert PREFIX in finishes[0]

            fresh_final = finishes[1]
            assert PREFIX not in fresh_final, (
                f"fresh-bubble finalize repeats the sealed prefix: {fresh_final!r}"
            )
            assert fresh_final == TAIL, (
                f"fresh-bubble finalize must carry only the post-seal tail; got "
                f"{fresh_final!r}"
            )
        finally:
            for t in list(adapter._stream_turns.values()):
                adapter._cancel_keepalive(t)
                adapter._cancel_rotation_check(t)
