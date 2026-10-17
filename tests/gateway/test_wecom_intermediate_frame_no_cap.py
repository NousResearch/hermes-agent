"""Regression: WeCom native streaming must NOT drop intermediate frames past 85.

Root cause of the "frozen bubble" content-freeze on long turns: the fire-and-forget
intermediate-frame path enforced ``MAX_INTERMEDIATE_FRAMES = 85``. With ~1fps timer ticks
plus text streaming, any turn longer than ~85s exhausted the cap and every subsequent
intermediate frame was silently dropped — the user saw frozen content (timer + body stuck)
while the agent kept working. The cap was a leftover of the old BlockChunker/webhook regime;
the fire-and-forget WebSocket long-connection path has no WeCom-imposed frame-count limit
(official docs cap only content byte-length and session frequency). The cap removal
(fix(wecom) "remove MAX_INTERMEDIATE_FRAMES cap that froze long turns") was lost in a rebase
and the cap came back; this pins the correct behavior so it cannot regress again.

Contract: pure identity-dedup is the ONLY intermediate gate. N distinct cumulative bodies
must produce N intermediate (finish=False) sends to the wire — no cap-based drop — for N well
past 85. Drives the real ``_send_stream_frame_inner`` with only the byte-level
``_send_stream_reply`` seam faked, so the real send/dedup control flow runs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wecom.adapter import WeComAdapter


CHAT_ID = "chat-cap"
REQ_ID = "req-cap"
TURN_ID = "turn-cap"


def _make_adapter() -> WeComAdapter:
    adapter = WeComAdapter(PlatformConfig(enabled=True, extra={"stream_keepalive_enabled": True}))
    adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
    return adapter


def _intermediate_calls(mock: AsyncMock) -> list:
    """finish=False (intermediate) sends that reached ``_send_stream_reply``."""
    return [c for c in mock.await_args_list if c.kwargs.get("finish") is False]


class TestNoIntermediateFrameCap:
    @pytest.mark.asyncio
    async def test_intermediate_frames_past_85_are_not_dropped(self):
        """RED on the reinstated cap: send 120 distinct cumulative bodies as intermediates.

        Pre-fix (cap = 85) drops every frame after the 85th → only 85 reach the wire and the
        bubble freezes on frame 85's content. Post-fix (identity-dedup only) all 120 reach the
        wire. We assert the LATER content (frame 120) actually got sent, which is exactly what
        the freeze hid.
        """
        adapter = _make_adapter()
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply

            N = 120
            for i in range(N):
                # Cumulative, strictly-growing, distinct text so identity-dedup never suppresses.
                await adapter._send_stream_frame_inner(
                    f"cumulative body up to frame {i:04d}",
                    chat=CHAT_ID, finalize=False, turn_id=TURN_ID,
                )

            inter = _intermediate_calls(reply)
            # No cap: every distinct body past 85 still reaches the wire.
            assert len(inter) >= N, (
                f"expected >= {N} intermediate sends (no frame cap), got {len(inter)} — "
                "MAX_INTERMEDIATE_FRAMES cap is silently dropping frames past 85, freezing the bubble"
            )
            # The freeze specifically hid LATE content: the last frame's body must have gone out.
            last_bodies = [c.args[2] for c in inter]
            assert f"frame {N - 1:04d}" in last_bodies[-1], (
                "the final cumulative body never reached the wire — bubble froze on an earlier frame"
            )
        finally:
            await adapter.disconnect()
