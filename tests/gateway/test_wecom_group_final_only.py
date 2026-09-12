"""Behavior-contract tests for the WeCom **group-chat** answer policy (2026-09-12).

Incident (2026-09-12 12:04, a production group): the group saw the bot's tool /
script frames ("Reading skill …", "🖥️ Running python3 …", script bodies) while
the 174-char final answer never reached the members.  Two independent causes,
both fixed in ``plugins/platforms/wecom/streaming.py``:

Fix A — ``group_final_only`` (default on).  WeCom advertises
``SUPPORTS_NATIVE_STREAMING``, so the gateway's stream consumer routes every
tool-progress line into the native bubble itself
(``stream_consumer.accepts_tool_progress`` → ``on_tool_progress``, composed as
``"<text>\\n\\n---\\n<progress>"``).  For a **group** the adapter now withholds
every non-final frame and strips the progress overlay from the frame that does
go out, so only the finished answer is ever written to the group.  DMs keep the
previous streaming behaviour.  Toggle = ``extra['group_final_only']``.

Fix B — ``group_final_ack_strict`` (default on).  Upstream converts an ack
timeout into "assumed delivered" and lets the gateway suppress the normal final
send; in a group that can mean the answer never arrives.  For a group the
finalize frame must now be POSITIVELY acked, otherwise the finalize declines
(return False) so the consumer's ``send()`` fallback delivers the answer as a
message.  Toggle = ``extra['group_final_ack_strict']``.

These drive the REAL ``WeComAdapter._send_stream_frame_inner`` / ``_finalize_turn``
with only the byte-level ``_send_stream_reply`` seam faked, so the actual control
flow runs.  Assertions read observable adapter state: the return value (what the
consumer keys its fallback on), what actually reached the wire, whether the turn
survived, and the group set membership.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wecom.adapter import WeComAdapter
from plugins.platforms.wecom.streaming import (
    GROUP_FINAL_ACK_STRICT_DEFAULT,
    GROUP_FINAL_ONLY_DEFAULT,
    GROUP_PROGRESS_SEPARATOR,
    ack_confirmed,
)

GROUP_CHAT = "group-chat-1"
DM_CHAT = "dm-chat-1"
REQ_ID = "req-group"
TURN_ID = "turn-group"

ANSWER = "今天的活动 14:00 开场，已确认 7 人到场，现场交付清单已发群里。"
PROGRESS = "🖥️ Running python3 scripts/check_status.py …\nReading skill ops-toolkit …"
# What stream_consumer composes for a native frame: answer text, then the tool-progress overlay.
COMPOSED_GROUP_TURN = f"{ANSWER}{GROUP_PROGRESS_SEPARATOR}{PROGRESS}"
ACK_TIMEOUT = {"errcode": 0, "errmsg": "ack_timeout_assumed_delivered", "ack_pending": True, "confirmed": False}
ACK_OK = {"errcode": 0, "errmsg": "ok", "confirmed": True}


def _make_adapter(*, group: bool, extra: dict | None = None) -> WeComAdapter:
    """Real adapter with only the byte-level stream writer faked."""
    merged = {"stream_keepalive_enabled": False, **(extra or {})}
    adapter = WeComAdapter(PlatformConfig(enabled=True, extra=merged))
    chat = GROUP_CHAT if group else DM_CHAT
    adapter._last_chat_req_ids[chat] = REQ_ID
    if group:
        adapter._group_chat_ids.add(GROUP_CHAT)
    return adapter


def _frames(mock: AsyncMock, *, finish: bool | None = None) -> list:
    calls = mock.await_args_list
    if finish is None:
        return calls
    return [c for c in calls if c.kwargs.get("finish") is finish]


def _contents(mock: AsyncMock, *, finish: bool | None = None) -> list[str]:
    return [c.args[2] for c in _frames(mock, finish=finish)]


# ===========================================================================
# Fix A — a group only ever receives the finished answer
# ===========================================================================


class TestGroupFinalOnly:
    @pytest.mark.asyncio
    async def test_group_intermediate_frames_are_withheld(self):
        """Group + mid-turn text (answer and tool progress) → nothing reaches the wire,
        but the turn still accumulates so the finalize can deliver it."""
        adapter = _make_adapter(group=True)
        try:
            reply = AsyncMock(return_value=ACK_OK)
            adapter._send_stream_reply = reply

            ok = await adapter._send_stream_frame_inner(
                COMPOSED_GROUP_TURN, chat=GROUP_CHAT, finalize=False, turn_id=TURN_ID,
            )

            assert ok is True, "withholding is not a failure — the consumer must not fall back mid-turn"
            assert _contents(reply, finish=False) == ["<think></think>"], (
                "the seed frame opens the thinking bubble; no content frame may follow it"
            )
            assert PROGRESS not in "".join(_contents(reply))
            assert ANSWER not in "".join(_contents(reply))
            turn = adapter._stream_turns[f"{GROUP_CHAT}:{TURN_ID}"]
            assert turn.accumulated_text == COMPOSED_GROUP_TURN, "text is still accumulated for the finalize"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_group_final_frame_drops_the_tool_progress_overlay(self):
        """The finalize frame carries the answer only — belt and braces on top of withholding."""
        adapter = _make_adapter(group=True)
        try:
            reply = AsyncMock(return_value=ACK_OK)
            adapter._send_stream_reply = reply

            ok = await adapter._send_stream_frame_inner(
                "", chat=GROUP_CHAT, finalize=False, turn_id=TURN_ID,
            )
            assert ok is True
            ok = await adapter._send_stream_frame_inner(
                COMPOSED_GROUP_TURN, chat=GROUP_CHAT, finalize=True, turn_id=TURN_ID,
            )

            assert ok is True
            finals = _contents(reply, finish=True)
            assert len(finals) == 1, "exactly one finalize frame"
            assert finals[0] == ANSWER, "tool-progress overlay stripped from the group final frame"
            assert PROGRESS not in finals[0]
            assert f"{GROUP_CHAT}:{TURN_ID}" not in adapter._stream_turns, "turn cleaned up after finalize"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_group_keepalive_never_writes_accumulated_text(self):
        """Keep-alive exists to refresh the stream window; it must not leak mid-turn text to a group."""
        adapter = _make_adapter(group=True, extra={"stream_keepalive_enabled": True})
        try:
            reply = AsyncMock(return_value=ACK_OK)
            adapter._send_stream_reply = reply
            await adapter._send_stream_frame_inner(COMPOSED_GROUP_TURN, chat=GROUP_CHAT, finalize=False, turn_id=TURN_ID)
            turn = adapter._stream_turns[f"{GROUP_CHAT}:{TURN_ID}"]
            reply.reset_mock()

            await adapter._keepalive_send(turn, TURN_ID)

            assert _contents(reply) == [], "no keep-alive content frame for a group"
            assert turn.keepalive_handle is not None, "timer stays armed so a live stream can still finalize"
            assert turn.last_sent_content != COMPOSED_GROUP_TURN
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_group_final_only_can_be_disabled(self):
        """Toggle: with the policy off, a group streams like a DM (toggle-validation)."""
        adapter = _make_adapter(group=True, extra={"group_final_only": False})
        try:
            reply = AsyncMock(return_value=ACK_OK)
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner("mid-turn text", chat=GROUP_CHAT, finalize=False, turn_id=TURN_ID)

            assert "mid-turn text" in _contents(reply, finish=False), "policy off → intermediates stream again"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_dm_intermediate_frames_still_stream(self):
        """Regression guard: direct messages keep the previous behaviour."""
        adapter = _make_adapter(group=False)
        try:
            reply = AsyncMock(return_value=ACK_OK)
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner("mid-turn text", chat=DM_CHAT, finalize=False, turn_id=TURN_ID)

            assert "mid-turn text" in _contents(reply, finish=False), "DM intermediates unchanged"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_group_progress_hint_without_separator_is_untouched(self):
        """No overlay present → the finalize text goes out verbatim."""
        adapter = _make_adapter(group=True)
        try:
            reply = AsyncMock(return_value=ACK_OK)
            adapter._send_stream_reply = reply
            await adapter._send_stream_frame_inner("", chat=GROUP_CHAT, finalize=False, turn_id=TURN_ID)

            await adapter._send_stream_frame_inner("纯答案，无进度叠加层", chat=GROUP_CHAT, finalize=True, turn_id=TURN_ID)

            assert _contents(reply, finish=True) == ["纯答案，无进度叠加层"]
        finally:
            await adapter.disconnect()


# ===========================================================================
# Fix B — a group finalize must be positively acked, else fall back to send()
# ===========================================================================


class TestGroupFinalAckStrict:
    @pytest.mark.asyncio
    async def test_group_finalize_unconfirmed_ack_declines_for_the_send_fallback(self):
        """Ack timeout (assumed delivered upstream) must NOT silence the group answer."""
        adapter = _make_adapter(group=True)
        try:
            reply = AsyncMock(return_value=ACK_TIMEOUT)
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner("", chat=GROUP_CHAT, finalize=False, turn_id=TURN_ID)
            ok = await adapter._send_stream_frame_inner(ANSWER, chat=GROUP_CHAT, finalize=True, turn_id=TURN_ID)

            assert ok is False, "unconfirmed group finalize declines → consumer send() delivers the answer"
            assert f"{GROUP_CHAT}:{TURN_ID}" not in adapter._stream_turns, "declined turn is retired"
            assert GROUP_CHAT not in adapter._stream_expired_chats, "chat not expired: a later turn may stream again"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_group_finalize_confirmed_ack_succeeds(self):
        adapter = _make_adapter(group=True)
        try:
            reply = AsyncMock(return_value=ACK_OK)
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner("", chat=GROUP_CHAT, finalize=False, turn_id=TURN_ID)
            ok = await adapter._send_stream_frame_inner(ANSWER, chat=GROUP_CHAT, finalize=True, turn_id=TURN_ID)

            assert ok is True, "positively acked group finalize is delivered in the bubble"
            assert f"{GROUP_CHAT}:{TURN_ID}" not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_dm_finalize_ack_timeout_keeps_upstream_behaviour(self):
        """Regression guard: in a DM a late ack is still trusted (no duplicate send)."""
        adapter = _make_adapter(group=False)
        try:
            reply = AsyncMock(return_value=ACK_TIMEOUT)
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner("", chat=DM_CHAT, finalize=False, turn_id=TURN_ID)
            ok = await adapter._send_stream_frame_inner(ANSWER, chat=DM_CHAT, finalize=True, turn_id=TURN_ID)

            assert ok is True, "DM behaviour unchanged: ack timeout still counts as delivered"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_group_ack_strict_can_be_disabled(self):
        """Toggle: with strict ack off, a group trusts the timeout like a DM (toggle-validation)."""
        adapter = _make_adapter(group=True, extra={"group_final_ack_strict": False})
        try:
            reply = AsyncMock(return_value=ACK_TIMEOUT)
            adapter._send_stream_reply = reply

            await adapter._send_stream_frame_inner("", chat=GROUP_CHAT, finalize=False, turn_id=TURN_ID)
            ok = await adapter._send_stream_frame_inner(ANSWER, chat=GROUP_CHAT, finalize=True, turn_id=TURN_ID)

            assert ok is True
        finally:
            await adapter.disconnect()


# ===========================================================================
# Policy plumbing
# ===========================================================================


class TestPolicyPlumbing:
    def test_defaults_are_on(self):
        assert GROUP_FINAL_ONLY_DEFAULT is True
        assert GROUP_FINAL_ACK_STRICT_DEFAULT is True
        adapter = _make_adapter(group=True)
        assert adapter._group_final_only is True
        assert adapter._group_final_ack_strict is True

    def test_policy_applies_to_groups_only(self):
        adapter = _make_adapter(group=True)
        assert adapter._is_group_chat(GROUP_CHAT) is True
        assert adapter._is_group_chat(DM_CHAT) is False
        assert adapter._is_group_chat("") is False
        assert adapter._is_group_chat(None) is False

    def test_ack_confirmed_distinguishes_timeout_from_ack(self):
        assert ack_confirmed(ACK_TIMEOUT) is False, "assumed-delivery timeout is not a confirmation"
        assert ack_confirmed(ACK_OK) is True
        assert ack_confirmed({"errcode": 0}) is True, "plain success payload counts"
        assert ack_confirmed({"errcode": 0, "confirmed": False}) is False
        assert ack_confirmed(None) is False
        assert ack_confirmed("nope") is False
