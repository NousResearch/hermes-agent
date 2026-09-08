"""Regression for PR #96942 reviewer finding #1 (ACK-settlement race).

Two coupled defects in ``plugins/platforms/wecom/streaming.py``:

Part A — final-ACK timeout falsely reports delivery.
  ``_send_reply_queued`` awaits the FINAL frame's ack up to ``_REPLY_ACK_TIMEOUT``.
  On ``asyncio.TimeoutError`` the OLD code returned
  ``{"errcode": 0, "errmsg": "ack_timeout_assumed_delivered", "ack_pending": True}``
  — i.e. it ASSUMED delivery, which can silently drop the final answer.

  Approved fix: on timeout, RE-SEND the identical final frame ONCE (same
  stream_id + same content is cumulative and ACKed within the 10-min window —
  verified by live WS probe; no duplicate bubble, no 6000). Then:
    * re-send ACK errcode 0            → normal delivered response;
    * re-send ACK 846608/846604        → WeComStreamExpiredError propagates
                                         (consumer fallback), NOT assumed-delivered;
    * re-send ALSO times out (~30s)    → degrade to
                                         ``{"errcode": 0, "errmsg": "settlement_indeterminate", ...}``
                                         (the poison signal the consumer chain
                                         already understands).
  ``ack_timeout_assumed_delivered`` must never be returned anywhere.

Part B — coalesce/final-fence ACK-identity race.
  WeCom acks carry only ``req_id`` (no per-frame id), so ``_resolve_reply_ack``
  resolves whichever frame currently occupies ``queue.pending_ack``. An
  ACK-triggered ``_flush_coalesced`` could publish a buffered successor B into
  the slot right as the finalize registers frame F, letting a later ACK(B)
  certify F. Approved fix: a ``ReplyQueue.finalizing`` fence — once a final
  frame begins finalization no coalesced successor may be published (neither the
  ``skip_if_pending`` coalesce branch nor ``_flush_coalesced``).

These tests instantiate a REAL ``WeComAdapter`` and mock only ``_send_json``
(the transport byte-writer), controlling ACK arrival via the pending future.
Each scenario is RED on the pre-fix code (assume-delivered / no fence) and GREEN
after.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


CHAT_ID = "chat-1"
REQ_ID = "req-1"


# ---------------------------------------------------------------------------
# Real WeComAdapter + a scriptable transport.
# ---------------------------------------------------------------------------


def _make_adapter():
    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._ws = MagicMock(closed=False)
    adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
    return adapter


def _install_transport(adapter, *, final_actions):
    """Install a ``_send_json`` that records frames and scripts ACK arrival.

    Non-final (finish=False) frames — seed / intermediate — have their ack
    resolved immediately (errcode 0) and ``pending_ack`` cleared, so the
    finalize's pre-drain never blocks on them (mirrors
    tests/gateway/test_wecom.py ``_mock_send_json_with_immediate_ack``).

    Final (finish=True) frames are matched to ``final_actions[i]`` by 0-based
    order of finish=true sends:
      * ``None``               → leave the ack pending (→ wait_for TimeoutError);
      * a ``dict`` payload     → resolve the pending future with it (the ack);
      * an ``Exception``       → raise it from ``_send_json`` (dead transport).
    """
    frames: list[dict] = []
    final_bodies: list[dict] = []
    final_idx = [0]

    async def _send_json(payload: dict) -> None:
        frames.append(payload)
        stream = payload.get("body", {}).get("stream", {})
        finish = bool(stream.get("finish"))
        req = payload.get("headers", {}).get("req_id")
        if not finish:
            q = adapter._reply_queues.get(req)
            if q and q.pending_ack and not q.pending_ack.future.done():
                q.pending_ack.future.set_result({"errcode": 0, "errmsg": "ok"})
                q.pending_ack = None
            return
        # A finish=true frame reached the wire.
        final_bodies.append(payload.get("body", {}))
        idx = final_idx[0]
        final_idx[0] += 1
        action = final_actions[idx] if idx < len(final_actions) else None
        if action is None:
            return  # ack never comes → the awaiter times out
        if isinstance(action, Exception):
            raise action
        q = adapter._reply_queues.get(req)
        if q and q.pending_ack and not q.pending_ack.future.done():
            q.pending_ack.future.set_result(action)

    adapter._send_json = _send_json
    adapter._recorded_frames = frames
    adapter._final_bodies = final_bodies
    return frames, final_bodies


async def _seed_turn(adapter, *, turn_id="turn-1"):
    """Open a turn (seed + a content frame) so a later finalize has a live turn."""
    await adapter.send_stream_frame("some content", chat_id=CHAT_ID, turn_id=turn_id)
    turn = adapter._stream_turns[f"{CHAT_ID}:{turn_id}"]
    assert turn.seeded
    return turn


async def _cleanup(adapter) -> None:
    for task in list(getattr(adapter, "_control_workers", {}).values()) + list(
        getattr(adapter, "_chat_workers", {}).values()
    ):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    try:
        await adapter.disconnect()
    except Exception:
        pass


# ===========================================================================
# Scenario 1 — timeout → re-send succeeds → normal delivery.
# ===========================================================================


class TestFinalAckTimeoutResendSucceeds:
    @pytest.mark.asyncio
    async def test_timeout_then_resend_ack_delivers_normally(self):
        """First final ack times out; the re-sent identical final frame is ACKed.

        RED (pre-fix): no re-send — the first timeout returns
        ``ack_timeout_assumed_delivered`` and ``_send_json`` sends the final body
        only ONCE.
        GREEN (post-fix): the final body is re-sent (TWICE, identical), the ACK
        arrives, and the turn finalizes through the normal DELIVERED path.
        """
        from plugins.platforms.wecom.adapter import StreamFrameResult

        adapter = _make_adapter()
        adapter._REPLY_ACK_TIMEOUT = 0.05  # snappy timeout for attempt 1
        # attempt 1 → no ack (times out); attempt 2 (re-send) → errcode 0 ack.
        _, final_bodies = _install_transport(
            adapter, final_actions=[None, {"errcode": 0, "errmsg": "ok"}]
        )
        try:
            turn = await _seed_turn(adapter)

            result = await adapter.send_stream_frame(
                "final answer text", chat_id=CHAT_ID, finalize=True, turn_id="turn-1"
            )

            # The SAME final body was put on the wire twice (re-send is identical).
            assert len(final_bodies) == 2, (
                f"expected the final frame to be re-sent exactly once (2 sends), "
                f"got {len(final_bodies)}"
            )
            assert final_bodies[0] == final_bodies[1], "re-send must be byte-identical"
            assert final_bodies[0]["stream"]["finish"] is True

            # Normal delivered path — NOT indeterminate, NOT assumed-delivered.
            assert result is StreamFrameResult.DELIVERED
            assert turn.finalized is True
            assert f"{CHAT_ID}:turn-1" not in adapter._stream_turns
        finally:
            await _cleanup(adapter)


# ===========================================================================
# Scenario 2 — timeout → re-send hits 846608 → expiry propagates.
# ===========================================================================


class TestFinalAckTimeoutResendExpires:
    @pytest.mark.asyncio
    async def test_timeout_then_resend_846608_propagates_as_expiry(self):
        """First final times out; the re-send's ack is 846608 (stream past the wall).

        RED (pre-fix): first timeout → assumed-delivered, no re-send, expiry never
        surfaces (consumer never falls back).
        GREEN (post-fix): re-send ack 846608 → ``_send_stream_reply`` raises
        ``WeComStreamExpiredError`` → ``_finalize_turn`` maps it to FAILED so the
        consumer falls back to a proactive send. NOT assumed-delivered.
        """
        from plugins.platforms.wecom.adapter import (
            StreamFrameResult,
            STREAM_EXPIRED_ERRCODE,
        )

        adapter = _make_adapter()
        adapter._REPLY_ACK_TIMEOUT = 0.05
        _, final_bodies = _install_transport(
            adapter,
            final_actions=[None, {"errcode": STREAM_EXPIRED_ERRCODE, "errmsg": "expired"}],
        )
        try:
            turn = await _seed_turn(adapter)

            result = await adapter.send_stream_frame(
                "final answer text", chat_id=CHAT_ID, finalize=True, turn_id="turn-1"
            )

            assert len(final_bodies) == 2, "final frame must be re-sent once before expiry"
            # Expiry path: finalize returns FAILED (falsy) → consumer fallback reachable.
            assert result is StreamFrameResult.FAILED
            assert bool(result) is False
            # The turn/chat is marked expired (proactive-send fallback path).
            assert CHAT_ID in adapter._stream_expired_chats
            assert turn.expired is True
        finally:
            await _cleanup(adapter)

    @pytest.mark.asyncio
    async def test_resend_846608_raises_expired_error_at_reply_layer(self):
        """Lower-level assertion: at ``_send_stream_reply`` the re-send 846608 raises.

        Confirms the expiry is a real ``WeComStreamExpiredError`` (not swallowed),
        directly exercising ``_send_reply_queued`` → ``_send_stream_reply``.
        """
        from plugins.platforms.wecom.adapter import (
            WeComStreamExpiredError,
            STREAM_EXPIRED_ERRCODE,
        )

        adapter = _make_adapter()
        adapter._REPLY_ACK_TIMEOUT = 0.05
        _install_transport(
            adapter,
            final_actions=[None, {"errcode": STREAM_EXPIRED_ERRCODE, "errmsg": "expired"}],
        )
        try:
            with pytest.raises(WeComStreamExpiredError) as exc:
                await adapter._send_stream_reply(REQ_ID, "stream-1", "final", finish=True)
            assert exc.value.errcode == STREAM_EXPIRED_ERRCODE
        finally:
            await _cleanup(adapter)


# ===========================================================================
# Scenario 3 — timeout → re-send also times out → settlement_indeterminate.
# ===========================================================================


class TestFinalAckDoubleTimeoutIndeterminate:
    @pytest.mark.asyncio
    async def test_both_attempts_timeout_returns_settlement_indeterminate(self):
        """Both the first final ack and the re-send ack time out.

        RED (pre-fix): the single attempt returns ``ack_timeout_assumed_delivered``
        and ``turn.finalized`` is set True (false positive).
        GREEN (post-fix): the final is re-sent once, both time out, and the result
        degrades to ``settlement_indeterminate`` (errcode 0) → INDETERMINATE, with
        ``turn.finalized`` NOT set (reuses the settlement-indeterminate contract).
        """
        from plugins.platforms.wecom.adapter import StreamFrameResult

        adapter = _make_adapter()
        adapter._REPLY_ACK_TIMEOUT = 0.05
        _, final_bodies = _install_transport(adapter, final_actions=[None, None])
        try:
            turn = await _seed_turn(adapter)

            result = await adapter.send_stream_frame(
                "final answer text", chat_id=CHAT_ID, finalize=True, turn_id="turn-1"
            )

            assert len(final_bodies) == 2, "both attempts must reach the wire"
            assert result is StreamFrameResult.INDETERMINATE
            assert bool(result) is True  # frame sent — no fallback
            # KEY: unconfirmed delivery must NOT be recorded as finalized.
            assert turn.finalized is False
            assert CHAT_ID not in adapter._stream_expired_chats
        finally:
            await _cleanup(adapter)

    @pytest.mark.asyncio
    async def test_double_timeout_reply_layer_returns_indeterminate_not_assumed(self):
        """At ``_send_reply_queued`` the double-timeout return is the indeterminate
        signal, and ``ack_timeout_assumed_delivered`` never appears."""
        adapter = _make_adapter()
        adapter._REPLY_ACK_TIMEOUT = 0.05
        _, final_bodies = _install_transport(adapter, final_actions=[None, None])
        try:
            body = {
                "msgtype": "stream",
                "stream": {"id": "s1", "finish": True, "content": "done"},
            }
            resp = await adapter._send_reply_queued(REQ_ID, body, is_final=True)

            assert len(final_bodies) == 2
            assert resp.get("errmsg") == "settlement_indeterminate"
            assert resp.get("errcode") == 0
            assert resp.get("errmsg") != "ack_timeout_assumed_delivered"
            # queue fully released after both attempts.
            assert REQ_ID not in adapter._reply_queues
        finally:
            await _cleanup(adapter)


# ===========================================================================
# Scenario 4 — finalizing fence blocks a coalesced successor from stealing the
# final frame's ack slot.
# ===========================================================================


class TestFinalizingFenceBlocksSuccessor:
    @pytest.mark.asyncio
    async def test_flush_coalesced_is_noop_while_finalizing(self):
        """With ``queue.finalizing = True``, ``_flush_coalesced`` must not publish.

        Direct-invariant form (the deterministic assertion the spec sanctions):
        a buffered successor B is present, finalization has started, and the flush
        is a no-op — B is never written and stays buffered (so the final frame
        keeps sole ownership of the ack slot).
        """
        from plugins.platforms.wecom.streaming import ReplyQueue

        adapter = _make_adapter()
        sent: list[dict] = []

        async def _record_send(payload):
            sent.append(payload)

        adapter._send_json = _record_send
        try:
            queue = ReplyQueue(REQ_ID)
            adapter._reply_queues[REQ_ID] = queue
            queue.coalesced_body = {
                "msgtype": "stream",
                "stream": {"id": "s1", "finish": False, "content": "successor B"},
            }
            queue.coalesce_count = 1
            queue.finalizing = True

            await adapter._flush_coalesced(REQ_ID)

            assert sent == [], "fence breached: coalesced successor B was published while finalizing"
            assert queue.coalesced_body is not None, "buffered B must remain unsent under the fence"
        finally:
            await _cleanup(adapter)

    @pytest.mark.asyncio
    async def test_coalesce_branch_suppressed_while_finalizing(self):
        """The ``skip_if_pending`` coalesce branch must not buffer a NEW successor
        once finalization owns the slot."""
        from plugins.platforms.wecom.streaming import ReplyQueue, ReplyFrame

        adapter = _make_adapter()
        sent: list[dict] = []

        async def _record_send(payload):
            sent.append(payload)

        adapter._send_json = _record_send
        try:
            queue = ReplyQueue(REQ_ID)
            adapter._reply_queues[REQ_ID] = queue
            # An intermediate A is in flight (pending_ack occupied) and the fence is up.
            fut = asyncio.get_running_loop().create_future()
            queue.pending_ack = ReplyFrame(
                body={"stream": {"id": "s1", "finish": False, "content": "A"}},
                future=fut, is_final=False,
            )
            queue.finalizing = True

            resp = await adapter._send_reply_queued(
                REQ_ID,
                {"msgtype": "stream", "stream": {"id": "s1", "finish": False, "content": "B"}},
                is_final=False, skip_if_pending=True,
            )

            assert resp.get("skipped") is True
            assert sent == [], "coalesce branch must not send while finalizing"
            assert queue.coalesced_body is None, (
                "fence breached: a successor was buffered into the slot while finalizing"
            )
            if not fut.done():
                fut.cancel()
        finally:
            await _cleanup(adapter)

    @pytest.mark.asyncio
    async def test_ack_of_A_while_finalizing_does_not_publish_B_or_resolve_final(self):
        """Integrated: intermediate A pending + B coalesced; finalization starts and
        registers final F; delivering ACK(A) schedules a flush — the fence keeps B
        unpublished, and neither ACK(A) nor a stray ACK(B) resolves F's future.

        The final's own ack never arrives (final_actions=[None]) so the finalize
        would sit awaiting; we drive it as a background task, deliver ACK(A), and
        assert the invariants, then let it degrade. This is the mis-attribution
        the fence prevents.
        """
        from plugins.platforms.wecom.streaming import ReplyQueue, ReplyFrame

        adapter = _make_adapter()
        adapter._REPLY_ACK_TIMEOUT = 0.05
        _, final_bodies = _install_transport(adapter, final_actions=[None, None])
        try:
            queue = ReplyQueue(REQ_ID)
            adapter._reply_queues[REQ_ID] = queue
            # Intermediate A in flight.
            a_future = asyncio.get_running_loop().create_future()
            a_frame = ReplyFrame(
                body={"msgtype": "stream", "stream": {"id": "s1", "finish": False, "content": "A"}},
                future=a_future, is_final=False, sent_at=0.0,
            )
            queue.pending_ack = a_frame
            # B buffered behind A.
            queue.coalesced_body = {
                "msgtype": "stream",
                "stream": {"id": "s1", "finish": False, "content": "B"},
            }
            queue.coalesce_count = 1

            # Start the finalize: it flips finalizing on, drains A, registers F.
            fin_task = asyncio.create_task(
                adapter._send_reply_queued(
                    REQ_ID,
                    {"msgtype": "stream", "stream": {"id": "s1", "finish": True, "content": "F"}},
                    is_final=True,
                )
            )
            # Let the finalize reach the drain await on A's future.
            await asyncio.sleep(0)

            # Deliver ACK(A): resolves the drain and (pre-fence) would schedule a
            # _flush_coalesced that republishes B into the slot.
            await adapter._dispatch_payload({"headers": {"req_id": REQ_ID}, "body": {"errcode": 0}})
            # Give any scheduled flush task a chance to run.
            for _ in range(5):
                await asyncio.sleep(0)

            # No intermediate "B" frame was ever put on the wire — only final "F" sends.
            b_frames = [
                f for f in adapter._recorded_frames
                if f.get("body", {}).get("stream", {}).get("content") == "B"
            ]
            assert b_frames == [], "fence breached: coalesced B was published during finalization"

            resp = await asyncio.wait_for(fin_task, timeout=1.0)
            # F's future was resolved by neither ACK(A) nor a stray B — it degraded.
            assert resp.get("errmsg") == "settlement_indeterminate"
            assert resp.get("errmsg") != "ack_timeout_assumed_delivered"
            assert all(b["stream"]["content"] == "F" for b in final_bodies)
        finally:
            await _cleanup(adapter)
