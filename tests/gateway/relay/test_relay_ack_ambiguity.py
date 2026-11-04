"""Regression: the transport's timeout-shaped RESULT is ambiguous, not a
rejection (PR 85796 review round 2, finding 1).

The production ws transport does not raise on ack timeout — it returns
{"success": False, "error": "relay outbound timed out", "ambiguous": True}.
The round-1 fixes keyed ambiguity handling entirely on the exception
channel, so the shape production actually produces was misclassified:

  - a lost SEAL ack skipped the idempotent retry and fell straight to a
    plain send (duplicate final whenever the seal had actually applied);
  - a lost FRAME ack was treated as a definite connector rejection and
    DISARMED interception — frozen native stream beside a plain final
    (the original G-D1 ambiguous-ack defect, moved to the result channel).

Contract now: transport tags ack-timeouts ambiguous=True (fail-fast
closing/not-connected results stay unmarked — nothing was sent); the
adapter retries the idempotent seal frame on ambiguous results exactly as
for exceptions, and disarms interception only on definite rejections.
"""

import asyncio

import pytest

from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class AckLossTransport:
    """Returns the production timeout shape for selected ops."""

    def __init__(self, lose_acks_for=(), lose_times=None):
        self.ops = []
        self.lose_acks_for = set(lose_acks_for)
        self.lose_times = dict(lose_times or {})  # key -> remaining losses
        self._n = 0

    def _lost(self, key):
        if key not in self.lose_acks_for:
            return False
        remaining = self.lose_times.get(key)
        if remaining is None:
            return True
        if remaining <= 0:
            return False
        self.lose_times[key] = remaining - 1
        return True

    async def send_outbound(self, payload, platform=None):
        op = payload.get("op")
        key = "seal" if (op == "draft" and payload.get("final")) else op
        self.ops.append((op, bool(payload.get("final")), str(payload.get("content"))[:30]))
        if self._lost(key):
            return {
                "success": False,
                "error": "relay outbound timed out",
                "ambiguous": True,
            }
        self._n += 1
        return {"success": True, "message_id": f"ts.{self._n}"}


class TestSealAckLoss:
    @pytest.mark.asyncio
    async def test_lost_seal_ack_retries_idempotent_frame(self):
        """One lost seal ack → the SAME final frame is retried; its ack
        carries the stream ts; NO plain send fires."""
        t = AckLossTransport(lose_acks_for=("seal",), lose_times={"seal": 1})
        adapter, _ = _connected_adapter()
        adapter._transport = t
        md = {"message_id": "m.1"}
        await adapter.send_draft("C1", 7, "partial", metadata=md)
        r = await adapter.send("C1", "complete", metadata=dict(md))
        assert r.success
        seal_attempts = [o for o in t.ops if o[0] == "draft" and o[1]]
        assert len(seal_attempts) == 2, "ambiguous seal result must retry"
        assert not [o for o in t.ops if o[0] == "send"], (
            "a retried-and-acked seal must not be followed by a plain send"
        )

    @pytest.mark.asyncio
    async def test_seal_ack_lost_twice_reports_ambiguous_failure(self):
        t = AckLossTransport(lose_acks_for=("seal",))
        adapter, _ = _connected_adapter()
        adapter._transport = t
        md = {"message_id": "m.2"}
        await adapter.send_draft("C1", 8, "partial", metadata=md)
        r = await adapter.send("C1", "complete", metadata=dict(md))
        # Fail-open: the plain send goes out (send acks fine here) — a
        # possible duplicate beats a silent loss after double ack loss.
        assert r.success
        assert [o for o in t.ops if o[0] == "draft" and o[1]], "seal attempted"
        assert [o for o in t.ops if o[0] == "send"], "fail-open plain send ran"


class TestFrameAckLoss:
    @pytest.mark.asyncio
    async def test_lost_frame_ack_keeps_interception_armed(self):
        """THE round-2 regression: a timeout-shaped frame result must not
        disarm — the connector may have applied the frame, and the
        turn-final must still seal the stream."""
        t = AckLossTransport(lose_acks_for=("draft",), lose_times={"draft": 1})
        adapter, _ = _connected_adapter()
        adapter._transport = t
        md = {"message_id": "m.3"}
        r1 = await adapter.send_draft("C1", 9, "partial", metadata=md)
        assert not r1.success
        key = adapter._draft_key("C1", md)
        assert adapter._open_draft_by_chat.get(key) == 9, (
            "ambiguous frame result disarmed interception (round-2 finding 1b)"
        )
        final = await adapter.send("C1", "complete", metadata=dict(md))
        assert final.success
        seals = [o for o in t.ops if o[0] == "draft" and o[1]]
        assert len(seals) == 1 and seals[0][2] == "complete"

    @pytest.mark.asyncio
    async def test_definite_rejection_still_disarms(self):
        """Sibling guard: an explicit non-ambiguous rejection keeps the
        round-1 disarm semantics."""

        class RejectTransport:
            def __init__(self):
                self.ops = []

            async def send_outbound(self, payload, platform=None):
                self.ops.append((payload.get("op"), bool(payload.get("final"))))
                if payload.get("op") == "draft":
                    return {"success": False, "error": "stream_gone"}
                return {"success": True, "message_id": "m"}

        adapter, _ = _connected_adapter()
        t = RejectTransport()
        adapter._transport = t
        md = {"message_id": "m.4"}
        r = await adapter.send_draft("C1", 10, "partial", metadata=md)
        assert not r.success
        assert not adapter._open_draft_by_chat, "definite rejection must disarm"
        await adapter.send("C1", "final", metadata=dict(md))
        assert t.ops[-1] == ("send", False)


class TestTransportTagsAmbiguity:
    @pytest.mark.asyncio
    async def test_ws_transport_timeout_result_is_tagged(self):
        """Source-of-truth check on the production transport: the ack
        timeout branch carries ambiguous=True; fail-fast branches do not."""
        from gateway.relay.ws_transport import WebSocketRelayTransport

        transport = WebSocketRelayTransport.__new__(WebSocketRelayTransport)
        transport._closing = False
        transport._ws = object()  # non-None so we reach the pending path
        transport._pending = {}
        transport._outbound_timeout_s = 0.01
        transport._bot_id_for = lambda p: None

        async def _send(frame):
            return None  # sent fine; ack never arrives

        transport._send = _send
        result = await transport.send_outbound({"op": "send"})
        assert result["success"] is False
        assert result.get("ambiguous") is True

        transport._closing = True
        result2 = await transport.send_outbound({"op": "send"})
        assert result2["success"] is False
        assert "ambiguous" not in result2, "fail-fast paths are definite"
