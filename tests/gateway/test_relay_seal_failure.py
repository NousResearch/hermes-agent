"""Regression: seal-time transport failure must never lose or duplicate
the final answer (silent-loss class — PR 85796 review finding B1/B3-adjacent).

Three invariants:

1. A transport exception during the sealing draft frame retries the SAME
   idempotent frame once (the connector's sealed-key tombstone returns the
   original stream ts for a repeated final, so a retry can never open a
   second stream or post a duplicate).

2. When the seal cannot be delivered at all, the adapter reports failure —
   and the consumer must NOT record final delivery.  The prior behavior
   let the turn-final retry re-enter the draft-frame path, whose no-op
   dedupe compared against the last UNSEALED frame and reported success
   with zero transport calls: flags went green, the gateway suppressed its
   fallback, and the user never received the answer.

3. The turn-final retry path always calls with finalize=True so it can
   never be routed through the draft-frame branch at all.
"""

import asyncio

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class FlakyTransport:
    """Raises on selected ops; records everything that got through."""

    def __init__(self, raise_on=(), raise_times=None):
        self.ops = []
        self.attempts = []
        self.raise_on = set(raise_on)
        # op -> remaining raise count (None = raise forever)
        self.raise_times = dict(raise_times or {})

    def _should_raise(self, op, final):
        key = "seal" if (op == "draft" and final) else op
        if key not in self.raise_on:
            return False
        remaining = self.raise_times.get(key)
        if remaining is None:
            return True
        if remaining <= 0:
            return False
        self.raise_times[key] = remaining - 1
        return True

    async def send_outbound(self, payload, platform=None):
        op = payload.get("op")
        final = bool(payload.get("final"))
        self.attempts.append((op, final))
        if self._should_raise(op, final):
            raise ConnectionError("socket dropped mid-write")
        self.ops.append(dict(payload))
        return {"success": True, "message_id": "1723600000.42"}


def _armed_adapter(transport):
    adapter, _ = _connected_adapter()
    adapter._transport = transport
    return adapter


class TestSealTransportError:
    @pytest.mark.asyncio
    async def test_seal_exception_retries_same_idempotent_frame(self):
        """One transport drop during the seal → the SAME final frame is
        retried and its success (with the stream ts) is the send result."""
        t = FlakyTransport(raise_on=("seal",), raise_times={"seal": 1})
        adapter = _armed_adapter(t)
        md = {"thread_ts": "1700.1"}
        await adapter.send_draft("C1", 7, "answer", metadata=md)
        result = await adapter.send("C1", "answer complete", metadata=md)
        assert result.success
        assert result.message_id == "1723600000.42"
        seal_attempts = [a for a in t.attempts if a == ("draft", True)]
        assert len(seal_attempts) == 2, "seal must be retried exactly once"
        # Both attempts carried the identical frame (idempotent replay).
        finals = [o for o in t.ops if o.get("op") == "draft" and o.get("final")]
        assert len(finals) == 1 and finals[0]["content"] == "answer complete"
        # No plain send: a landed retry means no duplicate.
        assert not [o for o in t.ops if o.get("op") == "send"]

    @pytest.mark.asyncio
    async def test_seal_exception_twice_reports_failure(self):
        """Both attempts down → SendResult failure (fail-open plain send in
        send() then also fails on the dead transport; the caller sees the
        error instead of a phantom success)."""
        t = FlakyTransport(raise_on=("seal", "send"))
        adapter = _armed_adapter(t)
        md = {"thread_ts": "1700.2"}
        await adapter.send_draft("C1", 8, "answer", metadata=md)
        with pytest.raises(ConnectionError):
            # Seal fails twice -> falls through to plain send -> transport
            # still down -> the plain-send exception propagates (existing
            # plain-send contract; callers catch).
            await adapter.send("C1", "answer complete", metadata=md)
        # The seal was attempted twice, never recorded as delivered.
        assert [a for a in t.attempts if a == ("draft", True)] == [
            ("draft", True),
            ("draft", True),
        ]

    @pytest.mark.asyncio
    async def test_consumer_never_records_delivery_on_dead_transport(self):
        """THE silent-loss regression: transport dead at seal time → the
        consumer's delivery flags must stay False so the gateway's normal
        fallback send path still owns the final."""
        t = FlakyTransport(raise_on=("seal", "send"))
        adapter = _armed_adapter(t)
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(
            adapter, "C1", cfg, metadata={"thread_ts": "1700.3"},
        )
        task = asyncio.create_task(sc.run())
        sc.on_delta("complete answer")
        await asyncio.sleep(0.08)
        sc.finish("complete answer")
        await task
        # Frames streamed, but the final never reached the wire:
        assert [o for o in t.ops if o.get("op") == "draft" and not o.get("final")]
        assert not [o for o in t.ops if o.get("op") == "draft" and o.get("final")]
        assert not [o for o in t.ops if o.get("op") == "send"]
        # The consumer must not claim delivery — this is what previously
        # suppressed the gateway fallback and silently ate the answer.
        assert sc.final_response_sent is False
        assert getattr(sc, "_final_content_delivered", False) is False
        assert sc.delivered_final_matches("complete answer") is not True
