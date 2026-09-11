"""Regression: the consumer-declared final + interim-send contract
(live findings, 2026-08-16 canary — the duplicate-final class).

Three invariants:

1. ``finish(final_text=...)`` makes the finalize payload the AUTHORITATIVE
   completed final_response — post-stream augmentation (file-mutation
   verifier footer) rides the seal/final edit instead of arriving via a
   separate corrective send (live finding #11).

2. Interim sends (commentary) from the consumer carry ``_interim_send``
   metadata, and the relay adapter's seal-interception ignores them — a
   mid-turn commentary must never seal the live native stream (which
   orphaned the true final into a plain-send duplicate).

3. The queued-follow-up lane reconciles an unconfirmed final by EDITING the
   consumer's delivered message in place, not by plain-sending a duplicate.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_draft_adapter():
    """BasePlatformAdapter subclass with stream-is-the-message drafts."""
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    A = type("StreamIsMsgAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 39000})
    A.__abstractmethods__ = frozenset()
    a = A.__new__(A)
    a._typing_paused = set()
    a._fatal_error_message = None
    a.draft_stream_is_message = True
    a.draft_calls = []
    a.send_calls = []
    a.edit_calls = []

    def _supports(chat_type=None, metadata=None):
        return True
    a.supports_draft_streaming = _supports

    async def _send_draft(*, chat_id, draft_id, content, metadata=None):
        a.draft_calls.append({"draft_id": draft_id, "content": content})
        return SendResult(success=True, message_id=None)
    a.send_draft = _send_draft

    async def _send(chat_id, content, reply_to=None, metadata=None, **kw):
        a.send_calls.append({"content": content, "metadata": dict(metadata or {})})
        return SendResult(success=True, message_id="sealed_ts_1")
    a.send = _send

    async def _edit(chat_id, message_id, content, finalize=False):
        a.edit_calls.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)
    a.edit_message = _edit
    return a


class TestConsumerDeclaredFinal:
    @pytest.mark.asyncio
    async def test_finish_final_text_rides_the_final_send(self):
        """The footer-bearing final_response must BE the finalize payload —
        one message, no separate corrective send needed."""
        adapter = _make_draft_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)

        task = asyncio.create_task(sc.run())
        sc.on_delta("streamed answer body")
        await asyncio.sleep(0.06)
        # Turn completes; turn_finalizer appended the verifier footer.
        final_with_footer = (
            "streamed answer body\n\n"
            "⚠️ File-mutation verifier: 1 file(s) were NOT modified this turn."
        )
        sc.finish(final_with_footer)
        await task

        # The turn-final send carried the COMPLETE footer-bearing final.
        assert adapter.send_calls, "expected a turn-final send"
        assert adapter.send_calls[-1]["content"] == final_with_footer
        # And the recorded payload reconciles → gateway suppression is safe.
        assert sc.delivered_final_matches(final_with_footer) is True

    @pytest.mark.asyncio
    async def test_finish_bare_keeps_legacy_behavior(self):
        adapter = _make_draft_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)
        task = asyncio.create_task(sc.run())
        sc.on_delta("plain answer")
        await asyncio.sleep(0.06)
        sc.finish()
        await task
        assert adapter.send_calls[-1]["content"] == "plain answer"


class TestInterimSendContract:
    @pytest.mark.asyncio
    async def test_commentary_is_marked_interim(self):
        adapter = _make_draft_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)
        ok = await sc._send_commentary("Using the browser tool…")
        assert ok is True
        assert adapter.send_calls[-1]["metadata"].get("_interim_send") is True

    @pytest.mark.asyncio
    async def test_relay_adapter_interim_send_does_not_seal(self):
        """An armed open draft must survive an interim send untouched."""
        from tests.gateway.relay.test_relay_live_cards import _connected_adapter

        adapter, _ = _connected_adapter(
            supported_ops=("send", "edit", "typing", "draft"),
        )

        class _T:
            def __init__(self):
                self.ops = []
            async def send_outbound(self, payload, platform=None):
                self.ops.append(dict(payload))
                return {"success": True, "message_id": "111.222"}

        t = _T()
        adapter._transport = t
        md = {"thread_ts": "1700.42"}
        await adapter.send_draft("C1", 5, "streaming...", metadata=md)
        key = adapter._draft_key("C1", md)
        assert adapter._open_draft_by_chat.get(key) == 5

        # Interim commentary while the stream is open: must NOT seal.
        res = await adapter.send(
            "C1", "delegation dispatched, continuing…",
            metadata={**md, "_interim_send": True},
        )
        assert res.success
        assert adapter._open_draft_by_chat.get(key) == 5, "interim send sealed the stream"
        sent_ops = [o for o in t.ops if o["op"] == "send"]
        assert len(sent_ops) == 1
        # Marker never leaks to the wire.
        assert "_interim_send" not in (sent_ops[0].get("metadata") or {})

        # The true turn-final still seals.
        final = await adapter.send("C1", "streaming... done.", metadata=md)
        assert final.success
        assert key not in adapter._open_draft_by_chat
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert len(seals) == 1

    @pytest.mark.asyncio
    async def test_send_for_platform_interim_does_not_seal(self):
        """The delivery-resolver egress door honors the interim contract too.

        send_for_platform is the second seal-interception site (finding #7);
        an interim send routed through it must neither seal the open stream
        nor leak the gateway-internal marker onto the wire.
        """
        from tests.gateway.relay.test_relay_live_cards import _connected_adapter

        adapter, _ = _connected_adapter(
            supported_ops=("send", "edit", "typing", "draft"),
        )

        class _T:
            def __init__(self):
                self.ops = []
                # fronts_platform reads the handshake identity set off the
                # transport; advertise slack so send_for_platform proceeds.
                self._identities = [("slack", "U-bot")]
            async def send_outbound(self, payload, platform=None):
                self.ops.append(dict(payload))
                return {"success": True, "message_id": "111.333"}

        t = _T()
        adapter._transport = t
        md = {"thread_ts": "1700.77"}
        await adapter.send_draft("C2", 9, "streaming...", metadata=md)
        key = adapter._draft_key("C2", md)
        assert adapter._open_draft_by_chat.get(key) == 9

        res = await adapter.send_for_platform(
            "slack", "C2", "interim status", metadata={**md, "_interim_send": True},
        )
        assert res.success
        assert adapter._open_draft_by_chat.get(key) == 9, "interim send_for_platform sealed the stream"
        sent_ops = [o for o in t.ops if o["op"] == "send"]
        assert len(sent_ops) == 1
        assert "_interim_send" not in (sent_ops[0].get("metadata") or {})


class TestFinalAdoptionGuards:
    @pytest.mark.asyncio
    async def test_no_stream_turn_does_not_adopt_final(self):
        """finish(final_text) on a turn that never streamed must not move
        delivery ownership into the consumer — the gateway's normal final
        send path owns those turns (non-streaming models, tool-only turns)."""
        adapter = _make_draft_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)
        task = asyncio.create_task(sc.run())
        # No on_delta at all — straight to completion with a payload.
        sc.finish("the final answer from a non-streaming turn")
        await task
        assert adapter.send_calls == [], "no-stream turn must not deliver via the consumer"
        assert adapter.draft_calls == []


class TestQueuedLaneReconcile:
    @pytest.mark.asyncio
    async def test_queued_first_response_edits_in_place(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        adapter = _make_draft_adapter()
        sc = GatewayStreamConsumer(adapter, "D1")
        sc._message_id = "sealed_ts_9"
        source = SimpleNamespace(chat_id="D1")
        await GatewayRunner._deliver_queued_first_response(
            runner,
            "the complete final with footer",
            source=source,
            adapter=adapter,
            metadata=None,
            text_already_delivered=False,
            deliver_media=False,
            stream_consumer=sc,
        )
        # Edited the sealed message; did NOT plain-send a duplicate.
        assert adapter.edit_calls == [
            {"message_id": "sealed_ts_9", "content": "the complete final with footer"}
        ]
        assert adapter.send_calls == []

    @pytest.mark.asyncio
    async def test_queued_first_response_falls_back_without_message(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        adapter = _make_draft_adapter()
        sc = SimpleNamespace(message_id=None, _turn_split_delivery=False)
        source = SimpleNamespace(chat_id="D1")
        await GatewayRunner._deliver_queued_first_response(
            runner,
            "final text",
            source=source,
            adapter=adapter,
            metadata=None,
            text_already_delivered=False,
            deliver_media=False,
            stream_consumer=sc,
        )
        assert adapter.edit_calls == []
        assert len(adapter.send_calls) == 1
