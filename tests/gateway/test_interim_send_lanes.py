"""Regression: every gateway-side mid-turn send lane is marked interim
(PR 85796 review, B5 + heartbeat/inactivity findings).

Seal-interception on stream-is-the-message adapters converts the first
unmarked send to an armed (chat, turn) key into draft(final=true). The
consumer's own interim lanes (commentary, tail flush) were marked, but
four gateway-side lanes that fire DURING a streaming turn were not:

  - long-running heartbeat ("⏳ Working — N min"), default every 180s
  - inactivity warning ("⚠️ No activity …")
  - plain-text approval fallback (button lane failed)
  - background-review notice

Live probe: the 3-minute heartbeat sealed the live stream with status
text; the real final then posted as a duplicate, and later frames were
silently swallowed by the seal tombstone.

These tests pin the helper's contract and the adapter-level behavior for
a heartbeat-shaped send; the lane wiring itself is a one-line metadata
wrap per call site (grep `_interim_metadata(` in gateway/run.py).
"""

import pytest

from gateway.run import _interim_metadata
from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class RecordingTransport:
    def __init__(self):
        self.ops = []

    async def send_outbound(self, payload, platform=None):
        self.ops.append(dict(payload))
        return {"success": True, "message_id": "m.1"}


class TestInterimMetadataHelper:


    def test_does_not_mutate_input(self):
        src = {"thread_id": "t1"}
        _interim_metadata(src)
        assert "_interim_send" not in src


class TestHeartbeatShapedSendDoesNotSeal:
    @pytest.mark.asyncio
    async def test_status_send_with_interim_marker_leaves_stream_open(self):
        """The exact live-probe scenario, fixed: an armed stream + a
        heartbeat-shaped status send marked via _interim_metadata → the
        stream stays open, the status posts as its own message, and the
        true turn-final still seals with the answer."""
        adapter, _ = _connected_adapter()
        t = RecordingTransport()
        adapter._transport = t
        md = {"thread_ts": "1700.5", "message_id": "1700.501"}
        await adapter.send_draft("C1", 7, "partial answer", metadata=md)

        r = await adapter.send(
            "C1",
            "⏳ Working — 3 min — terminal",
            metadata=_interim_metadata(dict(md)),
        )
        assert r.success
        # Status went out as a plain send; no seal, marker stripped.
        sends = [o for o in t.ops if o["op"] == "send"]
        assert len(sends) == 1
        assert "_interim_send" not in (sends[0].get("metadata") or {})
        assert not [o for o in t.ops if o["op"] == "draft" and o.get("final")]

        # The real final still seals with the ANSWER, not the status text.
        rf = await adapter.send("C1", "partial answer, done.", metadata=dict(md))
        assert rf.success
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert len(seals) == 1
        assert seals[0]["content"] == "partial answer, done."


class TestStatusLaneDoesNotSeal:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("event_type", ["lifecycle", "warn"])
    async def test_agent_status_mid_stream_leaves_stream_open(self, tmp_path, monkeypatch, event_type):
        """The agent status lane (``TurnRunner._status_callback_sync``) fires mid-turn —
        keepalive retries, fallback restores, interrupted streams. The relay adapter has no
        ``send_or_update_status``, so the status lands on ``adapter.send``; unmarked, it sealed
        the live Slack stream with the status text and the answer posted as a duplicate."""
        import asyncio
        from types import SimpleNamespace

        from gateway.config import Platform
        from gateway.run_turn_runner import TurnRunner
        from gateway.session import SessionSource
        from gateway.turn_context import TurnContext

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter, _ = _connected_adapter()
        t = RecordingTransport()
        adapter._transport = t
        md = {"thread_ts": "1700.5", "message_id": "1700.501"}
        await adapter.send_draft("C1", 7, "partial answer", metadata=md)

        source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel", thread_id="1700.5")
        ctx = TurnContext(source=source, user_config={}, _run_still_current=lambda: True,
                          _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata=dict(md))
        turn = TurnRunner(SimpleNamespace(), ctx)
        loop = asyncio.get_running_loop()
        scheduled = []
        turn._schedule = lambda coro, *_: scheduled.append(asyncio.run_coroutine_threadsafe(coro, loop)) or scheduled[-1]

        await asyncio.to_thread(turn._status_callback_sync, event_type,
                                "Stream interrupted - using delivered content as final response")
        await asyncio.gather(*(asyncio.wrap_future(f) for f in scheduled))

        sends = [o for o in t.ops if o["op"] == "send"]
        assert len(sends) == 1
        assert "_interim_send" not in (sends[0].get("metadata") or {})
        assert not [o for o in t.ops if o["op"] == "draft" and o.get("final")]

        rf = await adapter.send("C1", "partial answer, done.", metadata=dict(md))
        assert rf.success
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert [s["content"] for s in seals] == ["partial answer, done."]
