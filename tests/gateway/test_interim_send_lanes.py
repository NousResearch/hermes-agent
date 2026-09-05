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
    def test_marks_and_preserves(self):
        md = _interim_metadata({"thread_id": "t1", "notify": False})
        assert md["_interim_send"] is True
        assert md["thread_id"] == "t1"

    def test_none_input(self):
        assert _interim_metadata(None) == {"_interim_send": True}

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
