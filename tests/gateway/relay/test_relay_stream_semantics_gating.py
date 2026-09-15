"""Regression: stream-is-the-message is a SLACK semantic, not a relay
semantic (PR 85796 review, B4).

The base send_draft contract is Telegram-shaped: the draft animates and
clears client-side, and the final answer arrives as a separate REAL send
that becomes the history message. Slack native streaming inverts this —
the stream IS the message and the turn-final seals it in place.

The relay adapter hardcoded draft_stream_is_message = True for every
descriptor, so a Telegram (or any non-Slack) connector advertising the
draft op had its turn-final intercepted into draft(final=true): no real
message was ever posted to the chat history. Live-probed on the review
branch (`platform telegram … ops [draft(final=False), draft(final=True)]`,
no send op).

Now the flag is gated on the negotiated descriptor platform. A future
platform with genuine stream-is-the-message semantics should advertise it
via the descriptor rather than widening the gate by guesswork.
"""

import pytest

from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class RecordingTransport:
    def __init__(self):
        self.ops = []

    async def send_outbound(self, payload, platform=None):
        self.ops.append(dict(payload))
        return {"success": True, "message_id": "m.1"}


class TestStreamIsMessageGating:
    def test_slack_descriptor_gets_stream_is_message(self):
        adapter, _ = _connected_adapter()  # platform="slack" default
        assert adapter.draft_stream_is_message is True

    def test_telegram_descriptor_does_not(self):
        adapter, _ = _connected_adapter(
            platform="telegram",
            markdown_dialect="markdown_v2",
            supported_ops=("send", "edit", "typing", "draft"),
        )
        assert adapter.draft_stream_is_message is False

    @pytest.mark.asyncio
    async def test_telegram_final_is_a_real_send_not_a_seal(self):
        """The B4 probe: telegram + draft op → frames go out as drafts,
        the final goes out as a REAL send (history message), never as
        draft(final=true)."""
        adapter, _ = _connected_adapter(
            platform="telegram",
            markdown_dialect="markdown_v2",
            supported_ops=("send", "edit", "typing", "draft"),
        )
        t = RecordingTransport()
        adapter._transport = t
        md = {"reply_to_message_id": "evt.1"}
        await adapter.send_draft("T1", 3, "partial", metadata=md)
        r = await adapter.send("T1", "complete answer", metadata=dict(md))
        assert r.success
        ops = [(o["op"], o.get("final")) for o in t.ops]
        assert ("draft", False) in ops
        assert ("send", None) in ops, ops
        assert not [o for o in t.ops if o["op"] == "draft" and o.get("final")], (
            "telegram-shaped connector must never receive draft(final=true) "
            "as the turn-final — the draft clears client-side and the real "
            "send is the history message"
        )

    @pytest.mark.asyncio
    async def test_slack_final_still_seals(self):
        """Sibling guard: the gate must not have broken the Slack lane."""
        adapter, _ = _connected_adapter()
        t = RecordingTransport()
        adapter._transport = t
        md = {"reply_to_message_id": "evt.2"}
        await adapter.send_draft("C1", 4, "partial", metadata=md)
        r = await adapter.send("C1", "complete answer", metadata=dict(md))
        assert r.success
        assert [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert not [o for o in t.ops if o["op"] == "send"]
