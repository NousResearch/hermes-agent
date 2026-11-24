"""Regression: stream semantics and draft capability resolve PER CHAT on
a multi-platform relay (PR 85796 review round 2, finding 2).

One RelayAdapter fronts N platforms (Phase 1.5): descriptors accumulate
per platform on the transport and outbound frames are tagged per chat.
The round-1 gate keyed draft_stream_is_message and
supports_draft_streaming() off the PRIMARY scalar descriptor only:

  - Slack primary + Telegram chat: the Telegram chat's turn-final was
    intercepted into draft(final=true) — no real Telegram history
    message (probed on the head);
  - Telegram primary + Slack chat: the Slack chat was denied native
    streaming entirely.

Both now resolve through _descriptor_for_chat (the same per-chat
machinery max_message_length already uses); the scalar remains the
fallback for unknown chats and single-platform gateways.
"""

import pytest

from tests.gateway.relay.test_relay_live_cards import _connected_adapter, make_desc


def _desc_for(platform, **kw):
    return make_desc(
        platform=platform,
        label=platform.title(),
        markdown_dialect="markdown_v2" if platform == "telegram" else "slack",
        max_message_length=4096,
        **kw,
    )


class RecordingTransport:
    def __init__(self, descriptors_by_platform=None):
        self.ops = []
        self._descs = dict(descriptors_by_platform or {})

    def descriptor_for_platform(self, platform):
        return self._descs.get(platform)

    async def send_outbound(self, payload, platform=None):
        self.ops.append(dict(payload))
        return {"success": True, "message_id": "ts.1"}


def _multi_adapter(primary="slack", others=("telegram",)):
    """Adapter with a primary descriptor + per-platform secondaries."""
    adapter, _ = _connected_adapter() if primary == "slack" else _connected_adapter(
        platform=primary, markdown_dialect="markdown_v2",
        supported_ops=("send", "edit", "typing", "draft"),
    )
    descs = {p: _desc_for(p) for p in (primary, *others)}
    t = RecordingTransport(descs)
    adapter._transport = t
    return adapter, t


class TestPerChatStreamSemantics:
    @pytest.mark.asyncio
    async def test_slack_primary_telegram_chat_gets_real_final(self):
        """The round-2 probe: telegram chat on a Slack-primary adapter →
        frames stream, the final is a REAL send, never a seal."""
        adapter, t = _multi_adapter(primary="slack", others=("telegram",))
        adapter._platform_by_chat["TG1"] = "telegram"
        md = {"message_id": "m.1"}
        await adapter.send_draft("TG1", 3, "partial", metadata=md)
        r = await adapter.send("TG1", "complete", metadata=dict(md))
        assert r.success
        assert not [o for o in t.ops if o["op"] == "draft" and o.get("final")], (
            "telegram chat sealed by the Slack primary's semantic"
        )
        assert [o for o in t.ops if o["op"] == "send"]

    @pytest.mark.asyncio
    async def test_slack_primary_slack_chat_still_seals(self):
        adapter, t = _multi_adapter(primary="slack", others=("telegram",))
        adapter._platform_by_chat["C1"] = "slack"
        md = {"message_id": "m.2"}
        await adapter.send_draft("C1", 4, "partial", metadata=md)
        r = await adapter.send("C1", "complete", metadata=dict(md))
        assert r.success
        assert [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert not [o for o in t.ops if o["op"] == "send"]

    def test_telegram_primary_slack_chat_gets_native_streaming(self):
        """Inverse starvation: a Telegram primary must not deny a
        secondary Slack chat its draft capability or seal semantics."""
        adapter, _t = _multi_adapter(primary="telegram", others=("slack",))
        adapter._platform_by_chat["C9"] = "slack"
        assert adapter.supports_draft_streaming(chat_id="C9") is True
        assert adapter.stream_is_message_for_chat("C9") is True
        # And the primary's own chats keep telegram semantics:
        adapter._platform_by_chat["TG9"] = "telegram"
        assert adapter.stream_is_message_for_chat("TG9") is False

    def test_unknown_chat_falls_back_to_scalar(self):
        adapter, _t = _multi_adapter(primary="slack", others=("telegram",))
        # Never seen inbound — scalar (slack primary) governs.
        assert adapter.stream_is_message_for_chat("UNSEEN") is True

    @pytest.mark.asyncio
    async def test_capability_gate_respects_chat_descriptor(self):
        """A chat whose platform descriptor lacks the draft op must raise
        NotImplementedError even when the primary advertises it."""
        adapter, t = _multi_adapter(primary="slack", others=())
        # Telegram descriptor WITHOUT the draft op:
        t._descs["telegram"] = _desc_for(
            "telegram", supported_ops=("send", "edit", "typing")
        )
        adapter._platform_by_chat["TG2"] = "telegram"
        with pytest.raises(NotImplementedError):
            await adapter.send_draft("TG2", 5, "x", metadata={})
