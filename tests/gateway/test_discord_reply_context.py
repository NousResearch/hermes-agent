"""Regression tests: Discord reply/forward context must reach the agent turn (#119230).

A reply (or reply to a forwarded message) must carry the referenced message's
author, channel, text, and attachment metadata into the agent turn — not just
a bare reply_to_id. Outbound replies must still anchor on the CURRENT inbound
message id, never the referenced target id.
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _TextChannel:
    """Fake Discord text channel (not a DM, not a Thread)."""

    def __init__(self, channel_id=100, name="general", guild_name="Test Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _empty():
            return
            yield
        return _empty()


def _make_message(*, msg_id=42, channel, content="explain this", reference=None,
                  attachments=None, author=None, msg_type=None):
    if author is None:
        author = SimpleNamespace(id=7, display_name="Alice", name="Alice", bot=False)
    return SimpleNamespace(
        id=msg_id,
        content=content,
        mentions=[],
        attachments=list(attachments or []),
        reference=reference,
        message_snapshots=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        guild=SimpleNamespace(id=1, name="Test Server"),
        type=(msg_type if msg_type is not None
              else discord_platform.discord.MessageType.reply),
    )


def _make_reference(*, message_id=11, channel_id=100, content="original text",
                    author_id=5, author_name="Bob", is_bot=False,
                    ref_channel_id=100, attachments=None):
    author = SimpleNamespace(id=author_id, display_name=author_name,
                             name=author_name, bot=is_bot)
    resolved = SimpleNamespace(
        id=message_id,
        content=content,
        author=author,
        channel=SimpleNamespace(id=ref_channel_id),
        attachments=list(attachments or []),
    )
    return SimpleNamespace(message_id=message_id, channel_id=channel_id,
                           resolved=resolved)


@pytest.fixture
def adapter(monkeypatch):
    for var in ("DISCORD_REQUIRE_MENTION", "DISCORD_AUTO_THREAD",
                "DISCORD_NO_THREAD_CHANNELS", "DISCORD_FREE_RESPONSE_CHANNELS",
                "DISCORD_ALLOWED_CHANNELS", "DISCORD_IGNORED_CHANNELS",
                "DISCORD_HISTORY_BACKFILL", "DISCORD_ALLOW_BOTS",
                "DISCORD_IGNORE_NO_MENTION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    config = PlatformConfig(enabled=True, token="***")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    a._text_batch_delay_seconds = 0
    a.handle_message = AsyncMock()
    return a


async def _dispatch(adapter, message):
    assert await adapter._handle_message(message) is True
    assert adapter.handle_message.call_count == 1
    return adapter.handle_message.call_args.args[0]


class TestDiscordReplyContext:
    @pytest.mark.asyncio
    async def test_reply_carries_author_channel_and_text(self, adapter):
        channel = _TextChannel(channel_id=100)
        ref = _make_reference(message_id=11, content="original text",
                              author_id=5, author_name="Bob",
                              ref_channel_id=100)
        event = await _dispatch(adapter, _make_message(channel=channel, reference=ref))
        assert event.reply_to_message_id == "11"
        assert event.reply_to_text == "original text"
        assert event.reply_to_author_id == "5"
        assert event.reply_to_author_name == "Bob"
        assert event.reply_to_channel_id == "100"
        assert event.reply_to_is_own_message is False

    @pytest.mark.asyncio
    async def test_reply_to_bot_message_marks_own(self, adapter):
        channel = _TextChannel(channel_id=100)
        ref = _make_reference(message_id=12, content="bot said this",
                              author_id=999, author_name="Hermes", is_bot=True)
        event = await _dispatch(adapter, _make_message(channel=channel, reference=ref))
        assert event.reply_to_is_own_message is True
        assert event.reply_to_author_id == "999"

    @pytest.mark.asyncio
    async def test_cross_channel_reply_preserves_origin_channel(self, adapter):
        channel = _TextChannel(channel_id=200)
        ref = _make_reference(message_id=13, channel_id=300, ref_channel_id=300,
                              content="forwarded thing", author_name="Carol")
        event = await _dispatch(adapter, _make_message(channel=channel, reference=ref))
        assert event.reply_to_channel_id == "300"
        assert event.reply_to_channel_id != "200"

    @pytest.mark.asyncio
    async def test_referenced_attachments_recorded_distinctly(self, adapter, monkeypatch):
        seen = {}

        async def fake_collect(all_attachments):
            seen["count"] = len(all_attachments)
            return [], [], [], None

        monkeypatch.setattr(adapter, "_collect_attachment_media", fake_collect)
        channel = _TextChannel(channel_id=100)
        att = SimpleNamespace(id=77, filename="report.pdf",
                              content_type="application/pdf", size=100, url="http://x/y")
        ref = _make_reference(message_id=14, content="see attached",
                              attachments=[att])
        event = await _dispatch(adapter, _make_message(channel=channel, reference=ref))
        # Referenced attachment is still delivered to the agent (re-hosted media path).
        assert seen["count"] == 1
        # ... but recorded distinctly from the current message's own attachments.
        assert event.reply_to_attachment_names == ["report.pdf"]

    @pytest.mark.asyncio
    async def test_plain_message_has_no_reply_context(self, adapter):
        channel = _TextChannel(channel_id=100)
        # Reply-typed but no reference (e.g. unresolvable): skips auto-thread
        # and must leave every reply field empty.
        msg = _make_message(channel=channel, reference=None)
        msg.reference = None
        event = await _dispatch(adapter, msg)
        assert event.reply_to_message_id is None
        assert event.reply_to_text is None
        assert event.reply_to_author_id is None
        assert event.reply_to_channel_id is None
        assert event.reply_to_attachment_names == []
        assert event.reply_to_is_own_message is False


class TestReplyContextRendering:
    def test_agent_turn_renders_author_channel_and_attachments(self):
        from gateway.platforms.event import MessageEvent
        from gateway.run_inbound import GatewayInboundMixin
        event = MessageEvent(
            text="explain this", reply_to_message_id="11",
            reply_to_text="original text", reply_to_author_id="5",
            reply_to_author_name="Bob", reply_to_channel_id="100",
            reply_to_attachment_names=["report.pdf"],
        )
        out = GatewayInboundMixin._prepend_inbound_reply_context(event, None, event.text)
        assert "Bob" in out
        assert "100" in out
        assert "original text" in out
        assert "report.pdf" in out
        assert out.endswith("explain this")

    def test_own_message_renders_previous_message(self):
        from gateway.platforms.event import MessageEvent
        from gateway.run_inbound import GatewayInboundMixin
        event = MessageEvent(
            text="thanks", reply_to_message_id="12",
            reply_to_text="bot said this", reply_to_is_own_message=True,
        )
        out = GatewayInboundMixin._prepend_inbound_reply_context(event, None, event.text)
        assert "your previous message" in out

    def test_legacy_text_only_format_unchanged(self):
        from gateway.platforms.event import MessageEvent
        from gateway.run_inbound import GatewayInboundMixin
        event = MessageEvent(text="q?", reply_to_message_id="42",
                             reply_to_text="quoted words")
        out = GatewayInboundMixin._prepend_inbound_reply_context(event, None, event.text)
        assert out.startswith('[Replying to: "quoted words"]')


class TestRelayWireMapping:
    def test_extended_reply_to_maps(self):
        from gateway.relay.ws_transport import _event_from_wire
        raw = {
            "text": "explain this", "message_id": "50",
            "source": {"platform": "discord", "chat_id": "100", "user_id": "7"},
            "reply_to_message_id": "11",
            "reply_to": {
                "message_id": "11", "channel_id": "300",
                "text": "forwarded thing",
                "author": {"id": "5", "name": "Carol"},
                "is_own": False,
                "attachments": [{"filename": "a.png"}],
            },
        }
        event = _event_from_wire(raw)
        assert event.reply_to_message_id == "11"
        assert event.reply_to_text == "forwarded thing"
        assert event.reply_to_author_id == "5"
        assert event.reply_to_author_name == "Carol"
        assert event.reply_to_channel_id == "300"
        assert event.reply_to_attachment_names == ["a.png"]

    def test_legacy_wire_still_maps(self):
        from gateway.relay.ws_transport import _event_from_wire
        raw = {
            "text": "hi", "message_id": "51",
            "source": {"platform": "discord", "chat_id": "100", "user_id": "7"},
            "reply_to_message_id": "9",
            "reply_to": {"text": "old", "author": "Dave", "is_own": True},
        }
        event = _event_from_wire(raw)
        assert event.reply_to_text == "old"
        assert event.reply_to_author_name == "Dave"
        assert event.reply_to_is_own_message is True


class TestOutboundAnchor:
    def test_anchor_uses_current_message_not_reference(self):
        from gateway.platforms.base import _reply_anchor_for_event
        from gateway.platforms.event import MessageEvent
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(platform=Platform.DISCORD, chat_id="100",
                               chat_type="group", user_id="7")
        event = MessageEvent(text="explain this", source=source,
                             message_id="50", reply_to_message_id="11",
                             reply_to_text="original text")
        assert _reply_anchor_for_event(event) == "50"
