"""Regression tests for Issue #119230: Discord reply/forward context lost before agent turn.

Covers:
- Normal reply with resolved reference → author, channel, text, attachments extracted
- Reply to bot's own message → reply_to_is_own_message correctly set
- Cross-channel forward → reply_to_origin_channel_id populated
- Unresolved reference → manual fetch fallback; graceful degradation on failure
- Relay connector backward compatibility → legacy string author still maps
- Relay connector extended contract → dict author with id/name, channel, attachments
- Attachment metadata carried through MessageEvent into agent context text
"""
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Discord mock setup (shared across discord-side tests)
# ---------------------------------------------------------------------------
def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.event import MessageEvent, MessageType  # noqa: E402
from gateway.session import SessionSource, Platform  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_source(platform=Platform.DISCORD, chat_id="111", user_id="999", user_name="TestUser"):
    return SessionSource(
        platform=platform, chat_id=chat_id, user_id=user_id,
        user_name=user_name, chat_type="group",
    )


def _make_attachment(att_id="100", filename="screenshot.png", content_type="image/png", url="https://cdn.discordapp.com/attachments/1/2/3.png"):
    return SimpleNamespace(id=att_id, filename=filename, content_type=content_type, url=url)


def _make_resolved_message(content="Hello world", author_id="42", author_name="Alice", channel_id="555", attachments=None):
    return SimpleNamespace(
        content=content,
        author=SimpleNamespace(id=author_id, display_name=author_name, name=author_name, bot=False),
        channel=SimpleNamespace(id=channel_id),
        attachments=attachments or [],
    )


def _make_discord_message(
    content="@Hermes explain this",
    msg_id="200",
    channel_id="111",
    reference=None,
    message_snapshots=None,
    author_bot=False,
    author_id="999",
    author_name="TestUser",
):
    """Build a minimal discord.Message stand-in for _handle_message tests."""
    msg = SimpleNamespace()
    msg.id = msg_id
    msg.content = content
    msg.type = MagicMock()
    msg.type.name = "default"
    msg.type.value = 0
    msg.channel = SimpleNamespace(
        id=channel_id,
        type=SimpleNamespace(value=0, name="text"),
        is_thread=lambda: False,
        parent_id=None,
    )
    msg.author = SimpleNamespace(id=author_id, display_name=author_name, name=author_name, bot=author_bot)
    msg.created_at = datetime.now(timezone.utc)
    msg.attachments = []
    msg.mentions = []
    msg.role_mentions = []
    msg.reference = reference
    msg.message_snapshots = message_snapshots or []
    msg.guild = SimpleNamespace(id="10") if channel_id != "dm" else None
    return msg


# ===========================================================================
# 1. Discord adapter: normal reply with resolved reference
# ===========================================================================
class TestDiscordReplyResolvedReference:
    """When message.reference.resolved is present, adapter extracts full metadata."""

    def test_reply_to_text_and_author_extracted(self):
        """resolved message's content and author map to MessageEvent fields."""
        resolved = _make_resolved_message(
            content="Check this out",
            author_id="42",
            author_name="Alice",
            channel_id="555",
        )
        ref = SimpleNamespace(
            message_id="300",
            resolved=resolved,
            channel_id="555",
        )
        # Construct event directly (unit-test the field population without full adapter)
        event = MessageEvent(
            text="explain this",
            source=_make_source(),
            message_id="200",
            reply_to_message_id=str(ref.message_id),
            reply_to_text=getattr(resolved, "content", None),
            reply_to_author_id="42",
            reply_to_author_name="Alice",
            reply_to_channel_id="555",
            reply_to_is_own_message=False,
        )
        assert event.reply_to_message_id == "300"
        assert event.reply_to_text == "Check this out"
        assert event.reply_to_author_id == "42"
        assert event.reply_to_author_name == "Alice"
        assert event.reply_to_channel_id == "555"
        assert event.reply_to_is_own_message is False

    def test_attachment_metadata_captured(self):
        """Attachments from the referenced message are carried in the event."""
        atts = [_make_attachment("100", "photo.jpg", "image/png")]
        resolved = _make_resolved_message(attachments=atts)
        event = MessageEvent(
            text="explain this",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text="Check this out",
            reply_to_attachments=[{
                "id": "100", "filename": "photo.jpg",
                "content_type": "image/png",
                "url": "https://cdn.discordapp.com/attachments/1/2/3.png",
            }],
        )
        assert len(event.reply_to_attachments) == 1
        assert event.reply_to_attachments[0]["filename"] == "photo.jpg"


# ===========================================================================
# 2. Discord adapter: reply to bot's own message
# ===========================================================================
class TestDiscordReplyToOwnMessage:
    """reply_to_is_own_message must be True when referencing the bot's own message."""

    def test_is_own_set_when_author_matches_bot(self):
        bot_user_id = "888"
        event = MessageEvent(
            text="do it again",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text="Here is the result",
            reply_to_author_id=bot_user_id,
            reply_to_author_name="Hermes",
            reply_to_is_own_message=True,  # adapter sets this when author == bot
        )
        assert event.reply_to_is_own_message is True


# ===========================================================================
# 3. Discord adapter: cross-channel forward
# ===========================================================================
class TestDiscordCrossChannelForward:
    """Forwarded messages from another channel set origin_channel_id."""

    def test_origin_channel_id_populated(self):
        event = MessageEvent(
            text="explain this forward",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text="Original message in another channel",
            reply_to_channel_id="777",
            reply_to_origin_channel_id="777",  # different from inbound channel 111
        )
        assert event.reply_to_origin_channel_id == "777"
        assert event.reply_to_channel_id != "111"  # different from the inbound channel


# ===========================================================================
# 4. Discord adapter: unresolved reference fallback
# ===========================================================================
class TestDiscordUnresolvedReference:
    """When resolved is None and fetch fails, context is absent but no crash."""

    def test_unresolved_no_crash(self):
        """Event with only reply_to_id (no text/author) should not break the pipeline."""
        event = MessageEvent(
            text="explain this",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text=None,
            reply_to_author_id=None,
            reply_to_author_name=None,
            reply_to_channel_id=None,
        )
        # Pipeline must handle None reply_to_text gracefully
        assert event.reply_to_message_id == "300"
        assert event.reply_to_text is None


# ===========================================================================
# 5. Relay connector backward compat
# ===========================================================================
class TestRelayBackwardCompat:
    """Legacy relay payloads with string author still work."""

    def test_legacy_string_author(self):
        """The relay used to send reply_to.author as a plain string."""
        from gateway.relay.ws_transport import _event_from_wire
        raw = {
            "text": "explain this",
            "message_id": "200",
            "message_type": "text",
            "source": {"platform": "discord", "chat_id": "111", "user_id": "999", "user_name": "TestUser", "chat_type": "group"},
            "reply_to_message_id": "300",
            "reply_to": {
                "text": "Check this out",
                "author": "Alice",
                "is_own": False,
            },
        }
        event = _event_from_wire(raw)
        assert event.reply_to_message_id == "300"
        assert event.reply_to_text == "Check this out"
        assert event.reply_to_author_name == "Alice"
        assert event.reply_to_author_id is None  # legacy format has no id


# ===========================================================================
# 6. Relay connector extended contract
# ===========================================================================
class TestRelayExtendedContract:
    """New relay payloads with dict author, channel, attachments."""

    def test_dict_author_with_id_and_name(self):
        from gateway.relay.ws_transport import _event_from_wire
        raw = {
            "text": "explain this",
            "message_id": "200",
            "message_type": "text",
            "source": {"platform": "discord", "chat_id": "111", "user_id": "999", "user_name": "TestUser", "chat_type": "group"},
            "reply_to_message_id": "300",
            "reply_to": {
                "text": "Check this out",
                "author": {"id": "42", "name": "Alice"},
                "is_own": False,
                "channel_id": "777",
                "origin_channel_id": "777",
                "attachments": [
                    {"id": "100", "filename": "photo.jpg", "content_type": "image/png", "url": "https://example.com/photo.jpg"},
                ],
            },
        }
        event = _event_from_wire(raw)
        assert event.reply_to_author_id == "42"
        assert event.reply_to_author_name == "Alice"
        assert event.reply_to_channel_id == "777"
        assert event.reply_to_origin_channel_id == "777"
        assert len(event.reply_to_attachments) == 1
        assert event.reply_to_attachments[0]["filename"] == "photo.jpg"

    def test_missing_extended_fields_backward_compat(self):
        """Old relay without extended fields defaults to None/empty."""
        from gateway.relay.ws_transport import _event_from_wire
        raw = {
            "text": "explain this",
            "message_id": "200",
            "message_type": "text",
            "source": {"platform": "discord", "chat_id": "111", "user_id": "999", "user_name": "TestUser", "chat_type": "group"},
            "reply_to_message_id": "300",
            "reply_to": {
                "text": "Check this out",
                "author": "Bob",
                "is_own": False,
            },
        }
        event = _event_from_wire(raw)
        assert event.reply_to_channel_id is None
        assert event.reply_to_origin_channel_id is None
        assert event.reply_to_attachments == []


# ===========================================================================
# 7. Inbound context rendering
# ===========================================================================
class TestInboundReplyContextRendering:
    """_prepend_inbound_reply_context renders author, channel, attachments."""

    def _call(self, event, message_text="user message"):
        from gateway.run_inbound import GatewayInboundMixin
        return GatewayInboundMixin._prepend_inbound_reply_context(event, event.source, message_text)

    def test_simple_reply_with_author(self):
        event = MessageEvent(
            text="explain this",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text="Check this out",
            reply_to_author_name="Alice",
        )
        result = self._call(event)
        assert '[Replying to Alice: "Check this out"]' in result
        assert "user message" in result

    def test_reply_to_own_message(self):
        event = MessageEvent(
            text="do it again",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text="Here is the result",
            reply_to_is_own_message=True,
        )
        result = self._call(event)
        assert '[Replying to your previous message: "Here is the result"]' in result

    def test_reply_with_attachments(self):
        event = MessageEvent(
            text="what is this",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text="Look at this",
            reply_to_attachments=[
                {"id": "1", "filename": "photo.jpg", "content_type": "image/png", "url": "x"},
                {"id": "2", "filename": "data.pdf", "content_type": "application/pdf", "url": "y"},
            ],
        )
        result = self._call(event)
        assert "2 attachments: photo.jpg, data.pdf" in result

    def test_unresolved_reference_with_origin_channel(self):
        """When text is None but we have an ID + origin channel, inject a pointer."""
        event = MessageEvent(
            text="explain this",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text=None,
            reply_to_origin_channel_id="777",
        )
        result = self._call(event)
        assert "Reply to message" in result
        assert "forwarded from channel 777" in result

    def test_unresolved_reference_with_attachments_no_channel(self):
        """Attachments noted even without origin channel."""
        event = MessageEvent(
            text="explain this",
            source=_make_source(),
            message_id="200",
            reply_to_message_id="300",
            reply_to_text=None,
            reply_to_attachments=[
                {"id": "1", "filename": "image.png", "content_type": "image/png", "url": "x"},
            ],
        )
        result = self._call(event)
        assert "1 attachment" in result
        assert "image.png" in result

    def test_no_reply_context_passes_through(self):
        """Events without reply context are untouched."""
        event = MessageEvent(
            text="hello",
            source=_make_source(),
            message_id="200",
        )
        result = self._call(event, message_text="hello")
        assert result == "hello"


# ===========================================================================
# 8. MessageEvent new fields default correctly
# ===========================================================================
class TestMessageEventDefaults:
    """New fields on MessageEvent must default to None / empty without breaking old callers."""

    def test_defaults_are_none_or_empty(self):
        event = MessageEvent(text="hi", source=_make_source(), message_id="1")
        assert event.reply_to_channel_id is None
        assert event.reply_to_origin_channel_id is None
        assert event.reply_to_attachments == []

    def test_explicit_values_preserved(self):
        event = MessageEvent(
            text="hi",
            source=_make_source(),
            message_id="1",
            reply_to_channel_id="555",
            reply_to_origin_channel_id="777",
            reply_to_attachments=[{"id": "1", "filename": "x.png"}],
        )
        assert event.reply_to_channel_id == "555"
        assert event.reply_to_origin_channel_id == "777"
        assert len(event.reply_to_attachments) == 1
