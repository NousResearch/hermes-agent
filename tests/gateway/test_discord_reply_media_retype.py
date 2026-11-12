"""Regression tests for #150824: replies to the bot's own media must not
inherit the replied-to attachment's message type or enter the user STT pipeline.

Two live flavors reproduced 2026-11-07 (build 0.23.0+701/9ffa7dff23):
- typed reply to the bot's native voice note -> event retyped VOICE -> the bot's
  own TTS transcribed and injected as user speech;
- typed reply to the bot's plain MP3 attachment -> event retyped AUDIO -> the
  bot's attachment rides along as user media.

Contracts:
- A: a typed reply to a BOT-authored message keeps MessageType.TEXT; the
  bot-originated referenced audio is not treated as user voice input.
- B: a typed reply to a HUMAN-authored media message still classifies from the
  referenced attachment (feature preserved, #46107): voice stays STT-eligible.
- C: a direct (non-reply) human voice note still classifies VOICE (no regression
  to the normal voice path).
- D: gateway-side guard: even if an event somehow carries a bot-replied audio,
  ``_event_media_is_stt_input`` rejects media on events that replied to the bot.
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import _event_media_is_stt_input


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
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

from plugins.platforms.discord.adapter import DiscordAdapter


def _att(content_type="audio/ogg", filename="voice-message.ogg", is_voice_message=True):
    """Duck-typed discord.Attachment with the native voice-note markers."""
    att = MagicMock()
    att.content_type = content_type
    att.filename = filename
    att.size = 161732
    att.is_voice_message = is_voice_message
    return att


def _resolved(author_bot, attachments):
    resolved = MagicMock()
    resolved.attachments = attachments
    resolved.author = SimpleNamespace(bot=author_bot, id="999")
    return resolved


def _message(reference=None, own_attachments=(), content="reply text"):
    msg = MagicMock()
    msg.content = content
    msg.attachments = list(own_attachments)
    msg.reference = reference
    return msg


def _classify(adapter, msg, snapshot_attachments=()):
    """Run the classification block from _handle_message via the extracted helper."""
    referenced = []
    reference = getattr(msg, "reference", None)
    resolved_reference = getattr(reference, "resolved", None) if reference else None
    if resolved_reference is not None:
        referenced = list(getattr(resolved_reference, "attachments", []) or [])
    own = list(msg.attachments) + list(snapshot_attachments)
    bot_referenced = (
        referenced
        if resolved_reference is not None and getattr(getattr(resolved_reference, "author", None), "bot", False)
        else []
    )
    return adapter._classify_reply_media(own, referenced, bot_referenced, msg.content)


class TestReplyToBotAudioKeepsText:
    """Contract A: typed reply to the bot's voice note stays TEXT, bot audio not STT-eligible."""

    def test_reply_to_bot_voice_note_keeps_text_and_drops_stt(self):
        adapter = DiscordAdapter.__new__(DiscordAdapter)  # no init; pure helper use
        ref = MagicMock()
        ref.resolved = _resolved(author_bot=True, attachments=[_att()])
        msg = _message(reference=ref)

        msg_type, _user_media, _bot_referenced = _classify(adapter, msg)

        assert msg_type == MessageType.TEXT
        # No bot-quoted audio may land in the user-media bucket (that is the STT-eligible one).
        assert _user_media == []

    def test_reply_to_bot_plain_audio_also_keeps_text(self):
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        ref = MagicMock()
        ref.resolved = _resolved(
            author_bot=True,
            attachments=[_att(content_type="audio/mpeg", filename="song.mp3", is_voice_message=False)],
        )
        msg = _message(reference=ref)

        msg_type, _user_media, _bot_ctx = _classify(adapter, msg)

        assert msg_type == MessageType.TEXT


class TestReplyToHumanMediaPreserved:
    """Contract B: #46107's feature - replies to human media still carry it."""

    def test_reply_to_human_voice_note_is_stt_eligible(self):
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        ref = MagicMock()
        ref.resolved = _resolved(author_bot=False, attachments=[_att()])
        msg = _message(reference=ref)

        msg_type, _user_media, _bot_ctx = _classify(adapter, msg)

        assert msg_type == MessageType.VOICE

    def test_reply_to_human_image_classifies_photo(self):
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        ref = MagicMock()
        ref.resolved = _resolved(
            author_bot=False,
            attachments=[_att(content_type="image/png", filename="pic.png", is_voice_message=False)],
        )
        msg = _message(reference=ref)

        msg_type, _user_media, _bot_ctx = _classify(adapter, msg)

        assert msg_type == MessageType.PHOTO


class TestDirectVoiceUnchanged:
    """Contract C: non-reply voice notes classify exactly as before."""

    def test_direct_voice_note_is_voice(self):
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        msg = _message(reference=None, own_attachments=[_att()])

        msg_type, _user_media, _bot_ctx = _classify(adapter, msg)

        assert msg_type == MessageType.VOICE


class TestGatewaySttGuard:
    """Contract D: defense in depth at the gateway boundary."""

    @staticmethod
    def _event(message_type, media_types, own_message):
        return MessageEvent(
            text="reply", message_type=message_type,
            media_urls=["/tmp/a.ogg"], media_types=media_types,
            reply_to_message_id="123", reply_to_is_own_message=own_message,
        )

    def test_voice_event_replied_to_bot_is_not_stt_input(self):
        ev = self._event(MessageType.VOICE, ["audio/ogg"], own_message=True)
        assert _event_media_is_stt_input(ev, 0) is False

    def test_voice_event_replied_to_human_is_stt_input(self):
        ev = self._event(MessageType.VOICE, ["audio/ogg"], own_message=False)
        assert _event_media_is_stt_input(ev, 0) is True

    def test_direct_voice_event_is_stt_input(self):
        ev = MessageEvent(
            text="hi", message_type=MessageType.VOICE,
            media_urls=["/tmp/a.ogg"], media_types=["audio/ogg"],
            reply_to_message_id=None,
        )
        assert _event_media_is_stt_input(ev, 0) is True
