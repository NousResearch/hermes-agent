"""Regression tests: Discord VC voice replies on NON-streamed voice-input turns (#101185).

Companion to PR #101186. The base-adapter auto-TTS fallback (`_wants_auto_tts`) is not a
guarantee on every voice-channel path, and the runner's final dedup line
`return not (is_voice_input and not already_sent)` skips unconditionally for voice input.
When the bot is connected to the guild's voice channel, exactly one spoken reply per turn
is the contract — the runner must take it over when nothing else will play it.

The VC-membership guard must resolve through the text-channel binding + active mixer
(the same path that proves membership for the "One moment" tool ack), not only
`event.raw_message`: on a spoken turn the event is synthetic, and under multiplexing the
delivery adapter can differ from the one whose `_voice_clients` holds the connection.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

GUILD = "1539781321914581154"
TEXT_CHANNEL = "1539781322749255704"


def _make_runner() -> GatewayRunner:
    from unittest.mock import patch
    with patch("gateway.run.GatewayRunner._load_voice_modes", return_value={}):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
    return runner


def _make_vc_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.platform = Platform.DISCORD
    adapter._auto_tts_mode = "all"
    adapter.send_voice = AsyncMock()
    adapter.play_in_voice_channel = AsyncMock(return_value=True)
    adapter.is_in_voice_channel = MagicMock(return_value=True)
    adapter._voice_text_channels = {int(GUILD): int(TEXT_CHANNEL)}
    adapter.voice_mixer_active = MagicMock(return_value=True)
    return adapter


def _make_voice_event() -> MessageEvent:
    """Exactly the shape `_handle_voice_channel_input` builds for a spoken turn."""
    from types import SimpleNamespace
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.DISCORD, chat_id=TEXT_CHANNEL, user_id="u1", user_name="User",
        ),
        message_type=MessageType.VOICE,
        message_id="456",
        raw_message=SimpleNamespace(guild_id=int(GUILD), guild=None),
    )


class TestVcVoiceReplyNotStreamed:

    def test_vc_connected_voice_input_unstreamed_fires_runner_reply(self):
        """voice_only chat + bot in the VC + text NOT streamed -> runner speaks it.

        Today this returns False ("base adapter handles it") even when the base path
        never plays; #101186 fixes this shape but only via `_get_guild_id` +
        `is_in_voice_channel`.
        """
        runner = _make_runner()
        adapter = _make_vc_adapter()
        runner.adapters[Platform.DISCORD] = adapter
        runner._voice_mode[f"discord:{TEXT_CHANNEL}"] = "voice_only"

        event = _make_voice_event()
        assert runner._should_send_voice_reply(event, "hello", [], already_sent=False) is True

    def test_vc_membership_resolves_through_binding_when_raw_missing(self):
        """A spoken-turn source whose raw_message lacks the guild still resolves VC
        membership through the text-channel binding + active mixer (the ack path)."""
        runner = _make_runner()
        adapter = _make_vc_adapter()
        adapter.is_in_voice_channel = MagicMock(return_value=False)  # connection moved on
        runner.adapters[Platform.DISCORD] = adapter
        runner._voice_mode[f"discord:{TEXT_CHANNEL}"] = "voice_only"

        event = _make_voice_event()
        # Mixer gone / not connected: no spoken reply, and NO second text-channel voice note.
        assert runner._should_send_voice_reply(event, "hello", [], already_sent=False) is False

    def test_not_in_vc_keeps_existing_dedup(self):
        """No VC connection -> the upstream dedup (base adapter auto-TTS) still applies."""
        runner = _make_runner()
        adapter = _make_vc_adapter()
        adapter.is_in_voice_channel = MagicMock(return_value=False)
        adapter.voice_mixer_active = MagicMock(return_value=False)
        runner.adapters[Platform.DISCORD] = adapter
        runner._voice_mode[f"discord:{TEXT_CHANNEL}"] = "voice_only"

        event = _make_voice_event()
        assert runner._should_send_voice_reply(event, "hello", [], already_sent=False) is False

    def test_voice_off_never_speaks_even_in_vc(self):
        """/voice off stays a hard override even with the bot connected in the VC."""
        runner = _make_runner()
        adapter = _make_vc_adapter()
        runner.adapters[Platform.DISCORD] = adapter
        runner._voice_mode[f"discord:{TEXT_CHANNEL}"] = "off"

        event = _make_voice_event()
        assert runner._should_send_voice_reply(event, "hello", [], already_sent=False) is False
