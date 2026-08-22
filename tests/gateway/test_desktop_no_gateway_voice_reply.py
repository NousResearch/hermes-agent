"""The gateway must not send an auto-TTS voice reply to the desktop surface (#90297).

When `voice.auto_tts` is on, two independent TTS paths fire for the same
reply on the desktop: the gateway's `_send_voice_reply` (via
`adapter.send_voice`) and the desktop app's `useAutoSpeakReplies` hook (via
`/api/audio/speak` → `playSpeechText`). Neither knows about the other, so
every reply plays twice (one "Generating speech", two "TTS audio saved").
The desktop hook is the authoritative speaker for that surface; the gateway
side now steps out.

The desktop platform value is produced by the session routing layer at
runtime (it is not a static ``Platform`` enum member), so the tests stub a
platform object whose ``.value`` is ``"desktop"`` — exactly the comparison
``_should_send_voice_reply`` performs.
"""

from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _DesktopPlatform(str):
    """Hashable stand-in for the runtime desktop platform value.

    The desktop platform is produced by the session routing layer at
    runtime (not a static ``Platform`` enum member), so the tests stub a
    hashable object whose ``.value`` is ``"desktop"`` — exactly the
    comparison and dict-key usage ``_should_send_voice_reply`` performs.
    """

    @property
    def value(self) -> str:
        return self


def _desktop_platform() -> _DesktopPlatform:
    return _DesktopPlatform("desktop")


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    return runner


def _make_event(platform) -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(
            platform=platform,
            chat_id="123",
            user_id="u1",
            user_name="User",
        ),
        message_type=MessageType.TEXT,
        message_id="456",
    )


class TestDesktopNeverGetsGatewayVoiceReply:
    def test_desktop_platform_skipped_even_with_auto_tts_on(self):
        """The #90297 shape: global auto_tts on and the desktop would
        otherwise qualify — the gateway must still stay silent because the
        desktop's useAutoSpeakReplies speaks the same reply itself."""
        runner = _make_runner()
        adapter = MagicMock()
        adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
        runner.adapters[_desktop_platform()] = adapter
        runner._voice_mode = {}  # no explicit mode: adapter_auto_tts governs

        event = _make_event(_desktop_platform())
        assert (
            runner._should_send_voice_reply(event, "spoken twice?", [])
            is False
        )

    def test_desktop_skipped_with_explicit_all_mode(self):
        """Even an explicit /voice all for the chat must not double-speak on
        the desktop — the hook fires on every reply regardless of mode."""
        runner = _make_runner()
        runner.adapters["desktop"] = MagicMock()
        runner._voice_mode = {("desktop", "123"): "all"}

        event = _make_event(_desktop_platform())
        assert runner._should_send_voice_reply(event, "hello", []) is False

    def test_other_platforms_still_send(self):
        """Guard: the skip is desktop-only — Telegram with auto_tts on still
        gets its gateway voice reply."""
        runner = _make_runner()
        adapter = MagicMock()
        adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        runner._voice_mode = {}

        event = _make_event(Platform.TELEGRAM)
        assert runner._should_send_voice_reply(event, "hello", []) is True
