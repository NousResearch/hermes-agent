"""Adapters without a native ``send_voice`` must not synthesize auto-TTS replies.

``voice.auto_tts`` is synced onto every adapter at connect. Email has no ``send_voice``
override — the base implementation is a friendly-failure notice — so a profile that
enables auto_tts for its Telegram voice lane made every email turn synthesize an MP3
and then email the sender "Couldn't deliver the audio attachment." alongside the
correct text reply (#141792). a2a already had both guards (#90103); these tests pin
the same two layers for every adapter whose class does not override ``send_voice``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _TextOnlyAdapter:
    """Email-like adapter: ``send_voice`` is the inherited base friendly-failure notice."""

    send_voice = BasePlatformAdapter.send_voice

    def __init__(self, platform: Platform):
        self.platform = platform
        self._auto_tts_default = True  # the pre-fix inherited state
        self._auto_tts_disabled_chats = set()
        self._auto_tts_enabled_chats = set()
        self._should_auto_tts_for_chat = MagicMock(return_value=True)


class TestSyncVoiceModeState:
    def test_email_never_inherits_global_auto_tts(self):
        runner = _make_runner()
        email_adapter = _TextOnlyAdapter(Platform.EMAIL)

        with patch(
            "hermes_cli.config.load_config", return_value={"voice": {"auto_tts": True}}
        ):
            runner._sync_voice_mode_state_to_adapter(email_adapter)

        assert email_adapter._auto_tts_default is False

    def test_a2a_stays_text(self):
        """#90103 regression: the explicit a2a guard keeps applying."""
        runner = _make_runner()
        a2a_adapter = _TextOnlyAdapter(Platform("a2a"))

        with patch(
            "hermes_cli.config.load_config", return_value={"voice": {"auto_tts": True}}
        ):
            runner._sync_voice_mode_state_to_adapter(a2a_adapter)

        assert a2a_adapter._auto_tts_default is False

    def test_platform_with_native_send_voice_still_inherits(self):
        """Regression: the structural gate must not silence voice-capable platforms."""
        runner = _make_runner()
        telegram_adapter = _make_adapter(Platform.TELEGRAM)
        telegram_adapter._auto_tts_disabled_chats = set()
        telegram_adapter._auto_tts_enabled_chats = set()

        with patch(
            "hermes_cli.config.load_config", return_value={"voice": {"auto_tts": True}}
        ):
            runner._sync_voice_mode_state_to_adapter(telegram_adapter)

        assert telegram_adapter._auto_tts_default is True


class TestShouldSendVoiceReply:
    def test_email_ignores_adapter_auto_tts(self):
        """The runner-side gate must hold even if the adapter-side default leaked in
        (e.g. an explicit ``/voice on`` opt-in, or state synced before this fix)."""
        runner = _make_runner()
        email_adapter = _TextOnlyAdapter(Platform.EMAIL)
        runner.adapters[Platform.EMAIL] = email_adapter
        event = _make_event(Platform.EMAIL, chat_id="sender@x")

        assert runner._should_send_voice_reply(event, "hello", []) is False

    def test_email_ignores_explicit_voice_all(self):
        """Email cannot deliver audio at all, so even a saved ``/voice tts`` mode must
        stay text — same semantics the a2a early exit already gives that platform."""
        runner = _make_runner()
        email_adapter = _TextOnlyAdapter(Platform.EMAIL)
        runner.adapters[Platform.EMAIL] = email_adapter
        event = _make_event(Platform.EMAIL, chat_id="sender@x")
        runner._voice_mode[runner._voice_key_for_source(event.source)] = "all"

        assert runner._should_send_voice_reply(event, "hello", []) is False

    def test_platform_with_native_send_voice_still_replies(self):
        """Regression: a voice-capable platform with mode ``all`` keeps its reply."""
        runner = _make_runner()
        telegram_adapter = _make_adapter(Platform.TELEGRAM)
        runner.adapters[Platform.TELEGRAM] = telegram_adapter
        event = _make_event(Platform.TELEGRAM, chat_id="123")
        runner._voice_mode[runner._voice_key_for_source(event.source)] = "all"

        assert runner._should_send_voice_reply(event, "hello", []) is True


def _make_runner() -> GatewayRunner:
    with patch("gateway.run.GatewayRunner._load_voice_modes", return_value={}):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
    return runner


def _make_adapter(platform: Platform) -> MagicMock:
    adapter = MagicMock()
    adapter.platform = platform
    adapter.send_voice = AsyncMock()
    adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
    return adapter


def _make_event(
    platform: Platform,
    chat_id: str = "123",
    message_type: MessageType = MessageType.TEXT,
) -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(
            platform=platform,
            chat_id=chat_id,
            user_id="u1",
            user_name="User",
        ),
        message_type=message_type,
        message_id="456",
    )
