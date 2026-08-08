import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ReplyDeliveryPolicy, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _PolicyAdapter:
    def __init__(self, policy):
        self.policy = policy
        self.observed = []
        self.sent_voice = []
        self.policy_calls = []

    def observe_inbound_message(self, event):
        self.observed.append(event)

    def reply_delivery_policy(self, event, response, *, voice_mode, already_sent):
        self.policy_calls.append(event)
        return self.policy

    async def send_voice(self, **kwargs):
        self.sent_voice.append(kwargs)
        return SendResult(success=True, message_id="voice-1")


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    platform = Platform.TELEGRAM
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={})}
    )
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    return runner


def _event(message_type=MessageType.TEXT, profile=None):
    return MessageEvent(
        text="hello",
        message_type=message_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
            profile=profile,
        ),
    )


def test_runner_uses_adapter_reply_delivery_policy_for_voice_decision():
    adapter = _PolicyAdapter(ReplyDeliveryPolicy(send_voice_reply=True))
    runner = _runner(adapter)
    event = _event(MessageType.TEXT)

    assert runner._should_send_voice_reply(event, "hi", [], already_sent=False) is True


def test_runner_ignores_non_policy_adapter_return_and_preserves_legacy_gate():
    adapter = _PolicyAdapter(object())
    runner = _runner(adapter)
    event = _event(MessageType.TEXT)

    assert runner._should_send_voice_reply(event, "hi", [], already_sent=False) is False


def test_multiplex_source_resolves_secondary_profile_adapter_policy():
    """Regression: a secondary-profile source must consult its own adapter's
    policy, not the default profile's (see 8a9bc38c)."""
    default_adapter = _PolicyAdapter(ReplyDeliveryPolicy(send_voice_reply=False))
    secondary_adapter = _PolicyAdapter(ReplyDeliveryPolicy(send_voice_reply=True))
    runner = _runner(default_adapter)
    runner._profile_adapters = {"lars": {Platform.TELEGRAM: secondary_adapter}}
    event = _event(MessageType.TEXT, profile="lars")

    assert runner._should_send_voice_reply(event, "hi", [], already_sent=False) is True
    assert secondary_adapter.policy_calls == [event]
    assert default_adapter.policy_calls == []


def test_multiplex_source_observes_secondary_profile_adapter():
    default_adapter = _PolicyAdapter(ReplyDeliveryPolicy())
    secondary_adapter = _PolicyAdapter(ReplyDeliveryPolicy())
    runner = _runner(default_adapter)
    runner._profile_adapters = {"lars": {Platform.TELEGRAM: secondary_adapter}}
    event = _event(MessageType.TEXT, profile="lars")

    runner._observe_inbound_message(event)

    assert secondary_adapter.observed == [event]
    assert default_adapter.observed == []


def test_policy_callback_failure_falls_back_to_legacy_delivery(caplog):
    """A raising policy callback must not disrupt the final reply: the legacy
    voice-mode gate is used instead and the error is logged."""

    class _RaisingPolicyAdapter(_PolicyAdapter):
        def reply_delivery_policy(self, event, response, *, voice_mode, already_sent):
            raise RuntimeError("policy exploded")

    adapter = _RaisingPolicyAdapter(None)
    runner = _runner(adapter)
    event = _event(MessageType.TEXT)

    with caplog.at_level("WARNING", logger="gateway.run"):
        # Legacy gate with voice_mode off: no voice reply, no exception.
        assert runner._should_send_voice_reply(event, "hi", [], already_sent=False) is False
        # Text delivery is never suppressed when the policy callback fails.
        assert (
            runner._should_suppress_text_after_voice_reply(event, "hi", True, already_sent=False)
            is False
        )

    assert any("reply_delivery_policy failed" in r.getMessage() for r in caplog.records)

    # Legacy gate still honors an explicit /voice all opt-in.
    runner._voice_mode = {runner._voice_key(Platform.TELEGRAM, "chat-1"): "all"}
    assert runner._should_send_voice_reply(event, "hi", [], already_sent=False) is True


def test_runner_observes_inbound_message_before_dispatch():
    adapter = _PolicyAdapter(ReplyDeliveryPolicy())
    runner = _runner(adapter)
    event = _event(MessageType.TEXT)

    runner._observe_inbound_message(event)

    assert adapter.observed == [event]


@pytest.mark.asyncio
async def test_send_voice_reply_reports_success(monkeypatch, tmp_path):
    adapter = _PolicyAdapter(ReplyDeliveryPolicy())
    runner = _runner(adapter)
    event = _event(MessageType.TEXT)
    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr("gateway.run.text_to_speech_tool", None, raising=False)

    def fake_tts(*, text, output_path):
        return '{"success": true, "file_path": "' + str(audio_path) + '"}'

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)
    monkeypatch.setattr("tools.tts_tool._strip_markdown_for_tts", lambda text: text)

    assert await runner._send_voice_reply(event, "hello") is True
    assert adapter.sent_voice


def test_default_policy_preserves_voice_mode_gate():
    from gateway.platforms.base import BasePlatformAdapter

    class _DefaultPolicyAdapter(BasePlatformAdapter):
        async def connect(self):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True)

        async def get_chat_info(self, chat_id):
            return {}

    adapter = _DefaultPolicyAdapter(PlatformConfig(enabled=True, extra={}), Platform.TELEGRAM)
    event = _event(MessageType.VOICE)

    off = adapter.reply_delivery_policy(event, "hi", voice_mode="off", already_sent=True)
    voice_only_not_streamed = adapter.reply_delivery_policy(
        event, "hi", voice_mode="voice_only", already_sent=False
    )
    voice_only_streamed = adapter.reply_delivery_policy(
        event, "hi", voice_mode="voice_only", already_sent=True
    )

    assert off.send_voice_reply is False
    assert voice_only_not_streamed.send_voice_reply is False
    assert voice_only_streamed.send_voice_reply is True
