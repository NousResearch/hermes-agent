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

    def observe_inbound_message(self, event):
        self.observed.append(event)

    def reply_delivery_policy(self, event, response, *, voice_mode, already_sent):
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


def _event(message_type=MessageType.TEXT):
    return MessageEvent(
        text="hello",
        message_type=message_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
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
