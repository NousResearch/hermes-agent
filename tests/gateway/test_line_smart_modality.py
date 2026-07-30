import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.line.adapter import LineAdapter, _env_enablement


_LINE_PLATFORM = LineAdapter(PlatformConfig(enabled=True, extra={})).platform


def _adapter(*, smart_modality=True):
    return LineAdapter(
        PlatformConfig(
            enabled=True,
            extra={"smart_modality": smart_modality},
        )
    )


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={_LINE_PLATFORM: PlatformConfig(enabled=True, extra={})}
    )
    runner.adapters = {_LINE_PLATFORM: adapter}
    runner._voice_mode = {}
    return runner


def _event(message_type, *, chat_id="Cline", user_id="Uline", chat_type="group"):
    return MessageEvent(
        text="hello",
        message_type=message_type,
        message_id=f"m-{message_type.value}",
        source=SessionSource(
            platform=_LINE_PLATFORM,
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
        ),
    )


def test_line_voice_turn_requests_voice_reply_without_voice_mode_toggle():
    adapter = _adapter()
    event = _event(MessageType.VOICE)

    policy = adapter.reply_delivery_policy(event, "hi", voice_mode="off", already_sent=True)

    assert policy.send_voice_reply is True
    assert policy.suppress_text_if_voice_reply_sent is True


def test_line_text_turn_stays_text_without_voice_mode_toggle():
    adapter = _adapter()
    event = _event(MessageType.TEXT)

    policy = adapter.reply_delivery_policy(event, "hi", voice_mode="off", already_sent=True)

    assert policy.send_voice_reply is False
    assert policy.suppress_text_if_voice_reply_sent is False


def test_line_photo_inherits_last_non_image_modality_per_group_user():
    adapter = _adapter()
    voice_event = _event(MessageType.VOICE, chat_id="Croom", user_id="Uvoice")
    adapter.observe_inbound_message(voice_event)

    same_user_photo = _event(MessageType.PHOTO, chat_id="Croom", user_id="Uvoice")
    other_user_photo = _event(MessageType.PHOTO, chat_id="Croom", user_id="Utext")

    same_policy = adapter.reply_delivery_policy(
        same_user_photo, "image answer", voice_mode="off", already_sent=True
    )
    other_policy = adapter.reply_delivery_policy(
        other_user_photo, "image answer", voice_mode="off", already_sent=True
    )

    assert same_policy.send_voice_reply is True
    assert same_policy.suppress_text_if_voice_reply_sent is True
    assert other_policy.send_voice_reply is False


def test_line_smart_modality_disabled_uses_default_voice_mode_gate():
    adapter = _adapter(smart_modality=False)
    voice_event = _event(MessageType.VOICE)

    off_policy = adapter.reply_delivery_policy(
        voice_event, "hi", voice_mode="off", already_sent=True
    )
    voice_policy = adapter.reply_delivery_policy(
        voice_event, "hi", voice_mode="voice_only", already_sent=True
    )

    assert off_policy.send_voice_reply is False
    assert off_policy.suppress_text_if_voice_reply_sent is False
    assert voice_policy.send_voice_reply is True
    assert voice_policy.suppress_text_if_voice_reply_sent is False


def test_line_smart_modality_env_overrides_config(monkeypatch):
    adapter = _adapter(smart_modality=False)
    voice_event = _event(MessageType.VOICE)

    monkeypatch.setenv("LINE_SMART_MODALITY", "true")
    assert adapter.reply_delivery_policy(
        voice_event, "hi", voice_mode="off", already_sent=True
    ).send_voice_reply is True

    monkeypatch.setenv("LINE_SMART_MODALITY", "false")
    assert adapter.reply_delivery_policy(
        voice_event, "hi", voice_mode="off", already_sent=True
    ).send_voice_reply is False


def test_line_voice_turn_gates_text_streaming_before_content_is_emitted():
    """Regression: with streaming enabled, streamed text is marked
    already_sent once delivered and can't be retracted, so a smart-voice
    turn must disable streaming up front (voice + duplicate text bug)."""
    adapter = _adapter()
    runner = _runner(adapter)

    runner._observe_inbound_message(_event(MessageType.VOICE))
    assert runner._suppress_text_streaming_for_voice(_event(MessageType.VOICE).source) is True

    # A text turn from the same participant re-enables streaming.
    runner._observe_inbound_message(_event(MessageType.TEXT))
    assert runner._suppress_text_streaming_for_voice(_event(MessageType.TEXT).source) is False


def test_line_streaming_not_gated_when_smart_modality_disabled():
    adapter = _adapter(smart_modality=False)
    runner = _runner(adapter)

    runner._observe_inbound_message(_event(MessageType.VOICE))
    assert runner._suppress_text_streaming_for_voice(_event(MessageType.VOICE).source) is False


@pytest.mark.asyncio
async def test_streaming_enabled_smart_voice_turn_has_no_duplicate_text(monkeypatch, tmp_path):
    """Integration-style walk of the streaming-enabled smart-voice path:
    the streaming gate fires before content is emitted, the reply goes out
    as exactly one voice message, and no text is ever delivered."""
    adapter = _adapter()
    runner = _runner(adapter)
    event = _event(MessageType.VOICE)

    sent_text = []
    sent_voice = []

    async def fake_send(chat_id, content, reply_to=None, metadata=None):
        sent_text.append(content)
        return SendResult(success=True, message_id="t1")

    async def fake_send_voice(**kwargs):
        sent_voice.append(kwargs)
        return SendResult(success=True, message_id="v1")

    monkeypatch.setattr(adapter, "send", fake_send)
    monkeypatch.setattr(adapter, "send_voice", fake_send_voice)

    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"audio")

    def fake_tts(*, text, output_path):
        return '{"success": true, "file_path": "' + str(audio_path) + '"}'

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)
    monkeypatch.setattr("tools.tts_tool._strip_markdown_for_tts", lambda text: text)

    # 1. Inbound dispatch observes the voice turn (runs before the agent).
    runner._observe_inbound_message(event)

    # 2. Streaming setup consults the gate and disables text streaming, so
    #    the final text is never delivered by the stream consumer and the
    #    turn ends with already_sent=False.
    streaming_gated = runner._suppress_text_streaming_for_voice(event.source)
    assert streaming_gated is True
    already_sent = False

    # 3. Final delivery: voice reply sent, text suppressed by policy.
    response = "final answer"
    assert runner._should_send_voice_reply(event, response, [], already_sent=already_sent) is True
    voice_sent = await runner._send_voice_reply(event, response)
    assert voice_sent is True
    if runner._should_suppress_text_after_voice_reply(
        event, response, voice_sent, already_sent=already_sent
    ):
        response = None
    if response is not None:
        await adapter.send(event.source.chat_id, response)

    assert len(sent_voice) == 1
    assert sent_text == []


def test_line_env_enablement_seeds_smart_modality_toggle(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
    monkeypatch.setenv("LINE_SMART_MODALITY", "true")

    assert _env_enablement()["smart_modality"] == "true"
