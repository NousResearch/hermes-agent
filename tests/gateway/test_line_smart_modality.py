from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
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


def test_line_env_enablement_seeds_smart_modality_toggle(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
    monkeypatch.setenv("LINE_SMART_MODALITY", "true")

    assert _env_enablement()["smart_modality"] == "true"
