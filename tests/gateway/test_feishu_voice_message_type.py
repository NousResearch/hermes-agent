"""Regression tests for Feishu native voice-note classification.

Lark's native ``audio`` msg_type is an in-app voice recording (uploaded
audio files arrive as ``file``/``media`` and normalize to ``document``). It
must be classified as MessageType.VOICE so the gateway auto-transcribes it
(Opus → STT), the same way Discord/DingTalk/Telegram do. Before the fix it
resolved to MessageType.AUDIO, which the gateway treats as a non-transcribed
file attachment — so a Feishu voice note silently reached the agent as
untranscribable audio. Follow-up to #28993 (Discord + DingTalk).
"""

from gateway.platforms.base import MessageType
from plugins.platforms.feishu.adapter import FeishuAdapter, FeishuNormalizedMessage


def _resolve(preferred: str, media_types):
    """Call _resolve_normalized_message_type without a full adapter init.

    The method only reads normalized.preferred_message_type and delegates to
    the static _resolve_media_message_type — no instance state — so we bypass
    __init__ (which needs Lark credentials/config) via __new__.
    """
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    normalized = FeishuNormalizedMessage(
        raw_type=preferred,
        text_content="",
        preferred_message_type=preferred,
    )
    return adapter._resolve_normalized_message_type(normalized, media_types)


def test_native_voice_audio_is_classified_as_voice():
    """Lark audio msg_type (voice recording) → VOICE, so it gets transcribed."""
    assert _resolve("audio", ["audio/opus"]) is MessageType.VOICE


def test_native_voice_audio_without_media_type_is_voice():
    """A voice note with no resolved mime still classifies as VOICE."""
    assert _resolve("audio", []) is MessageType.VOICE


def test_photo_and_document_unaffected():
    """The fix is scoped to the audio branch — other types are unchanged."""
    assert _resolve("photo", ["image/png"]) is MessageType.PHOTO
    assert _resolve("document", ["application/pdf"]) is MessageType.DOCUMENT
    assert _resolve("text", []) is MessageType.TEXT
