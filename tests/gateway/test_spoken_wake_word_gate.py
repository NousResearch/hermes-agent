"""A spoken wake word admits a group voice note under ``require_mention`` + ``mention_patterns``
(opt-in ``voice_mention_patterns``; #102330). The text gate runs before speech-to-text, so a
captionless voice note could never match. The turn reuses the gate's transcript: one STT call.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType

OGG = b"OggS" + b"\x01" * 64
PATTERNS = [r"(?i)\bhermes\b"]


def _stt(transcript):
    calls = []

    def transcribe(path, _model=None, _source=None):
        calls.append(path)
        return {"success": True, "transcript": transcript, "provider": "fake"}
    return transcribe, calls


def _telegram(voice_mention_patterns):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    extra = {"require_mention": True, "mention_patterns": PATTERNS, "allowed_chats": [], "allowed_topics": [],
             "group_allowed_chats": []}
    if voice_mention_patterns is not None:
        extra["voice_mention_patterns"] = voice_mention_patterns
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._is_callback_user_authorized = lambda *_a, **_k: True
    adapter.handle_message = AsyncMock()
    return adapter


def _voice_note():
    async def download():
        return bytearray(OGG)

    async def get_file():
        return SimpleNamespace(file_path="voice/file_1.oga", download_as_bytearray=download)

    return SimpleNamespace(
        message_id=42, text=None, caption=None, entities=[], caption_entities=[], message_thread_id=None,
        is_topic_message=False, chat=SimpleNamespace(id=-100, type="group", title="G", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Alice", first_name="Alice", is_bot=False),
        reply_to_message=None, date=None, media_group_id=None, sticker=None, photo=None, video=None,
        video_note=None, audio=None, document=None, forum_topic_created=None,
        voice=SimpleNamespace(file_id="v1", file_unique_id="uv1", duration=2, file_size=len(OGG), get_file=get_file))


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_in,spoken,admitted", [
    (True, "hermes, what is on today", True),
    (True, "just chatting about lunch", False),
    (None, "hermes, what is on today", False),  # off by default: no STT for unaddressed notes
], ids=["wake-word", "no-wake-word", "not-opted-in"])
async def test_telegram_group_voice_note_wakes_on_the_spoken_wake_word(tmp_path, monkeypatch, opt_in, spoken, admitted):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    transcribe, calls = _stt(spoken)
    adapter = _telegram(opt_in)
    msg = _voice_note()
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._handle_media_message(SimpleNamespace(message=msg, update_id=1), SimpleNamespace())

    assert adapter.handle_message.await_count == (1 if admitted else 0)
    assert len(calls) == (0 if opt_in is None else 1)
    if admitted:
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.VOICE and event.reply_expected is True
        assert len(event.media_urls) == 1 and event.media_types == ["audio/ogg"]


@pytest.mark.asyncio
async def test_the_turn_reuses_the_gate_transcript(tmp_path):
    """The runner's STT pass finds the gate's transcript by clip content (native admission
    re-captures the clip under a new path), so the provider is billed once."""
    from gateway.platforms.helpers import spoken_mention_clips
    from gateway.run_inbound_transcription import GatewayInboundTranscriptionMixin
    import re
    clip = tmp_path / "gate.ogg"
    clip.write_bytes(OGG)
    recaptured = tmp_path / "admitted.ogg"
    transcribe, calls = _stt("hermes, what is on today")
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        assert await spoken_mention_clips([str(clip)], [re.compile(p) for p in PATTERNS]) == [str(clip)]
        recaptured.write_bytes(clip.read_bytes())
        runner = GatewayInboundTranscriptionMixin()
        runner.config = SimpleNamespace(stt_enabled=True)
        text, transcripts = await runner._enrich_message_with_transcription("", [str(recaptured)])

    assert calls == [str(clip)]
    assert transcripts == ["hermes, what is on today"] and text == '"hermes, what is on today"'


@pytest.mark.asyncio
@pytest.mark.parametrize("spoken,admitted", [("hermes, what is on today", True), ("lunch?", False)])
async def test_whatsapp_group_voice_note_wakes_on_the_spoken_wake_word(tmp_path, monkeypatch, spoken, admitted):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    clip = tmp_path / "cache" / "audio" / "aud_1.ogg"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(OGG)
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True, "group_policy": "open"})
    adapter._dm_policy, adapter._group_policy, adapter._allow_from, adapter._group_allow_from = "pairing", "open", set(), set()
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._collect_bridge_media = AsyncMock(return_value=([str(clip)], ["audio/ogg"]))
    data = {"chatId": "123@g.us", "senderId": "1555@s.whatsapp.net", "isGroup": True, "body": "[ptt received]",
            "hasMedia": True, "mediaType": "ptt", "mime": "audio/ogg", "mediaUrls": [str(clip)], "messageId": "m1"}
    transcribe, calls = _stt(spoken)
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        event = await adapter._build_message_event(data)

    assert len(calls) == 1
    assert isinstance(event, MessageEvent) is admitted
    if admitted:
        assert event.media_urls == [str(clip)] and event.text == "" and event.reply_expected is True
    else:
        assert not clip.exists()  # fetched only for the check


@pytest.mark.asyncio
@pytest.mark.parametrize("spoken,admitted", [("hermes, what is on today", True), ("lunch?", False)])
async def test_photon_group_voice_note_wakes_on_the_spoken_wake_word(tmp_path, monkeypatch, spoken, admitted):
    import base64
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    adapter.handle_message = AsyncMock()
    transcribe, calls = _stt(spoken)
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._dispatch_inbound({
            "messageId": "v1", "space": {"id": "group-guid", "type": "group", "phone": None},
            "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z",
            "content": {"type": "voice", "name": "voice.ogg", "mimeType": "audio/ogg",
                        "data": base64.b64encode(OGG).decode()}})

    assert len(calls) == 1
    assert adapter.handle_message.await_count == (1 if admitted else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("spoken,admitted", [("hermes, what is on today", True), ("lunch?", False)])
async def test_bluebubbles_group_voice_memo_wakes_on_the_spoken_wake_word(tmp_path, monkeypatch, spoken, admitted):
    import asyncio
    import json
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://localhost:1234", "password": "secret", "require_mention": True,
        "mention_patterns": PATTERNS, "voice_mention_patterns": True, "send_read_receipts": False}))
    clip = tmp_path / "memo.ogg"
    clip.write_bytes(OGG)
    adapter._download_attachment = AsyncMock(return_value=str(clip))
    adapter.handle_message = AsyncMock()
    body = json.dumps({"type": "new-message", "data": {
        "guid": "m1", "text": "", "handle": {"address": "+15555550100"}, "isFromMe": False, "isGroup": True,
        "chats": [{"guid": "iMessage;+;group-chat"}], "attachments": [{"guid": "a1", "mimeType": "audio/ogg"}]}}).encode()
    request = SimpleNamespace(query={"password": "secret"}, headers={}, read=AsyncMock(return_value=body))
    transcribe, calls = _stt(spoken)
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._handle_webhook(request)
        await asyncio.sleep(0)

    assert len(calls) == 1
    assert adapter.handle_message.await_count == (1 if admitted else 0)
