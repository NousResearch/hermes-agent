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
    assert transcripts == ["hermes, what is on today"] and text.startswith('"hermes, what is on today"')
    assert "gate.ogg" not in text  # the note is the turn's own, never the gate copy's


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
    # Bridge-cached: the bridge's quotedMediaCache still points here for a later quoted reply.
    assert clip.exists()


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
@pytest.mark.parametrize("spoken,admitted", [("hermes, what is on today", True), ("just chatting about lunch", False)])
async def test_photon_grouped_voice_payload_matches_spoken_wake_word(tmp_path, monkeypatch, spoken, admitted):
    """Photon's ``group`` envelope can carry the voice item (with a photo beside it): only the audio is
    transcribed, and a rejected envelope leaves none of its cached files behind."""
    import base64
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    adapter.handle_message = AsyncMock()
    transcribe, calls = _stt(spoken)
    content = {"type": "group", "items": [
        {"content": {"type": "attachment", "name": "p.png", "mimeType": "image/png", "data": base64.b64encode(b"\x89PNG").decode()}},
        {"content": {"type": "voice", "name": "voice.ogg", "mimeType": "audio/ogg", "data": base64.b64encode(OGG).decode()}}]}
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._dispatch_inbound({
            "messageId": "grouped-v1", "space": {"id": "group-guid", "type": "group", "phone": None},
            "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z", "content": content})

    assert len(calls) == 1 and calls[0].endswith(".ogg")
    assert adapter.handle_message.await_count == int(admitted)
    cached = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix in (".png", ".ogg")]
    assert len(cached) == (2 if admitted else 0)


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
    adapter.set_authorization_check(lambda *a, **k: True)  # the runner wires one at startup
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


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", [False, None])
async def test_a_bluebubbles_memo_from_an_unauthorized_sender_is_never_fetched_or_transcribed(
        tmp_path, monkeypatch, verdict):
    """Only the gateway's explicit True permits the fetch: a refused or unwired sender's memo is
    neither downloaded nor sent to STT, so the spoken gate drops it even when it says the wake word."""
    import asyncio
    import json
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://localhost:1234", "password": "secret", "require_mention": True,
        "mention_patterns": PATTERNS, "voice_mention_patterns": True, "send_read_receipts": False}))
    adapter.set_authorization_check(None if verdict is None else (lambda *a, **k: verdict))
    adapter._download_attachment = AsyncMock(return_value=str(tmp_path / "memo.ogg"))
    adapter.handle_message = AsyncMock()
    body = json.dumps({"type": "new-message", "data": {
        "guid": "m1", "text": "", "handle": {"address": "+155****0100"}, "isFromMe": False, "isGroup": True,
        "chats": [{"guid": "iMessage;+;group-chat"}], "attachments": [{"guid": "a1", "mimeType": "audio/ogg"}]}}).encode()
    request = SimpleNamespace(query={"password": "secret"}, headers={}, read=AsyncMock(return_value=body))
    transcribe, calls = _stt("hermes, what is on today")
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._handle_webhook(request)
        await asyncio.sleep(0)

    adapter._download_attachment.assert_not_awaited()
    assert calls == []
    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_disabled_stt_never_transcribes_at_the_gate_even_with_local_fallback(tmp_path, monkeypatch):
    """``stt.enabled: false`` is not a provider failure: the local fallback must not run either."""
    import re
    from gateway.platforms.helpers import spoken_mention_clips
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("stt:\n  enabled: false\n")
    clip = tmp_path / "gate.ogg"
    clip.write_bytes(OGG)
    local = []

    def fallback(path, _model=None):
        local.append(path)
        return {"success": True, "transcript": "hermes, what is on today", "provider": "local"}
    with patch("tools.transcription_tools.transcribe_audio_local_fallback", side_effect=fallback):
        assert await spoken_mention_clips([str(clip)], [re.compile(p) for p in PATTERNS]) == []
    assert local == []


@pytest.mark.asyncio
async def test_telegram_gate_transcribes_in_the_routed_profile(tmp_path, monkeypatch):
    """A shared bot's chat routed to another profile: the gate's STT runs in that profile's scope
    (its ``stt`` config and secrets), the scope the turn enters, never the receiving bot's."""
    import contextlib
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    launch, routed = tmp_path / "launch", tmp_path / "routed"
    for home, enabled in ((launch, "false"), (routed, "true")):
        home.mkdir()
        (home / "config.yaml").write_text(f"stt:\n  enabled: {enabled}\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    entered = []

    @contextlib.asynccontextmanager
    async def scope(source):
        entered.append(source.chat_id)
        token = set_hermes_home_override(str(routed))
        try:
            yield
        finally:
            reset_hermes_home_override(token)
    homes = []

    def transcribe(path, _model=None, _source=None):
        homes.append(str(get_hermes_home()))
        return {"success": True, "transcript": "hermes, what is on today", "provider": "fake"}
    adapter = _telegram(True)
    adapter.gateway_runner = SimpleNamespace(_async_profile_scope_for_source=scope,
                                             _profile_name_for_source=lambda *_a, **_k: None)
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._handle_media_message(SimpleNamespace(message=_voice_note(), update_id=1), SimpleNamespace())

    assert entered == ["-100"] and homes == [str(routed)]
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_photon_gate_stt_never_holds_the_inbound_reader(tmp_path, monkeypatch):
    """A slow group voice note delays only its own chat: another chat's message (a /stop) is
    dispatched meanwhile, and a later message in the SAME chat still arrives after the note."""
    import asyncio
    import base64
    import json
    import threading
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    seen = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: seen.append(e.message_id))
    release = threading.Event()

    def slow(path, _model=None, _source=None):
        release.wait(10)
        return {"success": True, "transcript": "hermes, what is on today", "provider": "fake"}

    def line(mid, space, content):
        return json.dumps({"messageId": mid, "space": space, "sender": {"id": "+15551234567"},
                           "timestamp": "2026-05-14T19:06:32.000Z", "content": content})
    group = {"id": "group-guid", "type": "group", "phone": None}
    dm = {"id": "dm-guid", "type": "dm", "phone": "+15551234567"}
    voice = {"type": "voice", "name": "v.ogg", "mimeType": "audio/ogg", "data": base64.b64encode(OGG).decode()}
    with patch("tools.transcription_tools.transcribe_audio", side_effect=slow):
        await asyncio.wait_for(adapter._on_inbound_line(line("v1", group, voice)), 2)
        await asyncio.wait_for(adapter._on_inbound_line(line("t1", group, {"type": "text", "text": "hermes later"})), 2)
        await asyncio.wait_for(adapter._on_inbound_line(line("s1", dm, {"type": "text", "text": "/stop"})), 2)
        assert seen == ["s1"]
        release.set()
        for _ in range(200):
            if len(seen) == 3:
                break
            await asyncio.sleep(0.02)
    assert seen == ["s1", "v1", "t1"]


@pytest.mark.asyncio
async def test_the_gate_honours_the_runners_stt_setting(tmp_path, monkeypatch):
    """Legacy ``gateway.json`` ``stt_enabled: false`` turns STT off for the turn
    (``GatewayConfig.stt_enabled``); the gate reads that same value, so the note is never sent to STT."""
    import re
    from gateway.platforms.helpers import spoken_mention_clips
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    clip = tmp_path / "gate.ogg"
    clip.write_bytes(OGG)
    adapter = SimpleNamespace(gateway_runner=SimpleNamespace(config=SimpleNamespace(stt_enabled=False)))
    transcribe, calls = _stt("hermes, what is on today")
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        assert await spoken_mention_clips([str(clip)], [re.compile(p) for p in PATTERNS], adapter=adapter) == []
    assert calls == []


@pytest.mark.asyncio
async def test_a_gate_transcript_never_crosses_profiles(tmp_path):
    """The same forwarded clip in a chat routed to another profile is transcribed with THAT profile's
    STT settings: the gate cache is keyed by profile as well as by clip content."""
    import re
    from gateway.platforms.helpers import spoken_mention_clips
    from gateway.run_inbound_transcription import GatewayInboundTranscriptionMixin
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    clip = tmp_path / "gate.ogg"
    clip.write_bytes(OGG)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    transcribe, calls = _stt("hermes, what is on today")
    runner = GatewayInboundTranscriptionMixin()
    runner.config = SimpleNamespace(stt_enabled=True)
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        token = set_hermes_home_override(str(tmp_path / "a"))
        try:
            assert await spoken_mention_clips([str(clip)], [re.compile(p) for p in PATTERNS], keep=frozenset({str(clip)}))
        finally:
            reset_hermes_home_override(token)
        token = set_hermes_home_override(str(tmp_path / "b"))
        try:
            await runner._enrich_message_with_transcription("", [str(clip)])
        finally:
            reset_hermes_home_override(token)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_the_turn_notes_name_its_own_audio_never_the_gate_copy(tmp_path):
    """The gate caches transcript text only. A clip whose gate STT failed (the note admitted on
    another clip) is transcribed again by the turn, and every note is built from the turn's paths:
    the gate's copy is gone by then (native admission re-captures, a routed profile rehomes)."""
    import re
    from gateway.platforms.helpers import spoken_mention_clips
    from gateway.run_inbound_transcription import GatewayInboundTranscriptionMixin
    gate = [tmp_path / "gate_1.ogg", tmp_path / "gate_2.ogg"]
    gate[0].write_bytes(OGG)
    gate[1].write_bytes(OGG + b"2")
    calls = []

    def transcribe(path, _model=None, _source=None):
        calls.append(path)
        with open(path, "rb") as fh:
            if fh.read().endswith(b"2"):
                return {"success": False, "error": "provider down"}
        return {"success": True, "transcript": "hermes, what is on today", "provider": "fake"}
    runner = GatewayInboundTranscriptionMixin()
    runner.config = SimpleNamespace(stt_enabled=True)
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe), \
            patch("tools.transcription_tools.transcribe_audio_local_fallback", return_value={"success": False}):
        assert await spoken_mention_clips([str(p) for p in gate], [re.compile(p) for p in PATTERNS])
        turn = [tmp_path / "turn_1.ogg", tmp_path / "turn_2.ogg"]
        for src, dst in zip(gate, turn):
            dst.write_bytes(src.read_bytes())
            src.unlink()
        text, _ = await runner._enrich_message_with_transcription("", [str(p) for p in turn])

    assert calls[2:] == [str(turn[1])]
    assert "gate_" not in text and "turn_2.ogg" in text


@pytest.mark.asyncio
async def test_a_sender_the_runner_refuses_is_never_transcribed(tmp_path):
    """An open group forwards members the allowlist refuses; the runner refuses them only after the
    gate, so the gate asks the runner's own authorization first and never spends STT on them."""
    import re
    from gateway.platforms.helpers import spoken_mention_clips
    clip = tmp_path / "gate.ogg"
    clip.write_bytes(OGG)
    asked = []
    runner = SimpleNamespace(config=SimpleNamespace(stt_enabled=True),
                             _is_user_authorized_for_source=lambda src: asked.append(src.user_id) or False)
    adapter = SimpleNamespace(gateway_runner=runner, _canonicalize=lambda _src: None)
    transcribe, calls = _stt("hermes, what is on today")
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        assert await spoken_mention_clips([str(clip)], [re.compile(p) for p in PATTERNS], adapter=adapter,
                                          source=SimpleNamespace(user_id="blocked")) == []
    assert asked == ["blocked"] and calls == []


@pytest.mark.asyncio
async def test_photon_caf_attachment_voice_note_wakes_on_the_spoken_wake_word(tmp_path, monkeypatch):
    """iMessage voice notes also arrive as ``attachment`` payloads named ``.caf``: the gate treats
    them as voice exactly as normalisation does."""
    import base64
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    adapter.handle_message = AsyncMock()
    transcribe, calls = _stt("hermes, what is on today")
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._dispatch_inbound({
            "messageId": "caf1", "space": {"id": "group-guid", "type": "group", "phone": None},
            "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z",
            "content": {"type": "attachment", "name": "Audio Message.caf", "mimeType": "audio/x-caf",
                        "data": base64.b64encode(b"caff" + b"\x00" * 60).decode()}})

    assert len(calls) == 1 and adapter.handle_message.await_count == 1
    assert adapter.handle_message.await_args.args[0].message_type == MessageType.VOICE


@pytest.mark.asyncio
async def test_photon_document_first_envelope_still_sends_its_spoken_audio_to_stt(tmp_path, monkeypatch):
    """A group envelope whose first item is a document normalises to DOCUMENT, which the runner never
    transcribes; admitted by speech, its audio must still reach the turn's STT pass."""
    import base64
    from gateway.run_inbound import GatewayInboundMixin
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    adapter.handle_message = AsyncMock()
    transcribe, _calls = _stt("hermes, read this")
    content = {"type": "group", "items": [
        {"content": {"type": "attachment", "name": "a.pdf", "mimeType": "application/pdf", "data": base64.b64encode(b"%PDF").decode()}},
        {"content": {"type": "voice", "name": "voice.ogg", "mimeType": "audio/ogg", "data": base64.b64encode(OGG).decode()}}]}
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._dispatch_inbound({
            "messageId": "dv1", "space": {"id": "group-guid", "type": "group", "phone": None},
            "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z", "content": content})

    event = adapter.handle_message.await_args.args[0]
    _images, stt_paths, audio_files, _videos = GatewayInboundMixin._classify_inbound_media(event, False)
    assert len(stt_paths) == 1 and stt_paths[0].endswith(".ogg") and audio_files == []


@pytest.mark.asyncio
async def test_a_group_control_command_never_waits_on_an_earlier_voice_check(tmp_path, monkeypatch):
    """A same-group ``/stop`` sent while an earlier voice note is still at gate STT is dispatched at
    once (the base adapter's guards route it); ordinary text still waits behind the note."""
    import asyncio
    import base64
    import json
    import threading
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    seen = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: seen.append(e.message_id))
    release = threading.Event()

    def slow(path, _model=None, _source=None):
        release.wait(10)
        return {"success": True, "transcript": "hermes, what is on today", "provider": "fake"}

    def line(mid, content):
        return json.dumps({"messageId": mid, "space": {"id": "group-guid", "type": "group", "phone": None},
                           "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z", "content": content})
    voice = {"type": "voice", "name": "v.ogg", "mimeType": "audio/ogg", "data": base64.b64encode(OGG).decode()}
    with patch("tools.transcription_tools.transcribe_audio", side_effect=slow):
        await asyncio.wait_for(adapter._on_inbound_line(line("v1", voice)), 2)
        await asyncio.wait_for(adapter._on_inbound_line(line("t1", {"type": "text", "text": "hermes later"})), 2)
        # Photon's group mention gate needs the wake word on a command too.
        await asyncio.wait_for(adapter._on_inbound_line(line("s1", {"type": "text", "text": "/stop hermes"})), 2)
        assert seen == ["s1"]
        release.set()
        for _ in range(200):
            if len(seen) == 3:
                break
            await asyncio.sleep(0.02)
    assert seen == ["s1", "v1", "t1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [
    {"type": "attachment", "name": "p.png", "mimeType": "image/png", "data": "iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="},
    {"type": "attachment", "name": "a.pdf", "mimeType": "application/pdf", "data": "JVBERi0xLjQ="},
])
async def test_a_busy_session_sends_only_the_admitted_envelopes_audio_to_stt(tmp_path, monkeypatch, first):
    """An envelope speech admitted is VOICE; when it lands on a busy session (interrupt, steer, queue
    drain) only its audio enters STT, never the photo or PDF beside it."""
    import base64
    from gateway.run_inbound_transcription import GatewayInboundTranscriptionMixin
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    adapter.handle_message = AsyncMock()
    transcribe, _calls = _stt("hermes, look at this")
    content = {"type": "group", "items": [{"content": first}, {"content": {
        "type": "voice", "name": "v.ogg", "mimeType": "audio/ogg", "data": base64.b64encode(OGG).decode()}}]}
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._dispatch_inbound({
            "messageId": "mx1", "space": {"id": "group-guid", "type": "group", "phone": None},
            "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z", "content": content})

    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == MessageType.VOICE and len(event.media_urls) == 2
    stt_paths = GatewayInboundTranscriptionMixin._pending_event_audio_paths(None, event)
    assert len(stt_paths) == 1 and stt_paths[0].endswith(".ogg")


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new hermes", "/undo hermes", "/status hermes"])
async def test_a_session_command_keeps_its_place_behind_an_earlier_voice_note(tmp_path, monkeypatch, command):
    """Only the running-agent escape hatches skip the per-chat order: a ``/new``, ``/undo`` or
    ``/status`` sent after a voice note still at gate STT runs after that note, as on Telegram."""
    import asyncio
    import base64
    import json
    import threading
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    seen = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: seen.append(e.message_id))
    release = threading.Event()

    def slow(path, _model=None, _source=None):
        release.wait(10)
        return {"success": True, "transcript": "hermes, what is on today", "provider": "fake"}

    def line(mid, content):
        return json.dumps({"messageId": mid, "space": {"id": "group-guid", "type": "group", "phone": None},
                           "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z", "content": content})
    voice = {"type": "voice", "name": "v.ogg", "mimeType": "audio/ogg", "data": base64.b64encode(OGG).decode()}
    with patch("tools.transcription_tools.transcribe_audio", side_effect=slow):
        await asyncio.wait_for(adapter._on_inbound_line(line("v1", voice)), 2)
        await asyncio.wait_for(adapter._on_inbound_line(line("c1", {"type": "text", "text": command})), 2)
        await asyncio.wait_for(adapter._on_inbound_line(line("t1", {"type": "text", "text": "hermes later"})), 2)
        assert seen == []  # an inline dispatch would have landed during _on_inbound_line
        release.set()
        for _ in range(200):
            if len(seen) == 3:
                break
            await asyncio.sleep(0.02)
    assert seen == ["v1", "c1", "t1"]


@pytest.mark.asyncio
async def test_a_photon_caf_sent_as_octet_stream_is_transcribed_as_the_voice_it_counts_as(tmp_path, monkeypatch):
    """A ``.caf`` attachment with a generic ``application/octet-stream`` MIME counts as voice, so it
    reaches gate STT (and the turn's) like any other voice note instead of being dropped unheard."""
    import base64
    from gateway.run_inbound import GatewayInboundMixin
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    adapter.handle_message = AsyncMock()
    transcribe, calls = _stt("hermes, what is on today")
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._dispatch_inbound({
            "messageId": "caf2", "space": {"id": "group-guid", "type": "group", "phone": None},
            "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z",
            "content": {"type": "attachment", "name": "Audio Message.caf", "mimeType": "application/octet-stream",
                        "data": base64.b64encode(b"caff" + b"\x00" * 60).decode()}})

    assert len(calls) == 1 and adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    _images, stt_paths, _audio_files, _videos = GatewayInboundMixin._classify_inbound_media(event, False)
    assert stt_paths == event.media_urls and len(stt_paths) == 1


@pytest.mark.asyncio
async def test_a_threaded_reply_stop_never_waits_on_an_earlier_voice_check(tmp_path, monkeypatch):
    """A ``/stop`` sent as an iMessage threaded reply is still an escape hatch: it is dispatched while
    an earlier voice note in the same group is at gate STT, not queued behind it."""
    import asyncio
    import base64
    import json
    import threading
    from plugins.platforms.photon.adapter import PhotonAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PHOTON_PROJECT_ID", "p")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "s")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={
        "require_mention": True, "mention_patterns": PATTERNS, "voice_mention_patterns": True}))
    seen = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: seen.append(e.message_id))
    release = threading.Event()

    def slow(path, _model=None, _source=None):
        release.wait(10)
        return {"success": True, "transcript": "hermes, what is on today", "provider": "fake"}

    def line(mid, content):
        return json.dumps({"messageId": mid, "space": {"id": "group-guid", "type": "group", "phone": None},
                           "sender": {"id": "+15551234567"}, "timestamp": "2026-05-14T19:06:32.000Z", "content": content})
    voice = {"type": "voice", "name": "v.ogg", "mimeType": "audio/ogg", "data": base64.b64encode(OGG).decode()}
    stop = {"type": "reply", "targetMessageId": "old", "content": {"type": "text", "text": "/stop hermes"}}
    try:
        with patch("tools.transcription_tools.transcribe_audio", side_effect=slow):
            await asyncio.wait_for(adapter._on_inbound_line(line("v1", voice)), 2)
            await asyncio.wait_for(adapter._on_inbound_line(line("s1", stop)), 2)
            assert seen == ["s1"]
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("other", [{"mimeType": "application/pdf"}, {"mimeType": "image/png"}])
@pytest.mark.parametrize("memo", [{"mimeType": "audio/ogg"},
                                  {"mimeType": "application/octet-stream", "uti": "com.apple.coreaudio-format.caf"}])
async def test_a_bluebubbles_memo_beside_a_file_sends_only_its_audio_to_stt_and_the_turn(tmp_path, monkeypatch, other, memo):
    """A group envelope with a voice memo (an audio MIME, or a CAF memo sent as octet-stream) and a PDF
    or photo: only the memo enters gate STT, and the admitted turn still transcribes it (the file
    beside it never demotes it to an audio attachment)."""
    import asyncio
    import json
    from gateway.run_inbound import GatewayInboundMixin
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://localhost:1234", "password": "secret", "require_mention": True,
        "mention_patterns": PATTERNS, "voice_mention_patterns": True, "send_read_receipts": False}))
    adapter.set_authorization_check(lambda *a, **k: True)  # the runner wires one at startup
    clip, doc = tmp_path / "memo.ogg", tmp_path / "file.bin"
    clip.write_bytes(OGG)
    doc.write_bytes(b"%PDF-1.4")
    adapter._download_attachment = AsyncMock(side_effect=lambda guid, _att: str(clip if guid == "a1" else doc))
    adapter.handle_message = AsyncMock()
    body = json.dumps({"type": "new-message", "data": {
        "guid": "m1", "text": "", "handle": {"address": "+15555550100"}, "isFromMe": False, "isGroup": True,
        "chats": [{"guid": "iMessage;+;group-chat"}],
        "attachments": [{"guid": "a1", **memo}, {"guid": "d1", **other}]}}).encode()
    request = SimpleNamespace(query={"password": "secret"}, headers={}, read=AsyncMock(return_value=body))
    transcribe, calls = _stt("hermes, look at this")
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        await adapter._handle_webhook(request)
        await asyncio.sleep(0)

    assert calls == [str(clip)]
    event = adapter.handle_message.await_args.args[0]
    _images, stt_paths, _audio_files, _videos = GatewayInboundMixin._classify_inbound_media(event, False)
    assert stt_paths == [str(clip)]
