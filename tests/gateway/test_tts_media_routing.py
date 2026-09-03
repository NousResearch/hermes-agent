"""
Tests for cross-platform audio/voice media routing.

These tests pin the expected delivery path for audio media files across
Telegram (where Bot-API sendAudio only accepts MP3/M4A and .ogg/.opus
only renders as a voice bubble when explicitly flagged) and via
``GatewayRunner._deliver_media_from_response``.
"""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _MediaRoutingAdapter(BasePlatformAdapter):
    typing_send_requires_final_fence = True

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _event(thread_id=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="make speech",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


@pytest.mark.asyncio
async def test_base_adapter_routes_voice_tagged_telegram_ogg_media_tag_to_voice_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter._message_handler = AsyncMock(
        return_value=f"[[audio_as_voice]]\nMEDIA:{media_file}"
    )
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(media_file),
        metadata={"notify": True},
        is_voice=True,
    )
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconnect_replacement_owns_all_unsent_final_media(tmp_path, monkeypatch):
    stale = _MediaRoutingAdapter()
    replacement = _MediaRoutingAdapter()
    event = _event(thread_id="topic-1")
    image = _allowed_media_path(tmp_path, monkeypatch, "result.png")
    voice = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    video = _allowed_media_path(tmp_path, monkeypatch, "clip.mp4")
    document = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    stale._message_handler = AsyncMock(
        return_value=(
            "Final files\n[[audio_as_voice]]\n"
            f"MEDIA:{image}\nMEDIA:{voice}\nMEDIA:{video}\nMEDIA:{document}"
        )
    )

    runner = SimpleNamespace(_adapter_for_source=lambda _source: replacement)
    stale.gateway_runner = runner
    replacement.gateway_runner = runner
    for adapter in (stale, replacement):
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text"))
        adapter.send_multiple_images = AsyncMock(return_value=None)
        adapter.send_voice = AsyncMock(
            return_value=SendResult(success=True, message_id="voice")
        )
        adapter.send_video = AsyncMock(
            return_value=SendResult(success=True, message_id="video")
        )
        adapter.send_document = AsyncMock(
            return_value=SendResult(success=True, message_id="document")
        )

    await stale._process_message_background(event, build_session_key(event.source))

    stale.send.assert_not_awaited()
    stale.send_multiple_images.assert_not_awaited()
    stale.send_voice.assert_not_awaited()
    stale.send_video.assert_not_awaited()
    stale.send_document.assert_not_awaited()
    replacement.send.assert_awaited_once()
    replacement.send_multiple_images.assert_awaited_once()
    replacement.send_voice.assert_awaited_once()
    replacement.send_video.assert_awaited_once()
    replacement.send_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconnect_replacement_owns_unsent_automatic_tts(tmp_path):
    stale = _MediaRoutingAdapter()
    replacement = _MediaRoutingAdapter()
    event = _event(thread_id="topic-1")
    event.message_type = MessageType.VOICE
    stale._message_handler = AsyncMock(return_value="Voice final")
    stale._should_auto_tts_for_chat = lambda _chat_id: True

    runner = SimpleNamespace(_adapter_for_source=lambda _source: replacement)
    stale.gateway_runner = runner
    replacement.gateway_runner = runner
    stale.play_tts = AsyncMock(
        return_value=SendResult(success=True, message_id="stale-voice")
    )
    replacement.play_tts = AsyncMock(
        return_value=SendResult(success=True, message_id="replacement-voice")
    )
    stale.send = AsyncMock(return_value=SendResult(success=True, message_id="stale-text"))
    replacement.send = AsyncMock(
        return_value=SendResult(success=True, message_id="replacement-text")
    )
    tts_path = tmp_path / "reply.ogg"
    tts_path.write_bytes(b"audio")

    with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
        "tools.tts_tool.text_to_speech_tool",
        return_value=f'{{"file_path": "{tts_path}"}}',
    ):
        await stale._process_message_background(event, build_session_key(event.source))

    stale.play_tts.assert_not_awaited()
    stale.send.assert_not_awaited()
    replacement.play_tts.assert_awaited_once()
    # The short reply is the successful Telegram voice caption, so no duplicate
    # text send should be necessary on either transport.
    replacement.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_media_re_resolves_between_payloads(tmp_path, monkeypatch):
    first = _MediaRoutingAdapter()
    replacement = _MediaRoutingAdapter()
    event = _event(thread_id="topic-1")
    voice = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    document = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    current = {"adapter": first}
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _source: current["adapter"]

    async def send_voice_then_replace(**_kwargs):
        current["adapter"] = replacement
        return SendResult(success=True, message_id="voice")

    first.send_voice = AsyncMock(side_effect=send_voice_then_replace)
    first.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="stale-document")
    )
    replacement.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="replacement-document")
    )

    await runner._deliver_media_from_response(
        f"[[audio_as_voice]]\nMEDIA:{voice}\nMEDIA:{document}",
        event,
        first,
        thread_metadata={"thread_id": "topic-1"},
    )

    first.send_voice.assert_awaited_once()
    first.send_document.assert_not_awaited()
    replacement.send_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_queued_delivery_re_resolves_media_after_text(tmp_path, monkeypatch):
    first = _MediaRoutingAdapter()
    replacement = _MediaRoutingAdapter()
    event = _event(thread_id="topic-1")
    document = _allowed_media_path(tmp_path, monkeypatch, "queued.pdf")
    current = {"adapter": first}
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _source: current["adapter"]

    async def send_text_then_replace(*_args, **_kwargs):
        current["adapter"] = replacement
        return SendResult(success=True, message_id="text")

    first.send = AsyncMock(side_effect=send_text_then_replace)
    first.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="stale-document")
    )
    replacement.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="replacement-document")
    )

    await runner._deliver_queued_first_response(
        f"Queued final\nMEDIA:{document}",
        source=event.source,
        adapter=first,
        metadata={"thread_id": "topic-1"},
        event_message_id=event.message_id,
    )

    first.send.assert_awaited_once()
    first.send_document.assert_not_awaited()
    replacement.send_document.assert_awaited_once()


def _fake_runner(thread_meta):
    """Build a fake GatewayRunner-like object with the helper methods needed by
    _deliver_media_from_response."""
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    return runner


@pytest.mark.asyncio
async def test_streaming_delivery_blocks_media_path_outside_allowed_roots(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    allowed_root = tmp_path / "media-cache"
    allowed_root.mkdir()
    secret = tmp_path / "outside.pdf"
    secret.write_bytes(b"%PDF secret")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (allowed_root,),
    )
    # This test exercises the strict-allowlist path; force strict mode on
    # and disable recency trust so the freshly-written tmp_path file is not
    # auto-accepted by the trust window. (Recency trust is covered separately
    # in test_platform_base.py. The public default flipped to non-strict in
    # 2026-05; this test pins strict on explicitly.)
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{secret}",
        event,
        adapter,
    )

    adapter.send_document.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()


class _DiscordMediaFailureAdapter(BasePlatformAdapter):
    """Minimal adapter to exercise non-streaming MEDIA failure notification."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.DISCORD)
        self.notices: list[str] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        self.notices.append(content or "")
        return SendResult(success=True, message_id="notice")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.mark.asyncio
async def test_non_streaming_media_failure_notifies_user(tmp_path, monkeypatch):
    """Attachmentless send_video results must surface a user-visible notice (#66797)."""
    adapter = _DiscordMediaFailureAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "clip.mp4")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_video = AsyncMock(
        return_value=SendResult(
            success=False,
            error="Discord accepted the message but attached no files (clip.mp4)",
        )
    )
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_multiple_images = AsyncMock()

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_video.assert_awaited_once()
    assert adapter.notices == ["⚠️ Couldn't deliver the video attachment."]


class _DiscordMediaFailureAdapter(BasePlatformAdapter):
    """Minimal adapter to exercise non-streaming MEDIA failure notification."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.DISCORD)
        self.notices: list[str] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        self.notices.append(content or "")
        return SendResult(success=True, message_id="notice")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.mark.asyncio
async def test_queued_followup_delivery_strips_media_tag_from_text_and_sends_image(
    tmp_path, monkeypatch,
):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "pricelist.png")
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_source = lambda source, anchor=None: {"thread_id": "topic-1"}
    runner._reply_anchor_for_event = lambda event: event.message_id

    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_queued_first_response(
        runner,
        f"Quote here\nMEDIA:{media_file}",
        source=event.source,
        adapter=adapter,
        metadata={"thread_id": "topic-1"},
        event_message_id=event.message_id,
    )

    adapter.send.assert_awaited_once_with(
        "chat-1",
        "Quote here",
        metadata={"thread_id": "topic-1", "notify": True},
    )
    adapter.send_multiple_images.assert_awaited_once_with(
        chat_id="chat-1",
        images=[(f"file://{media_file.as_posix()}", "")],
        metadata={"thread_id": "topic-1", "notify": True},
    )


@pytest.mark.asyncio
async def test_queued_delivery_fences_lease_before_text_and_media(
    tmp_path, monkeypatch,
):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "fenced.png")
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        name="test",
        _fence_typing_lease_before_final=AsyncMock(),
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )
    metadata = {"_hermes_typing_lease_id": "lease-1"}

    await GatewayRunner._deliver_queued_first_response(
        runner,
        f"Final text\nMEDIA:{media_file}",
        source=event.source,
        adapter=adapter,
        metadata=metadata,
        event_message_id=event.message_id,
    )

    adapter._fence_typing_lease_before_final.assert_awaited_once_with(
        "chat-1", "lease-1"
    )


@pytest.mark.asyncio
async def test_queued_reconcile_supports_legacy_edit_signature_without_metadata():
    event = _event(thread_id="topic-1")
    runner = object.__new__(GatewayRunner)
    edit_calls = []

    async def legacy_edit_message(
        chat_id, message_id, content, *, finalize=False
    ):
        edit_calls.append((chat_id, message_id, content, finalize))
        return SendResult(success=True, message_id=message_id)

    adapter = SimpleNamespace(
        name="legacy",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        edit_message=legacy_edit_message,
        send=AsyncMock(return_value=SendResult(success=True, message_id="duplicate")),
    )
    stream_consumer = SimpleNamespace(
        message_id="preview-1",
        _turn_split_delivery=False,
    )

    await GatewayRunner._deliver_queued_first_response(
        runner,
        "Final text",
        source=event.source,
        adapter=adapter,
        metadata={"thread_id": "topic-1"},
        event_message_id=event.message_id,
        deliver_media=False,
        stream_consumer=stream_consumer,
    )

    assert edit_calls == [
        ("chat-1", "preview-1", "Final text", True)
    ]
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_followup_gets_successor_lease_for_typing():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="test-token")
    )
    adapter._bot = AsyncMock()
    session_key = "telegram:chat-1"
    old_lease = adapter._begin_typing_lease("chat-1", session_key)
    old_metadata = adapter._typing_metadata_for_session(session_key, {})
    await adapter._fence_typing_lease_before_final("chat-1", old_lease)

    successor_metadata = GatewayRunner._renew_followup_typing_lease(
        adapter,
        chat_id="chat-1",
        session_key=session_key,
        metadata={
            "thread_id": "123",
            "telegram_dm_topic_reply_fallback": True,
        },
    )

    await adapter.send_typing("chat-1", metadata=old_metadata)
    adapter._bot.send_chat_action.assert_not_awaited()

    await adapter.send_typing("chat-1", metadata=successor_metadata)
    adapter._bot.send_chat_action.assert_awaited_once()
    assert successor_metadata["thread_id"] == "123"
    assert successor_metadata["_hermes_typing_lease_id"] != old_lease


@pytest.mark.asyncio
async def test_queued_followup_delivery_reuses_routing_metadata_for_media(
    tmp_path, monkeypatch,
):
    """Queued text and media must stay on the same precomputed reply route."""
    event = _event(thread_id="source-topic")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "threaded.png")
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_source = (
        lambda source, reply_to_message_id=None: {"thread_id": "recomputed-topic"}
    )
    runner._reply_anchor_for_event = lambda event: event.message_id
    routing_metadata = {
        "thread_id": "queued-topic",
        "reply_to_message_id": "trigger-message",
    }

    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_queued_first_response(
        runner,
        f"Threaded image\nMEDIA:{media_file}",
        source=event.source,
        adapter=adapter,
        metadata=routing_metadata,
        event_message_id=event.message_id,
    )

    adapter.send.assert_awaited_once_with(
        "chat-1",
        "Threaded image",
        metadata={**routing_metadata, "notify": True},
    )
    adapter.send_multiple_images.assert_awaited_once_with(
        chat_id="chat-1",
        images=[(f"file://{media_file.as_posix()}", "")],
        metadata={**routing_metadata, "notify": True},
    )


@pytest.mark.asyncio
async def test_queued_followup_delivery_keeps_remote_image_url_in_text():
    event = _event(thread_id="topic-1")
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_source = lambda source, anchor=None: {"thread_id": "topic-1"}
    runner._reply_anchor_for_event = lambda event: event.message_id

    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    response = "See this mockup\nhttps://example.com/mockup.png"
    await GatewayRunner._deliver_queued_first_response(
        runner,
        response,
        source=event.source,
        adapter=adapter,
        metadata={"thread_id": "topic-1"},
        event_message_id=event.message_id,
    )

    adapter.send.assert_awaited_once_with(
        "chat-1",
        response,
        metadata={"thread_id": "topic-1", "notify": True},
    )
    adapter.send_multiple_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_followup_delivery_keeps_bare_local_path_in_text(
    tmp_path, monkeypatch,
):
    """Queued delivery must not strip paths its explicit-only uploader ignores."""
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "inspected.png")
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_source = (
        lambda source, reply_to_message_id=None: {"thread_id": "topic-1"}
    )
    runner._reply_anchor_for_event = lambda event: event.message_id

    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    response = f"The inspected file is at {media_file}."
    await GatewayRunner._deliver_queued_first_response(
        runner,
        response,
        source=event.source,
        adapter=adapter,
        metadata={"thread_id": "topic-1"},
        event_message_id=event.message_id,
    )

    adapter.send.assert_awaited_once_with(
        "chat-1",
        response,
        metadata={"thread_id": "topic-1", "notify": True},
    )
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_followup_delivery_preserves_protected_media_example():
    """Inline-code MEDIA examples must remain visible after queued text cleanup."""
    event = _event(thread_id="topic-1")
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_source = lambda source, anchor=None: {"thread_id": "topic-1"}
    runner._reply_anchor_for_event = lambda event: event.message_id

    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    response = "Tag files like `MEDIA:/tmp/example.png` in tool output."
    await GatewayRunner._deliver_queued_first_response(
        runner,
        response,
        source=event.source,
        adapter=adapter,
        metadata={"thread_id": "topic-1"},
        event_message_id=event.message_id,
    )

    adapter.send.assert_awaited_once_with(
        "chat-1",
        response,
        metadata={"thread_id": "topic-1", "notify": True},
    )
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_followup_delivery_skips_media_when_turn_failed():
    """A failed first turn delivers its (failure) text but never uploads
    attachments as if the turn succeeded — deliver_media=False mirrors the
    completed-turn path's ``not agent_result.get("failed")`` guard."""
    event = _event(thread_id="topic-1")
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_source = lambda source, anchor=None: {"thread_id": "topic-1"}
    runner._reply_anchor_for_event = lambda event: event.message_id

    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_queued_first_response(
        runner,
        "The request failed: provider exploded\nMEDIA:/tmp/pricelist.png",
        source=event.source,
        adapter=adapter,
        metadata={"thread_id": "topic-1"},
        event_message_id=event.message_id,
        deliver_media=False,
    )

    adapter.send.assert_awaited_once()
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_document.assert_not_awaited()
    adapter.send_video.assert_not_awaited()


class _QueuedMediaCaptureAdapter(BasePlatformAdapter):
    """Adapter that records text + native image delivery for queued-resend tests."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent = []
        self.images = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id=f"text-{len(self.sent)}")

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, metadata=None, **kwargs):
        self.images.append({"chat_id": chat_id, "image_path": image_path, "metadata": metadata})
        return SendResult(success=True, message_id=f"img-{len(self.images)}")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0.0):
        for image_url, _alt in images:
            path = image_url
            if path.startswith("file://"):
                path = path[len("file://"):]
            self.images.append({"chat_id": chat_id, "image_path": path, "metadata": metadata})

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _QueuedMediaAgent:
    calls = 0
    first_response = ""

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).calls += 1
        if type(self).calls == 1:
            return {
                "final_response": type(self).first_response,
                "messages": [],
                "api_calls": 1,
            }
        return {
            "final_response": "follow-up processed",
            "messages": [],
            "api_calls": 1,
        }


@pytest.mark.asyncio
async def test_queued_resend_branch_delivers_media_and_preserves_protected_example(
    tmp_path, monkeypatch,
):
    """Exercise the real queued first-response resend path in ``_run_agent``."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "quote.png")
    protected = "Tag files like `MEDIA:/tmp/example.png` in tool output."
    _QueuedMediaAgent.calls = 0
    _QueuedMediaAgent.first_response = f"Quote here\nMEDIA:{media_file}\n{protected}"

    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", _QueuedMediaAgent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = _QueuedMediaCaptureAdapter()
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        thread_id="topic-1",
    )
    session_key = build_session_key(source)
    adapter._pending_messages[session_key] = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-queued-media",
        session_key=session_key,
    )

    assert _QueuedMediaAgent.calls == 2
    assert result["final_response"] == "follow-up processed"
    first_texts = [call["content"] for call in adapter.sent if "Quote here" in call["content"]]
    assert first_texts, f"expected queued resend of first response, got: {adapter.sent!r}"
    assert f"MEDIA:{media_file}" not in first_texts[0]
    assert "`MEDIA:/tmp/example.png`" in first_texts[0]
    assert any(str(media_file) in img["image_path"] for img in adapter.images), (
        f"expected native image delivery via queued resend, got: {adapter.images!r}"
    )
