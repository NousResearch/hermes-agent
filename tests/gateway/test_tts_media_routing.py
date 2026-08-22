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
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _MediaRoutingAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _event(thread_id=None, platform=Platform.TELEGRAM):
    source = SessionSource(
        platform=platform,
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
    )
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_qqbot_image_batch_forwards_reply_anchor(tmp_path):
    adapter = _MediaRoutingAdapter(Platform.QQBOT)
    image = tmp_path / "chart.png"
    image.write_bytes(b"image")
    adapter.send_image_file = AsyncMock(
        return_value=SendResult(success=True, message_id="image")
    )

    await adapter._send_multiple_images_with_reply_anchor(
        chat_id="chat-1",
        images=[(f"file://{image}", "chart")],
        reply_to="msg-1",
    )

    adapter.send_image_file.assert_awaited_once_with(
        chat_id="chat-1",
        image_path=str(image),
        caption="chart",
        reply_to="msg-1",
        metadata=None,
    )


@pytest.mark.asyncio
async def test_base_non_qq_image_batch_preserves_public_send_contract(tmp_path):
    adapter = _MediaRoutingAdapter(Platform.TELEGRAM)
    image = tmp_path / "chart.png"
    image.write_bytes(b"image")
    adapter.send_image_file = AsyncMock(
        return_value=SendResult(success=True, message_id="image")
    )

    await adapter.send_multiple_images(
        chat_id="chat-1",
        images=[(f"file://{image}", "chart")],
    )

    adapter.send_image_file.assert_awaited_once_with(
        chat_id="chat-1",
        image_path=str(image),
        caption="chart",
        metadata=None,
    )


@pytest.mark.asyncio
async def test_base_qqbot_media_delivery_forwards_reply_anchor(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter(Platform.QQBOT)
    event = _event(platform=Platform.QQBOT)
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter._message_handler = AsyncMock(
        return_value=f"[[audio_as_voice]]\nMEDIA:{media_file}"
    )
    adapter.send_voice = AsyncMock(
        return_value=SendResult(success=True, message_id="voice")
    )

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(media_file),
        reply_to="msg-1",
        metadata={"notify": True},
    )


@pytest.mark.asyncio
async def test_base_qqbot_auto_tts_forwards_reply_anchor(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter(Platform.QQBOT)
    event = _event(platform=Platform.QQBOT)
    event.message_type = MessageType.VOICE
    tts_path = tmp_path / "speech.ogg"

    adapter._message_handler = AsyncMock(return_value="Spoken reply")
    adapter._should_auto_tts_for_chat = lambda chat_id: True
    adapter.play_tts = AsyncMock(
        return_value=SendResult(success=True, message_id="voice")
    )
    monkeypatch.setattr(
        "gateway.platforms.base.build_auto_tts_output_path",
        lambda platform: str(tts_path),
    )
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)

    def fake_tts(**kwargs):
        tts_path.write_bytes(b"audio")
        return f'{{"success": true, "file_path": "{tts_path}"}}'

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.play_tts.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(tts_path),
        caption=None,
        reply_to="msg-1",
        metadata={"notify": True},
    )


def _fake_runner(thread_meta, reply_anchor=None):
    """Build a fake GatewayRunner-like object with the helper methods needed by
    _deliver_media_from_response."""
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: reply_anchor,
    )
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "directive", "method_name", "path_kwarg"),
    [
        ("speech.ogg", "[[audio_as_voice]]\n", "send_voice", "audio_path"),
        ("clip.mp4", "", "send_video", "video_path"),
        ("report.pdf", "", "send_document", "file_path"),
    ],
)
async def test_streaming_media_delivery_forwards_reply_anchor(
    tmp_path, monkeypatch, filename, directive, method_name, path_kwarg,
):
    event = _event(platform=Platform.QQBOT)
    media_file = _allowed_media_path(tmp_path, monkeypatch, filename)
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}, reply_anchor="msg-1"),
        f"{directive}MEDIA:{media_file}",
        event,
        adapter,
    )

    getattr(adapter, method_name).assert_awaited_once_with(
        chat_id="chat-1",
        **{path_kwarg: str(media_file)},
        reply_to="msg-1",
        metadata={},
    )


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
    event = _event(thread_id="topic-1", platform=Platform.QQBOT)
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
        _send_multiple_images_with_reply_anchor=AsyncMock(return_value=None),
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
        metadata={"thread_id": "topic-1"},
    )
    adapter._send_multiple_images_with_reply_anchor.assert_awaited_once_with(
        chat_id="chat-1",
        images=[(f"file://{media_file.as_posix()}", "")],
        reply_to="msg-1",
        metadata={"thread_id": "topic-1"},
    )


@pytest.mark.asyncio
async def test_queued_followup_delivery_reuses_routing_metadata_for_media(
    tmp_path, monkeypatch,
):
    """Queued text and media must stay on the same precomputed reply route."""
    event = _event(thread_id="source-topic", platform=Platform.QQBOT)
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
        _send_multiple_images_with_reply_anchor=AsyncMock(return_value=None),
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
        metadata=routing_metadata,
    )
    adapter._send_multiple_images_with_reply_anchor.assert_awaited_once_with(
        chat_id="chat-1",
        images=[(f"file://{media_file.as_posix()}", "")],
        reply_to="msg-1",
        metadata=routing_metadata,
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
        metadata={"thread_id": "topic-1"},
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
        metadata={"thread_id": "topic-1"},
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
        metadata={"thread_id": "topic-1"},
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
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.QQBOT)
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
        self.images.append({
            "chat_id": chat_id,
            "image_path": image_path,
            "metadata": metadata,
            "reply_to": reply_to,
        })
        return SendResult(success=True, message_id=f"img-{len(self.images)}")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _QueuedMediaAgent:
    calls = 0
    first_response = ""

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
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
        platform=Platform.QQBOT,
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
        event_message_id="initial-1",
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
    assert any(img["reply_to"] == "initial-1" for img in adapter.images)
    assert all(img["reply_to"] != "queued-1" for img in adapter.images)
