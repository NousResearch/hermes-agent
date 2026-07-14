"""
Tests for cross-platform audio/voice media routing.

These tests pin the expected delivery path for audio media files across
Telegram (where Bot-API sendAudio only accepts MP3/M4A and .ogg/.opus
only renders as a voice bubble when explicitly flagged) and via
``GatewayRunner._deliver_media_from_response``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _MediaRoutingAdapter(BasePlatformAdapter):
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
async def test_base_adapter_routes_telegram_flac_media_tag_to_document_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.flac")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"notify": True},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_routes_non_voice_telegram_ogg_media_tag_to_document_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"notify": True},
    )
    adapter.send_voice.assert_not_awaited()


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


def _fake_runner(thread_meta):
    """Build a fake GatewayRunner-like object with the helper methods needed by
    _deliver_media_from_response."""
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    return runner


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_flac_media_tag_to_document_sender(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.flac")
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
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_non_voice_telegram_ogg_media_tag_to_document_sender(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
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
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_mp3_media_tag_to_voice_sender(tmp_path, monkeypatch):
    """MP3 audio on Telegram must go through send_voice (which routes to
    sendAudio internally); Telegram accepts MP3 for the audio player."""
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.mp3")
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
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_document.assert_not_awaited()


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

# ---------------------------------------------------------------------------
# Post-stream regression tests for explicit file:// image tags (PR #43332)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_stream_file_url_image_goes_to_send_multiple_images(tmp_path, monkeypatch):
    """file:// URI in explicit markdown image tag reaches send_multiple_images."""
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "screenshot.png")
    response = f"See: ![shot](file://{img.as_posix()})"
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="batch")),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    kwargs = adapter.send_multiple_images.call_args.kwargs
    images = kwargs.get("images")
    assert images is not None, "images kwarg missing"
    assert len(images) == 1
    file_uri = images[0][0]
    assert file_uri.startswith("file://")
    assert str(img) in file_uri


@pytest.mark.asyncio
async def test_post_stream_media_tag_still_delivers(tmp_path, monkeypatch):
    """MEDIA:/path/a.png in post-stream still reaches send_multiple_images."""
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "media.png")
    response = f"MEDIA:{img}"
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="batch")),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    kwargs = adapter.send_multiple_images.call_args.kwargs
    images = kwargs.get("images")
    assert images is not None, "images kwarg missing"
    assert len(images) == 1


@pytest.mark.asyncio
async def test_post_stream_bare_local_image_still_delivered(tmp_path, monkeypatch):
    """Bare local image path in response text is still delivered (deliverable mode)."""
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "bare.png")
    response = f"Here is your file: {img}"
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="batch")),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter,
    )

    from urllib.parse import unquote as _unquote
    adapter.send_multiple_images.assert_awaited_once()
    sent = adapter.send_multiple_images.call_args.kwargs["images"]
    assert len(sent) == 1
    sent_url = sent[0][0]
    assert sent_url.startswith("file://")
    decoded = _unquote(sent_url[len("file://"):])
    parts_img = str(img).replace("\\", "/")
    parts_decoded = decoded.replace("\\", "/")
    assert parts_img.casefold() in parts_decoded.casefold()


@pytest.mark.asyncio
async def test_post_stream_bare_local_pdf_still_delivered_as_document(tmp_path, monkeypatch):
    """Bare local non-image path goes to send_document, not send_multiple_images."""
    event = _event()
    pdf = _allowed_media_path(tmp_path, monkeypatch, "chart.pdf")
    response = f"Generated chart: {pdf}"
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="batch")),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once()
    adapter.send_multiple_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_json_embedded_file_url_not_attached(tmp_path, monkeypatch):
    """file:// image tag inside a JSON string value must NOT reach send_multiple_images."""
    event = _event()
    png = _allowed_media_path(tmp_path, monkeypatch, "stale.png")
    response = '{"result":"![img](file://%s)"}' % png.as_posix()
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="batch")),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter,
    )

    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_html_data_src_not_delivered(tmp_path, monkeypatch):
    """HTML img with data-src pointing to a real file must NOT trigger upload."""
    event = _event()
    png = _allowed_media_path(tmp_path, monkeypatch, "hidden.png")
    response = f'<img data-src="file://{png.as_posix()}">'
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="batch")),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter,
    )

    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


# ---------------------------------------------------------------------------
# send_multiple_images round-trip: verifies file:// path integrity
# ---------------------------------------------------------------------------


class _RoundtripAdapter(BasePlatformAdapter):
    """Minimal adapter that calls the base send_multiple_images."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.send_image_file = AsyncMock(return_value=SendResult(success=True, message_id="img"))

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.mark.asyncio
async def test_send_multiple_images_roundtrip_space_path(tmp_path):
    """file:// URI with %20-encoded space => send_image_file receives decoded path."""
    import urllib.parse

    adapter = _RoundtripAdapter()
    validated = str((tmp_path / "sub folder" / "shot.png").resolve())
    norm_url = "file://" + urllib.parse.quote(validated, safe="/:\\")

    await adapter.send_multiple_images(
        chat_id="chat-1",
        images=[(norm_url, "alt")],
    )

    adapter.send_image_file.assert_awaited_once()
    _, kwargs = adapter.send_image_file.call_args
    received_path = kwargs.get("image_path")
    assert received_path == validated, (
        f"send_image_file received {received_path!r}, expected {validated!r}"
    )


@pytest.mark.asyncio
async def test_send_multiple_images_roundtrip_literal_percent(tmp_path):
    """Filename containing literal % => send_image_file receives exact path."""
    import urllib.parse

    adapter = _RoundtripAdapter()
    validated = str((tmp_path / "100%25done.png").resolve())
    norm_url = "file://" + urllib.parse.quote(validated, safe="/:\\")

    await adapter.send_multiple_images(
        chat_id="chat-1",
        images=[(norm_url, "")],
    )

    adapter.send_image_file.assert_awaited_once()
    _, kwargs = adapter.send_image_file.call_args
    received_path = kwargs.get("image_path")
    assert received_path == validated, (
        f"send_image_file received {received_path!r}, expected {validated!r}"
    )


@pytest.mark.asyncio
async def test_send_multiple_images_roundtrip_normal_path(tmp_path):
    """Simple path without encoding -- no change."""
    import urllib.parse

    adapter = _RoundtripAdapter()
    validated = str((tmp_path / "normal.png").resolve())
    norm_url = "file://" + urllib.parse.quote(validated, safe="/:\\")

    await adapter.send_multiple_images(
        chat_id="chat-1",
        images=[(norm_url, "")],
    )

    adapter.send_image_file.assert_awaited_once()
    _, kwargs = adapter.send_image_file.call_args
    received_path = kwargs.get("image_path")
    assert received_path == validated, (
        f"send_image_file received {received_path!r}, expected {validated!r}"
    )


@pytest.mark.asyncio
async def test_post_stream_html_src_bare_path_not_delivered(tmp_path, monkeypatch):
    """HTML img with bare Windows/POSIX path must NOT trigger upload."""
    event = _event()
    _allowed_media_path(tmp_path, monkeypatch, "barepath.png")
    for response in [
        '<img src="C:\\\\Users\\\\test\\\\file.png">',
        '<img src="/tmp/file.png">',
        '<img src="smb://server/share/file.png">',
    ]:
        adapter = SimpleNamespace(
            name="test",
            extract_media=BasePlatformAdapter.extract_media,
            extract_images=BasePlatformAdapter.extract_images,
            extract_local_files=BasePlatformAdapter.extract_local_files,
            send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="batch")),
            send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
            send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
            send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
            send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        )

        await GatewayRunner._deliver_media_from_response(
            _fake_runner(None),
            response,
            event,
            adapter,
        )

        adapter.send_multiple_images.assert_not_awaited()
        adapter.send_image_file.assert_not_awaited()
        adapter.send_document.assert_not_awaited()
