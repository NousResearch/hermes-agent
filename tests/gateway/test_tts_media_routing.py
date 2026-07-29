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
import sys

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
async def test_post_stream_bare_local_image_is_not_delivered(tmp_path, monkeypatch):
    """Bare local image path in response text MUST NOT be auto-delivered.
    Post-stream delivery is explicit-only (#20834): bare paths that were
    already in the streamed text are not promoted to attachments."""
    event = _event()
    _allowed_media_path(tmp_path, monkeypatch, "bare.png")
    response = "Check your files at /tmp/bare.png"
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
    adapter.send_voice.assert_not_awaited()
    adapter.send_video.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_bare_local_pdf_is_not_delivered(tmp_path, monkeypatch):
    """Bare local non-image path MUST NOT be auto-delivered as document.
    Post-stream delivery is explicit-only (#20834): bare paths that were
    already in the streamed text are not promoted to attachments."""
    event = _event()
    _allowed_media_path(tmp_path, monkeypatch, "chart.pdf")
    response = "Generated chart at /tmp/chart.pdf"
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

    adapter.send_document.assert_not_awaited()
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()
    adapter.send_video.assert_not_awaited()


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
@pytest.mark.parametrize("filename,alt", [
    ("sub folder/shot.png", "alt"),
    ("100%25done.png", ""),
    ("normal.png", ""),
], ids=["space_path", "literal_percent", "normal_path"])
async def test_send_multiple_images_roundtrip(tmp_path, filename, alt):
    """file:// URI round-trip: send_multiple_images decodes URI to local path
    and passes it to send_image_file. Parametrized over space/literal%/normal."""
    import urllib.parse

    adapter = _RoundtripAdapter()
    validated = str((tmp_path / filename).resolve())
    norm_url = "file://" + urllib.parse.quote(validated, safe="/:\\\\")

    await adapter.send_multiple_images(
        chat_id="chat-1",
        images=[(norm_url, alt)],
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


# ---------------------------------------------------------------------------
# Dev tests: Windows paths, unicode paths, paths with spaces (PR #43332)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")
@pytest.mark.asyncio
async def test_post_stream_file_url_windows_path_delivered(tmp_path, monkeypatch):
    """``file:///C:/...`` style Windows URI (three slashes + drive letter)
    in a markdown image tag reaches ``send_multiple_images``.
    """
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "win_test.png")
    # Simulate Windows absolute path URI: file:///C:/path/to/file.png
    # Convert the resolved PosixPath to a "Windows-style" path string
    # by using as_posix() (forward slashes) which _normalize_file_url
    # handles via ``file:///C:/...`` syntax.
    win_uri = f"file:///{img.as_posix()}"
    response = f"See: ![shot]({win_uri})"
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
async def test_post_stream_file_url_unicode_path(tmp_path, monkeypatch):
    """file:// URI with unicode characters in path reaches send_multiple_images."""
    event = _event()
    # Create a file with unicode name
    img = _allowed_media_path(tmp_path, monkeypatch, "照片.png")
    response = f"See: ![photo](file://{img.as_posix()})"
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
    # The unicode filename must be present in the URI (percent-encoded)
    from urllib.parse import quote as _urlquote
    expected_part = _urlquote("照片.png")
    assert expected_part in file_uri, (
        f"Expected {expected_part} in {file_uri}"
    )


@pytest.mark.asyncio
async def test_post_stream_file_url_space_path(tmp_path, monkeypatch):
    """file:// URI with spaces in directory/file name reaches send_multiple_images."""
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "screen shot.png")
    response = f"See: ![img](file://{img.as_posix()})"
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
    # Space in filename must be percent-encoded
    assert "%20" in file_uri, (
        f"Expected percent-encoded space (%20) in {file_uri}"
    )


@pytest.mark.asyncio
async def test_post_stream_dedup_substring_safe(tmp_path, monkeypatch):
    """Different files (foo.png vs foo.png.backup.png) both deliver — no
    substring comparison bug."""
    event = _event()
    img1 = _allowed_media_path(tmp_path, monkeypatch, "foo.png")
    img2 = _allowed_media_path(tmp_path, monkeypatch, "foo.png.backup.png",)
    img2.write_bytes(b"backup")
    response = f"![a](file://{img1.as_posix()}) ![b](file://{img2.as_posix()})"
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
    args, kwargs = adapter.send_multiple_images.call_args
    images = kwargs.get("images")
    assert images is not None, "images kwarg missing"
    assert len(images) == 2, f"expected 2 images, got {len(images)}: {images}"
    uris = [uri for uri, _ in images]
    assert all(u.startswith("file://") for u in uris)


@pytest.mark.asyncio
async def test_post_stream_dedup_file_url_media_same_file(tmp_path, monkeypatch):
    """MEDIA local path + file:// URI for the same file deliver once."""
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "dedup.png")
    response = f"MEDIA:{img} ![shot](file://{img.as_posix()})"
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
    args, kwargs = adapter.send_multiple_images.call_args
    images = kwargs.get("images")
    assert images is not None, "images kwarg missing"
    # MEDIA image + extracted image = same file -> 1 total
    assert len(images) == 1, f"expected 1 image (dedup), got {len(images)}: {images}"


@pytest.mark.asyncio
async def test_post_stream_dedup_http_exact_url(tmp_path, monkeypatch):
    """Same HTTP URL twice delivers once via exact-string dedup."""
    event = _event()
    response = "![a](https://example.com/pic.png) ![b](https://example.com/pic.png)"
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
    args, kwargs = adapter.send_multiple_images.call_args
    images = kwargs.get("images")
    assert images is not None, "images kwarg missing"
    assert len(images) == 1, f"expected 1 image (dedup), got {len(images)}: {images}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path semantics")
@pytest.mark.asyncio
async def test_post_stream_history_dedup_case_variation(tmp_path, monkeypatch):
    """History contains C:\\...\\Image.PNG; current file URI pointing to the
    same file (case/slash variation) is deduplicated."""
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "image.png")
    # Simulate history containing a case-different variant of the SAME file
    alt_case = str(img).replace("image.png", "Image.PNG")
    # Response uses an alternative file:// encoding of the same file
    response = f"![shot](file://{img.as_posix()})"
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
        history_media_paths={alt_case},
    )

    # The file:// URI resolves to the same canonical path as history -> skip
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_history_dedup_same_file_via_media(tmp_path, monkeypatch):
    """History contains original MEDIA path; current percent-encoded file URI
    pointing to the same file is deduplicated."""
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "dedup_hist.png")
    # Response has BOTH a MEDIA: (which goes through extract_media) and an
    # explicit file:// tag; history contains the same path.
    response = f"MEDIA:{img} ![shot](file://{img.as_posix()})"
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
        history_media_paths={str(img)},
    )

    # The MEDIA: path matches history -> filtered out.
    # The file:// path also matches history -> filtered out.
    # Result: nothing to deliver.
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_history_dedup_different_paths_both_deliver(tmp_path, monkeypatch):
    """Two different files (one in history, one new) — both deliver."""
    event = _event()
    img1 = _allowed_media_path(tmp_path, monkeypatch, "already_sent.png")
    img2 = _allowed_media_path(tmp_path, monkeypatch, "new_file.png",)
    response = f"![old](file://{img1.as_posix()}) ![new](file://{img2.as_posix()})"
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
        history_media_paths={str(img1)},
    )

    # img1 matches history -> skipped; img2 is new -> delivered
    adapter.send_multiple_images.assert_awaited_once()
    args, kwargs = adapter.send_multiple_images.call_args
    images = kwargs.get("images")
    assert images is not None, "images kwarg missing"
    assert len(images) == 1, f"expected 1 new image, got {len(images)}: {images}"
    # Only the new file's URI is present
    assert str(img2) in images[0][0], f"Expected {img2} in {images[0][0]}"
    assert str(img1) not in images[0][0], f"Old file {img1} should not appear"
