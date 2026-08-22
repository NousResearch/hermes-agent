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
import os
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
@pytest.mark.parametrize("filename,alt,prefix", [
    ("sub folder/shot.png", "alt", "file://"),
    ("100%25done.png", "", "file://"),
    ("normal.png", "", "file://"),
    ("normal.png", "", "FILE://"),
], ids=["space_path", "literal_percent", "normal_path", "uppercase_scheme"])
async def test_send_multiple_images_roundtrip(tmp_path, filename, alt, prefix):
    """file:// URI round-trip: send_multiple_images decodes URI to local path
    and passes it to send_image_file. Parametrized over space/literal%/normal/uppercase."""
    import urllib.parse

    adapter = _RoundtripAdapter()
    validated = str((tmp_path / filename).resolve())
    norm_url = prefix + urllib.parse.quote(validated, safe="/:\\\\")

    await adapter.send_multiple_images(
        chat_id="chat-1",
        images=[(norm_url, alt)],
    )

    adapter.send_image_file.assert_awaited_once()
    _, kwargs = adapter.send_image_file.call_args
    received = kwargs.get("image_path")
    assert received is not None
    # _normalize_file_url returns forward-slash paths; compare with
    # normcase to handle backslash/forward-slash differences.
    assert os.path.normcase(received) == os.path.normcase(validated), (
        f"send_image_file received {received!r}, expected {validated!r}"
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


def test_normalize_file_url_drive_relative_rejected():
    """file://C%3Arelative.png has authority C%3A and MUST be rejected."""
    assert BasePlatformAdapter._normalize_file_url("file://C%3Arelative.png") is None


def test_normalize_file_url_encoded_three_slash_drive():
    """The encoded three-slash variant (no authority, drive letter encoded
    in the path) must normalize identically to the plain form: the second
    drive-letter re-check added after percent-decoding handles it."""
    assert BasePlatformAdapter._normalize_file_url("file:///C%3A/dir/a.png") == "C:/dir/a.png"


@pytest.mark.parametrize("raw_path", [
    r"C:\Users\KK\image.png",
    r"C:\Users\KK\AppData\Local\Temp\shot.png",
    r"C:\path with spaces\image.png",
])
def test_normalize_file_url_accepts_production_default_quote_windows_uri(raw_path):
    """P1 regression: gateway delivery producers wrap local paths with the
    DEFAULT ``urllib.parse.quote()`` (safe='/'), which percent-encodes the
    Windows drive letter and backslashes into the authority segment:

        file://C%3A%5CUsers%5CKK%5Cimage.png

    Before the fix this form was rejected as a bogus UNC authority, so the
    image silently fell back to ``send_image`` with the raw URI.  It must
    normalize to the Windows local path instead.
    """
    import urllib.parse

    uri = "file://" + urllib.parse.quote(raw_path)
    assert uri.startswith("file://C%3A"), uri  # sanity: encoding actually happened

    normed = BasePlatformAdapter._normalize_file_url(uri)
    assert normed is not None, f"production quote form rejected: {uri!r}"
    expected = raw_path.replace("\\", "/")
    assert os.path.normcase(normed) == os.path.normcase(expected), (
        f"normalize {normed!r} != expected {expected!r} for {uri!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_path", [
    r"C:\Users\KK\image.png",
    r"C:\Users\KK\AppData\Local\Temp\shot.png",
])
async def test_send_multiple_images_default_quote_windows_uri_goes_to_send_image_file(raw_path):
    """P1 regression: a local image wrapped by the production default
    ``quote()`` must be decoded and delivered via ``send_image_file``, and
    MUST NOT fall back to ``send_image`` with the raw ``file://`` URI.
    """
    import urllib.parse

    adapter = _RoundtripAdapter()
    adapter.send_image = AsyncMock(return_value=SendResult(success=True, message_id="remote"))
    uri = "file://" + urllib.parse.quote(raw_path)

    await adapter.send_multiple_images(
        chat_id="chat-1",
        images=[(uri, "alt")],
    )

    adapter.send_image_file.assert_awaited_once()
    _, kwargs = adapter.send_image_file.call_args
    received_path = kwargs.get("image_path")
    assert received_path is not None, f"image_path kwarg missing in {kwargs}"
    expected = raw_path.replace("\\", "/")
    assert os.path.normcase(received_path) == os.path.normcase(expected), (
        f"send_image_file received {received_path!r}, expected {expected!r}"
    )
    # The whole point of the fix: the URI must never be treated as a remote URL
    adapter.send_image.assert_not_awaited()
    assert not any(
        str(call).startswith("file://") for call in adapter.send_image.call_args_list
    )


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


# ---------------------------------------------------------------------------
# New contract (#73771): explicit MEDIA:/file:// in a LATER turn is a deliberate
# resend and must NOT be blocked by history-based dedup. The same file may be
# delivered again when the model explicitly attaches it in a subsequent
# response. Only same-response canonical dedup applies (see
# test_post_stream_dedup_file_url_media_same_file).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_stream_explicit_resend_allowed_in_later_turn(tmp_path, monkeypatch):
    """A later turn explicitly repeating the same MEDIA: tag still delivers.

    Regression guard for #73771: cross-turn history dedup was removed. The
    post-stream rescan is explicit-only, and a MEDIA: directive in the final
    streamed reply is the model deliberately attaching a file — including a
    user-requested resend. Calling _deliver_media_from_response twice with the
    same response must deliver both times (no hidden history state).
    """
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "resend.png")
    response = f"MEDIA:{img} ![shot](file://{img.as_posix()})"

    def _make_adapter():
        return SimpleNamespace(
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

    # Turn 1 delivers the explicit media.
    adapter1 = _make_adapter()
    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter1,
    )
    adapter1.send_multiple_images.assert_awaited_once()

    # Turn 2 (a later turn, same explicit directive) must deliver AGAIN —
    # no cross-turn history dedup may suppress it.
    adapter2 = _make_adapter()
    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter2,
    )
    adapter2.send_multiple_images.assert_awaited_once()
    args, kwargs = adapter2.send_multiple_images.call_args
    images = kwargs.get("images")
    assert images is not None, "images kwarg missing"
    assert len(images) == 1, f"expected 1 image in resend turn, got {len(images)}: {images}"


@pytest.mark.asyncio
async def test_post_stream_file_url_resend_allowed_in_later_turn(tmp_path, monkeypatch):
    """A later turn explicitly repeating the same file:// image still delivers.

    Mirrors test_post_stream_explicit_resend_allowed_in_later_turn for the
    file:// markdown image path — the explicit file:// contract is also exempt
    from cross-turn dedup (#73771).
    """
    event = _event()
    img = _allowed_media_path(tmp_path, monkeypatch, "resend_uri.png")
    response = f"See: ![shot](file://{img.as_posix()})"

    def _make_adapter():
        return SimpleNamespace(
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

    adapter1 = _make_adapter()
    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter1,
    )
    adapter1.send_multiple_images.assert_awaited_once()

    adapter2 = _make_adapter()
    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        response,
        event,
        adapter2,
    )
    adapter2.send_multiple_images.assert_awaited_once()
    args, kwargs = adapter2.send_multiple_images.call_args
    images = kwargs.get("images")
    assert images is not None, "images kwarg missing"
    assert len(images) == 1, f"expected 1 image in resend turn, got {len(images)}: {images}"

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
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_multiple_images.assert_awaited_once_with(
        chat_id="chat-1",
        images=[(f"file://{media_file.as_posix()}", "")],
        metadata={"thread_id": "topic-1"},
    )


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
        metadata=routing_metadata,
    )
    adapter.send_multiple_images.assert_awaited_once_with(
        chat_id="chat-1",
        images=[(f"file://{media_file.as_posix()}", "")],
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
