"""A failed post-stream upload must reach the user as a notice (#141585).

``GatewayRunner._deliver_media_from_response`` uploads ``MEDIA:`` attachments
*after* the visible reply has been streamed. The text ("here is the file") is
already on screen, so an upload that returns ``SendResult(success=False)`` — or
raises — is a silent drop unless the adapter's failure notice runs. The
non-streaming path (``BasePlatformAdapter._deliver_media_attachments``, #66797)
notifies per failed attachment; this file pins the same contract for the
post-stream path.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _event():
    source = SessionSource(platform=Platform.SLACK, chat_id="C123CHAN", chat_type="group", thread_id=None)
    return MessageEvent(
        text="hi", message_type=MessageType.TEXT, source=source, message_id="171.000001",
    )


def _fake_runner(thread_meta):
    return SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )


def _adapter(**overrides):
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="imgs")),
        _notify_media_delivery_failure=AsyncMock(),
    )
    for name, value in overrides.items():
        setattr(adapter, name, value)
    return adapter


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return media_file.resolve()


@pytest.mark.asyncio
async def test_failed_document_upload_sends_a_failure_notice(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter = _adapter(send_document=AsyncMock(return_value=SendResult(success=False, error="rejected: mime")))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Here is the report.\nMEDIA:{media_file}", _event(), adapter,
    )

    adapter.send_document.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_awaited_once_with(
        "C123CHAN", str(media_file), is_voice=False, metadata={},
    )


@pytest.mark.asyncio
async def test_raised_document_upload_also_sends_a_failure_notice(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter = _adapter(send_document=AsyncMock(side_effect=RuntimeError("transport gone")))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Here is the report.\nMEDIA:{media_file}", _event(), adapter,
    )

    adapter._notify_media_delivery_failure.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_image_batch_sends_a_failure_notice(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "chart.png")
    adapter = _adapter(send_multiple_images=AsyncMock(return_value=SendResult(success=False, error="too large")))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Here is the chart.\nMEDIA:{media_file}", _event(), adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_awaited_once()
    assert str(media_file) in adapter._notify_media_delivery_failure.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_failed_url_image_batch_sends_a_failure_notice(raises):
    sender = (
        AsyncMock(side_effect=RuntimeError("transport gone")) if raises
        else AsyncMock(return_value=SendResult(success=False, error="too large"))
    )
    adapter = _adapter(send_multiple_images=sender)
    metadata = {"thread_id": "thread-1"}

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(metadata),
        "![chart](https://example.com/images/chart.png?size=large)", _event(), adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_awaited_once_with(
        "C123CHAN", "chart.png", is_voice=False, metadata=metadata,
    )


@pytest.mark.asyncio
async def test_malformed_image_url_does_not_drop_the_remaining_media(tmp_path, monkeypatch):
    """A URL that urlsplit rejects (unclosed IPv6 bracket) still gets its notice, and the
    MEDIA file after it still goes out."""
    document = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter = _adapter(send_multiple_images=AsyncMock(return_value=SendResult(success=False, error="bad url")))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"![x](https://[::1/x.png)\nMEDIA:{document}", _event(), adapter,
    )

    adapter._notify_media_delivery_failure.assert_awaited_once_with(
        "C123CHAN", "x.png", is_voice=False, metadata={},
    )
    adapter.send_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_voice_upload_preserves_voice_notice(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter = _adapter(send_voice=AsyncMock(return_value=SendResult(success=False, error="rejected")))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"[[audio_as_voice]]\nMEDIA:{media_file}", _event(), adapter,
    )

    adapter.send_voice.assert_awaited_once_with(
        chat_id="C123CHAN", audio_path=str(media_file), metadata={}, is_voice=True,
    )
    adapter._notify_media_delivery_failure.assert_awaited_once_with(
        "C123CHAN", str(media_file), is_voice=True, metadata={},
    )


@pytest.mark.asyncio
async def test_successful_uploads_stay_silent(tmp_path, monkeypatch):
    image = _allowed_media_path(tmp_path, monkeypatch, "chart.png")
    document = _allowed_media_path(tmp_path, monkeypatch, "notes.txt")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Two files.\nMEDIA:{image}\nMEDIA:{document}", _event(), adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    adapter.send_document.assert_awaited_once()
    adapter._notify_media_delivery_failure.assert_not_awaited()
