"""Post-stream media delivery is explicit-only (#20834).

``GatewayRunner._deliver_media_from_response`` runs AFTER streaming has sent
the visible reply. At that point a bare local filesystem path in the response
text is either text the user already saw, or stale inspected/tool content —
it is NOT an attachment request. Only explicit ``MEDIA:`` directives may
trigger post-stream uploads.

The non-streaming path (``gateway/platforms/base.py``) keeps its bare-path
auto-detect (``extract_local_files``) — that path controls what text is sent
and can strip the path from the visible reply, so auto-attach is intentional
there. This file pins the asymmetry.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _event():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123CHAN",
        chat_type="group",
        thread_id=None,
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="171.000001",
    )


def _fake_runner(thread_meta):
    return SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )


def _adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="imgs")),
        send=AsyncMock(return_value=SendResult(success=True, message_id="notice")),
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
async def test_bare_local_path_in_streamed_reply_is_not_uploaded(tmp_path, monkeypatch):
    """The #20834 shape: visible reply contains a bare path (from inspected
    content), no MEDIA: directive — nothing may be uploaded post-stream."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "mockup.png")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        f"The design lives at {media_file} if you want to look later.",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()
    adapter.send_video.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_media_tag_still_delivers_post_stream(tmp_path, monkeypatch):
    """Explicit MEDIA: directives keep working after the #20834 fix."""
    media_file = _allowed_media_path(tmp_path, monkeypatch, "chart.png")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}),
        f"Here is the chart.\nMEDIA:{media_file}",
        _event(),
        adapter,
    )

    adapter.send_multiple_images.assert_awaited_once()
    images_kwargs = adapter.send_multiple_images.await_args.kwargs
    assert images_kwargs["chat_id"] == "C123CHAN"
    assert images_kwargs["images"][0][0] == f"file://{quote(str(media_file))}"


@pytest.mark.asyncio
async def test_streamed_slack_delivers_one_cached_pdf(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"Your report is ready.\nMEDIA:{media_file}", _event(), adapter,
    )

    adapter.send_document.assert_awaited_once()
    assert adapter.send_document.await_args.kwargs["file_path"] == str(media_file)
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_streamed_slack_delivers_multiple_cached_pdfs(tmp_path, monkeypatch):
    first = _allowed_media_path(tmp_path, monkeypatch, "first.pdf")
    second = _allowed_media_path(tmp_path, monkeypatch, "second.pdf")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{first}\nMEDIA:{second}", _event(), adapter,
    )

    assert adapter.send_document.await_count == 2
    assert [call.kwargs["file_path"] for call in adapter.send_document.await_args_list] == [
        str(first), str(second),
    ]
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_streamed_slack_rejected_media_path_gets_generic_failure_notice(tmp_path, monkeypatch):
    _allowed_media_path(tmp_path, monkeypatch, "allowed.pdf")
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    rejected = tmp_path / "outside-cache.pdf"
    rejected.write_bytes(b"pdf")
    adapter = _adapter()

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{rejected}", _event(), adapter,
    )

    adapter.send_document.assert_not_awaited()
    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.args[1]
    assert "couldn't attach" in notice
    assert str(rejected) not in notice


@pytest.mark.asyncio
async def test_streamed_slack_failed_pdf_upload_gets_generic_failure_notice(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter = _adapter()
    adapter.send_document.return_value = SendResult(success=False, error="upload failed")

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{media_file}", _event(), adapter,
    )

    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.args[1]
    assert "couldn't attach" in notice
    assert str(media_file) not in notice


@pytest.mark.asyncio
async def test_streamed_slack_raised_pdf_upload_gets_generic_failure_notice(tmp_path, monkeypatch):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter = _adapter()
    adapter.send_document.side_effect = RuntimeError("provider unavailable")

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({}), f"MEDIA:{media_file}", _event(), adapter,
    )

    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.args[1]
    assert "couldn't attach" in notice
    assert str(media_file) not in notice
