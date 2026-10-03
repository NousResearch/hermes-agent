"""``[[as_document]]`` must keep video files as documents on the gateway delivery paths.

The standalone ``hermes send`` path has its own coverage
(``tests/tools/test_telegram_send_message_as_document.py``); this file pins the two
gateway-side delivery implementations:

- ``BasePlatformAdapter._deliver_media_attachments`` (non-streaming replies), and
- ``GatewayNotificationsMixin._deliver_media_from_response`` (streamed final replies).

Both previously routed every video extension to ``send_video`` regardless of the
``[[as_document]]`` directive (``force_document_attachments`` only forced *images*
to documents), so a requested file attachment arrived as an inline video message.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.session import Platform, SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter.send_video = AsyncMock(return_value=SendResult(success=True))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True))
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True))
    adapter.extract_images = lambda *_a, **_k: None
    adapter._notify_media_delivery_failure = AsyncMock()
    return adapter


def _event() -> MessageEvent:
    return MessageEvent(
        text="", source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345"))


@pytest.fixture(autouse=True)
def _stub_path_filter(monkeypatch):
    """The MEDIA path-safety filter resolves allow-roots from the host hermes home
    (``~/.hermes/profiles``) and has its own coverage; stub it so these tests measure
    the delivery-routing decision — which method each attachment type goes through."""
    from gateway.platforms.base import BasePlatformAdapter
    monkeypatch.setattr(
        BasePlatformAdapter, "filter_media_delivery_paths",
        staticmethod(lambda media_files, **kw: list(media_files or [])))


def _clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x1c" + b"ftyp" + b"\x00" * 32)
    return clip


@pytest.mark.asyncio
async def test_force_document_attachment_delivers_video_as_document(tmp_path) -> None:
    adapter = _adapter()
    clip = _clip(tmp_path)
    results: list = []

    await adapter._deliver_media_attachments(
        _event(), [(str(clip), False)], [], force_document_attachments=True,
        human_delay=0, metadata={}, record_delivery=results.append)

    adapter.send_document.assert_awaited_once()
    adapter.send_video.assert_not_awaited()
    assert [r.success for r in results] == [True]
    assert adapter.send_document.await_args.kwargs["file_path"] == str(clip)
    assert adapter.send_document.await_args.kwargs["disable_content_type_detection"] is True


@pytest.mark.asyncio
async def test_video_without_force_document_still_sends_as_video(tmp_path) -> None:
    adapter = _adapter()
    clip = _clip(tmp_path)

    await adapter._deliver_media_attachments(
        _event(), [(str(clip), False)], [], force_document_attachments=False,
        human_delay=0, metadata={}, record_delivery=[].append)

    adapter.send_video.assert_awaited_once()
    adapter.send_document.assert_not_awaited()


class _Notifier(GatewayNotificationsMixin):
    """Minimal host for the mixin method under test."""


@pytest.mark.asyncio
async def test_streamed_reply_force_document_delivers_video_as_document(tmp_path) -> None:
    adapter = _adapter()
    clip = _clip(tmp_path)
    notifier = _Notifier()

    await notifier._deliver_media_from_response(
        f"Here it is.\nMEDIA:{clip}[[as_document]]", _event(), adapter, thread_metadata={})

    adapter.send_document.assert_awaited_once()
    adapter.send_video.assert_not_awaited()
    assert adapter.send_document.await_args.kwargs["file_path"] == str(clip)
    assert adapter.send_document.await_args.kwargs["disable_content_type_detection"] is True


@pytest.mark.asyncio
async def test_streamed_reply_without_directive_still_sends_as_video(tmp_path) -> None:
    adapter = _adapter()
    clip = _clip(tmp_path)
    notifier = _Notifier()

    await notifier._deliver_media_from_response(
        f"Here it is.\nMEDIA:{clip}", _event(), adapter, thread_metadata={})

    adapter.send_video.assert_awaited_once()
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_forced_document_file_sends_without_the_file_form_flag(tmp_path) -> None:
    """Ordinary (non-``[[as_document]]``) document delivery must not touch server-side
    content-type detection — previews and waveforms for plain documents stay intact."""
    adapter = _adapter()
    doc = tmp_path / "notes.pdf"
    doc.write_bytes(b"%PDF-1.4")

    await adapter._deliver_media_attachments(
        _event(), [], [str(doc)], force_document_attachments=False,
        human_delay=0, metadata={}, record_delivery=[].append)

    adapter.send_document.assert_awaited_once()
    assert "disable_content_type_detection" not in adapter.send_document.await_args.kwargs
