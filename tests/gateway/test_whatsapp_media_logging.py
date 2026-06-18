"""Inbound media/document diagnostics for WhatsApp must go to the logger.

WhatsApp's poll loop and ``_build_message_event`` used to emit runtime
failures (image/voice cache misses, document-read errors, event-build
errors) via ``print()``. stdout is not captured by ``hermes logs``, so those
operator-facing diagnostics were invisible — the same gap Telegram fixed in
947e21b3d. These tests pin the behaviour to the logger and guard against a
regression back to ``print()``.
"""

import asyncio
import logging

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.whatsapp import WhatsAppAdapter

WA_LOGGER = "gateway.platforms.whatsapp"


def _make_adapter():
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))


def _dm_data(**overrides):
    data = {
        "chatId": "123@c.us",
        "senderId": "123@c.us",
        "senderName": "tester",
        "isGroup": False,
        "hasMedia": True,
        "body": "",
    }
    data.update(overrides)
    return data


def test_failed_image_cache_logs_warning_not_stdout(monkeypatch, caplog, capsys):
    """A cache miss is logged at WARNING and the original URL is kept."""
    adapter = _make_adapter()
    adapter._should_process_message = lambda data: True

    async def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("gateway.platforms.whatsapp.cache_image_from_url", _boom)

    url = "https://example.com/photo.jpg"
    data = _dm_data(mediaType="image/jpeg", mediaUrls=[url])

    with caplog.at_level(logging.WARNING, logger=WA_LOGGER):
        event = asyncio.run(adapter._build_message_event(data))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Failed to cache image" in r.getMessage() for r in warnings)
    # Graceful fallback: the message is still built with the original URL.
    assert event is not None
    assert event.media_urls == [url]
    assert event.media_types == ["image/jpeg"]
    # The diagnostic must not leak to stdout (the old print() behaviour).
    assert "Failed to cache image" not in capsys.readouterr().out


def test_successful_image_cache_logs_info(monkeypatch, caplog):
    """The cached-path success line is emitted at INFO (sibling parity)."""
    adapter = _make_adapter()
    adapter._should_process_message = lambda data: True

    async def _ok(*args, **kwargs):
        return "/cache/photo.jpg"

    monkeypatch.setattr("gateway.platforms.whatsapp.cache_image_from_url", _ok)

    data = _dm_data(mediaType="image/jpeg", mediaUrls=["https://example.com/photo.jpg"])

    with caplog.at_level(logging.INFO, logger=WA_LOGGER):
        event = asyncio.run(adapter._build_message_event(data))

    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("Cached user image" in r.getMessage() for r in infos)
    assert event is not None
    assert event.media_urls == ["/cache/photo.jpg"]


def test_failed_document_read_logs_warning(tmp_path, caplog):
    """A document text-read failure is logged and does not drop the event."""
    adapter = _make_adapter()
    adapter._should_process_message = lambda data: True

    # Absolute .txt path that does not exist → stat()/read_text() raises,
    # exercising the inbound document-read failure handler.
    missing = tmp_path / "note.txt"
    data = _dm_data(mediaType="application/octet-stream", mediaUrls=[str(missing)])

    with caplog.at_level(logging.WARNING, logger=WA_LOGGER):
        event = asyncio.run(adapter._build_message_event(data))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Failed to read document text" in r.getMessage() for r in warnings)
    # The event is still built; only the inline-text injection is skipped.
    assert event is not None
    assert event.message_type == MessageType.DOCUMENT
