"""A Telegram attachment download survives a transient network blip.

``get_file()`` was called exactly once: one ``TimedOut`` (Telegram processing the
file can outlast the 20 s read timeout) or one dropped connection lost the voice
note / audio / document outright, with nothing but a "could not be downloaded"
note to show for it. ``_fetch_telegram_file`` retries the whole fetch --
``get_file()`` plus the download that follows -- on ``TimedOut``/``NetworkError``,
and never on ``BadRequest`` (a permanent answer to what was asked, even though
``python-telegram-bot`` makes it a ``NetworkError`` subclass) nor on
``Forbidden``/``InvalidToken`` (not network errors at all).
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest, Forbidden, InvalidToken, TimedOut

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def sleeps(monkeypatch):
    """Record the retry pauses instead of waiting them out."""
    recorded: list[float] = []

    async def _sleep(delay, *args, **kwargs):
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return recorded


@pytest.fixture
def adapter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._surface_media_cache_failure = AsyncMock()
    a.handle_message = AsyncMock()
    return a


def _event(message_type=MessageType.VOICE) -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=message_type,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _file_obj(data: bytes = b"voice-bytes", path: str = "voice/file_1.oga"):
    f = AsyncMock()
    f.download_as_bytearray = AsyncMock(return_value=bytearray(data))
    f.file_path = path
    return f


class _Source:
    """A Telegram attachment whose ``get_file()`` plays back a scripted outcome list."""

    def __init__(self, outcomes, file_size: int = 64, file_name=None, mime_type=None):
        self.file_size = file_size
        self.file_name = file_name
        self.mime_type = mime_type
        self.outcomes = list(outcomes)
        self.calls = 0

    async def get_file(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_voice_download_retries_timed_out_and_lands(adapter, sleeps, caplog):
    caplog.set_level(logging.WARNING)
    source = _Source([TimedOut("timed out"), TimedOut("timed out"), _file_obj()])
    event = _event()

    handled = await adapter._cache_inbound_av(
        SimpleNamespace(), event, source, "voice message", "voice", ".ogg", "audio/ogg")

    assert handled is False
    assert source.calls == 3
    assert sleeps == [2.0, 5.0]
    assert len(event.media_urls) == 1 and event.media_types == ["audio/ogg"]
    assert adapter._surface_media_cache_failure.await_count == 0
    assert "attempt 1/3" in caplog.text and "attempt 2/3" in caplog.text


@pytest.mark.asyncio
async def test_bad_request_is_not_retried(adapter, sleeps):
    source = _Source([BadRequest("file is too big")])
    event = _event()

    handled = await adapter._cache_inbound_av(
        SimpleNamespace(), event, source, "voice message", "voice", ".ogg", "audio/ogg")

    assert handled is False
    assert source.calls == 1
    assert sleeps == []
    assert event.media_urls == []
    adapter._surface_media_cache_failure.assert_awaited_once()
    assert isinstance(adapter._surface_media_cache_failure.await_args.args[3], BadRequest)


@pytest.mark.parametrize("error", [Forbidden("bot was blocked by the user"), InvalidToken("invalid token")])
@pytest.mark.asyncio
async def test_non_network_telegram_errors_are_not_retried(adapter, sleeps, error):
    source = _Source([error])
    event = _event()

    await adapter._cache_inbound_av(
        SimpleNamespace(), event, source, "voice message", "voice", ".ogg", "audio/ogg")

    assert source.calls == 1
    assert sleeps == []
    assert isinstance(adapter._surface_media_cache_failure.await_args.args[3], type(error))


@pytest.mark.asyncio
async def test_three_timeouts_keep_the_old_failure_path(adapter, sleeps, caplog):
    caplog.set_level(logging.WARNING)
    source = _Source([TimedOut("a"), TimedOut("b"), TimedOut("c")])
    event = _event()

    handled = await adapter._cache_inbound_av(
        SimpleNamespace(), event, source, "voice message", "voice", ".ogg", "audio/ogg")

    assert handled is False
    assert source.calls == 3
    assert sleeps == [2.0, 5.0]
    assert event.media_urls == []
    adapter._surface_media_cache_failure.assert_awaited_once()
    assert isinstance(adapter._surface_media_cache_failure.await_args.args[3], TimedOut)
    assert "attempt 3/3" not in caplog.text  # the last failure re-raises instead of announcing a retry


@pytest.mark.asyncio
async def test_document_download_retries_network_error(adapter, sleeps):
    from telegram.error import NetworkError

    doc = _Source([NetworkError("connection reset"), _file_obj(b"%PDF-1.4", "documents/report.pdf")],
                  file_name="report.pdf", mime_type="application/pdf")
    msg = SimpleNamespace(document=doc)
    event = MessageEvent(
        text="report",
        message_type=MessageType.DOCUMENT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )

    handled = await adapter._cache_inbound_document(msg, event)

    assert handled is False
    assert doc.calls == 2
    assert sleeps == [2.0]
    assert len(event.media_urls) == 1
