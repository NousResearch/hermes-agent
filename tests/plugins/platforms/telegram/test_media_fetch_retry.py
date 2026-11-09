"""Inbound Telegram media survives one stalled getFile; a Bot API refusal is never retried."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


# SDK-free stand-ins: the adapter's classifiers key on PTB's class names.
class TimedOut(Exception):
    pass


class BadRequest(Exception):
    pass


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio_cache")
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")

    async def _no_wait(*_a, **_k):
        return None
    monkeypatch.setattr(asyncio, "sleep", _no_wait)
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a.handle_message = AsyncMock()
    return a


def _voice(failures):
    pending = list(failures)
    source = SimpleNamespace(calls=0, file_size=200)

    async def _get_file(*_a, **_k):
        source.calls += 1
        if pending:
            raise pending.pop(0)
        return SimpleNamespace(file_path="voice/file_1.oga",
                               download_as_bytearray=AsyncMock(return_value=bytearray(b"OggS" + b"\0" * 200)))
    source.get_file = _get_file
    return source


@pytest.mark.asyncio
@pytest.mark.parametrize("failures, calls, cached", [
    ([TimedOut()], 2, True),  # one stalled read: retried, the voice note is cached
    ([BadRequest("file is too big")], 1, False),  # a refusal is final: never retried
])
async def test_inbound_voice_fetch_retries_only_transient_errors(adapter, failures, calls, cached):
    source = _voice(failures)
    msg = SimpleNamespace(reply_text=AsyncMock(), chat=SimpleNamespace(id=1), message_thread_id=None)
    event = MessageEvent(text="", message_type=MessageType.VOICE)
    await adapter._cache_inbound_av(msg, event, source, "voice message", "voice", ".ogg", "audio/ogg")
    assert source.calls == calls
    assert bool(event.media_urls) is cached
    assert ("could not be downloaded" in (event.text or "")) is not cached

