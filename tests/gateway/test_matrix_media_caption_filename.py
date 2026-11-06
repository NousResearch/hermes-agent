"""Matrix v1.10 media captions must not become the cached file's name.

Per the spec, when ``content.filename`` is set and differs from ``body``,
``body`` is the user's caption and the original file name lives in
``filename``. The adapter cached the document under the caption text, losing
the extension — a PDF archived without ``.pdf`` (#135898). These tests pin
that the cache keys off ``filename`` when present and falls back to ``body``.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _make_adapter(monkeypatch):
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    from gateway.config import PlatformConfig
    from gateway.platforms import base as platforms_base
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org"}))
    adapter._startup_ts = time.time() - 10
    adapter._client = SimpleNamespace(download_media=AsyncMock(return_value=b"fixture-bytes"))
    adapter._voice_may_park = lambda *args, **kwargs: False
    adapter._resolve_message_context = AsyncMock(return_value=SimpleNamespace(room_id="!room:example.org"))
    adapter._build_inbound_event = AsyncMock(return_value=None)  # dispatch is not under test
    adapter.handle_message = AsyncMock()
    doc_cache = AsyncMock(return_value="/cache/documents/doc_1_invoice.pdf")
    audio_cache = AsyncMock(return_value="/cache/audios/aud_1_song.mp3")
    monkeypatch.setattr(platforms_base, "cache_document_from_bytes_async", doc_cache)
    monkeypatch.setattr(platforms_base, "cache_audio_from_bytes_async", audio_cache)
    return adapter, doc_cache, audio_cache


async def _handle(adapter, content):
    await adapter._handle_media_message(
        "!room:example.org", "@alice:example.org", "$doc1", time.time() * 1000, content, {},
        content["msgtype"])


@pytest.mark.asyncio
async def test_captioned_document_caches_under_declared_filename(monkeypatch):
    adapter, doc_cache, _ = _make_adapter(monkeypatch)
    await _handle(adapter, {
        "body": "please file this.", "filename": "invoice.pdf", "msgtype": "m.file",
        "url": "mxc://example.org/abc", "info": {"mimetype": "application/pdf", "size": 1024}})
    assert doc_cache.await_args.args == (b"fixture-bytes", "invoice.pdf")


@pytest.mark.asyncio
async def test_uncaptioned_document_falls_back_to_body(monkeypatch):
    adapter, doc_cache, _ = _make_adapter(monkeypatch)
    await _handle(adapter, {
        "body": "invoice.pdf", "msgtype": "m.file",
        "url": "mxc://example.org/abc", "info": {"mimetype": "application/pdf", "size": 1024}})
    assert doc_cache.await_args.args == (b"fixture-bytes", "invoice.pdf")


@pytest.mark.asyncio
async def test_blank_filename_is_ignored_rather_than_cached_as_name(monkeypatch):
    adapter, doc_cache, _ = _make_adapter(monkeypatch)
    await _handle(adapter, {
        "body": "report.pdf", "filename": "   ", "msgtype": "m.file",
        "url": "mxc://example.org/abc", "info": {"mimetype": "application/pdf", "size": 1024}})
    assert doc_cache.await_args.args == (b"fixture-bytes", "report.pdf")


@pytest.mark.asyncio
async def test_captioned_audio_keeps_real_extension(monkeypatch):
    """The audio branch derives its extension the same way — a caption like `listen to this`
    has no suffix, so without `filename` the clip silently falls back to `.ogg`."""
    adapter, _, audio_cache = _make_adapter(monkeypatch)
    await _handle(adapter, {
        "body": "listen to this", "filename": "song.mp3", "msgtype": "m.audio",
        "url": "mxc://example.org/abc2", "info": {"mimetype": "audio/mpeg", "size": 2048}})
    assert audio_cache.await_args.kwargs == {"ext": ".mp3"}
