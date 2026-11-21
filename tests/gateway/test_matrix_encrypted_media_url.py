"""Encrypted Matrix media must be fetched from ``file.url``, whatever the top-level ``url`` says.

NeoChat sends encrypted screenshots with a redundant local top-level ``url``
(``file:///…/screenshot.png``) next to the real encrypted attachment in ``file``. The adapter
rejected the whole event as "non-MXC" before ever looking at ``file.url``, so the image and its
caption were silently dropped (#154251). The decryption keys in ``file`` belong to ``file.url``,
so that URI is the one to validate and download; a non-MXC location is still refused.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageType

_CACHED = "/tmp/hermes-cache/screenshot.png"
_FILE_MXC = "mxc://matrix.example.org/example-media-id"


def _make_adapter(monkeypatch):
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    from gateway.config import PlatformConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org"}))
    adapter._startup_ts = time.time() - 10
    adapter.handle_message = AsyncMock()
    adapter._client = None
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
        display_name="Alice", room_topic=None, server_name="example.org", chat_type="dm"))
    adapter._is_dm_room = AsyncMock(return_value=True)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = lambda *a, **k: None
    adapter._download_and_cache_media = AsyncMock(return_value=_CACHED)
    return adapter


def _encrypted_file(url=_FILE_MXC):
    return {"url": url, "key": {"k": "key"}, "iv": "iv", "hashes": {"sha256": "hash"}, "v": "v2"}


async def _deliver(adapter, content):
    await adapter._handle_media_message(
        room_id="!dm:example.org", sender="@alice:example.org", event_id="$enc-image",
        event_ts=time.time(), source_content=content, relates_to={}, msgtype=content["msgtype"])


@pytest.mark.asyncio
async def test_redundant_local_top_level_url_does_not_drop_encrypted_image(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    encrypted_file = _encrypted_file()
    await _deliver(adapter, {
        "msgtype": "m.image", "body": "look at this",
        "url": "file:///home/example/.cache/KDE/neochat/screenshots/screenshot.png",
        "info": {"mimetype": "image/png", "size": 269764}, "file": encrypted_file})

    adapter._download_and_cache_media.assert_awaited_once()
    args = adapter._download_and_cache_media.await_args.args
    assert args[0] == _FILE_MXC
    assert args[2] is encrypted_file
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "look at this"
    assert event.media_urls == [_CACHED]
    assert event.message_type == MessageType.PHOTO


@pytest.mark.asyncio
async def test_encrypted_image_downloads_file_url_not_a_different_top_level_mxc(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    encrypted_file = _encrypted_file()
    await _deliver(adapter, {
        "msgtype": "m.image", "body": "screenshot.png", "url": "mxc://matrix.example.org/other-media-id",
        "info": {"mimetype": "image/png"}, "file": encrypted_file})

    args = adapter._download_and_cache_media.await_args.args
    assert args[0] == _FILE_MXC
    assert args[2] is encrypted_file
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_encrypted_media_with_non_mxc_file_url_is_still_rejected(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    await _deliver(adapter, {
        "msgtype": "m.image", "body": "screenshot.png", "info": {"mimetype": "image/png"},
        "file": _encrypted_file(url="file:///etc/passwd")})

    adapter._download_and_cache_media.assert_not_called()
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_unencrypted_media_with_non_mxc_url_is_still_rejected(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    await _deliver(adapter, {
        "msgtype": "m.image", "body": "screenshot.png", "url": "file:///home/example/screenshot.png",
        "info": {"mimetype": "image/png"}})

    adapter._download_and_cache_media.assert_not_called()
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_unencrypted_mxc_media_is_downloaded_without_decryption(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    await _deliver(adapter, {
        "msgtype": "m.image", "body": "screenshot.png", "url": "mxc://matrix.example.org/plain",
        "info": {"mimetype": "image/png"}})

    args = adapter._download_and_cache_media.await_args.args
    assert args[0] == "mxc://matrix.example.org/plain"
    assert args[2] is None
    adapter.handle_message.assert_awaited_once()
