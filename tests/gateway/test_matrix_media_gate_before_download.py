"""Matrix must gate (require_mention / allowed rooms) BEFORE downloading inbound media.

An unmentioned ``m.image`` in a gated group room used to be fetched from the homeserver
(``_download_and_cache_media``) and only then dropped by ``_resolve_message_context``.
The invariant: a dropped media event performs zero downloads; a mentioned one still does.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _make_adapter(monkeypatch):
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "true")
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
        display_name="Group Room", room_topic=None, server_name="example.org", chat_type="group"))
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/cached.png")
    return adapter


def _image_event(body):
    return SimpleNamespace(
        sender="@alice:example.org", event_id="$img1", room_id="!group:example.org",
        timestamp=int(time.time() * 1000),
        content={"body": body, "msgtype": "m.image", "url": "mxc://example.org/abc",
                 "info": {"mimetype": "image/png", "size": 1024}})


@pytest.mark.asyncio
@pytest.mark.parametrize("body, downloads, dispatched", [
    ("photo.png", 0, 0),           # unmentioned group media: never fetched
    ("@hermes:example.org look", 1, 1),  # mentioned: fetched and dispatched
])
async def test_unmentioned_group_media_is_not_downloaded(monkeypatch, body, downloads, dispatched):
    adapter = _make_adapter(monkeypatch)
    await adapter._on_room_message(_image_event(body))
    assert adapter._download_and_cache_media.await_count == downloads
    assert adapter.handle_message.await_count == dispatched
