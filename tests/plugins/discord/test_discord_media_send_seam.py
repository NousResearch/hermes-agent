"""Behavioral seam tests for Discord media-send ownership."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import plugins.platforms.discord.adapter as adapter_mod
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.media_send_mixin import DiscordMediaSendMixin


MOVED_PUBLIC_METHODS = (
    "send_multiple_images",
    "send_image_file",
    "send_image",
    "send_animation",
    "send_video",
    "send_document",
)
MOVED_METHODS = MOVED_PUBLIC_METHODS + (
    "_send_multiple_images_via_base_with_budget",
)


def test_discord_media_methods_are_owned_by_the_mixin():
    for method_name in MOVED_METHODS:
        assert getattr(DiscordAdapter, method_name) is getattr(
            DiscordMediaSendMixin, method_name
        )
        assert method_name not in DiscordAdapter.__dict__


def test_discord_media_mixin_is_first_and_typing_stays_on_adapter():
    assert DiscordAdapter.__mro__[1] is DiscordMediaSendMixin
    assert "send_typing" not in DiscordMediaSendMixin.__dict__
    assert "stop_typing" not in DiscordMediaSendMixin.__dict__
    assert "send_typing" in DiscordAdapter.__dict__
    assert "stop_typing" in DiscordAdapter.__dict__


@pytest.mark.asyncio
async def test_moved_send_image_observes_adapter_level_patch_seams(monkeypatch):
    url = "https://cdn.example.test/image.png"
    safety_checks = []
    guard_calls = []

    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    http_client = FakeHttpClient()

    async def record_safety_check(candidate):
        safety_checks.append(candidate)
        return True

    patched_async_is_safe_url = AsyncMock(side_effect=record_safety_check)

    async def patched_fetch(client, candidate, *, timeout, request_kwargs, download_budget=None):
        guard_calls.append((client, candidate, timeout, request_kwargs, download_budget))
        return 200, b"\x89PNG\r\n\x1a\n", {}

    fake_discord = SimpleNamespace(File=MagicMock())
    monkeypatch.setattr(adapter_mod, "discord", fake_discord)
    monkeypatch.setattr(
        adapter_mod, "async_is_safe_url", patched_async_is_safe_url
    )
    monkeypatch.setattr(
        adapter_mod,
        "_create_discord_image_http_client",
        lambda _proxy: http_client,
    )
    monkeypatch.setattr(
        adapter_mod,
        "_read_url_image_with_redirect_guard",
        patched_fetch,
    )

    channel = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(id="message-1")),
    )
    client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(return_value=channel),
    )
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = adapter_mod.Platform.DISCORD
    adapter._client = client
    adapter._is_forum_parent = MagicMock(return_value=False)

    result = await adapter.send_image("123", url, caption="caption")

    assert result.success is True
    assert result.message_id == "message-1"
    assert safety_checks == [url]
    patched_async_is_safe_url.assert_awaited_once_with(url)
    assert len(guard_calls) == 1
    assert guard_calls[0][0] is http_client
    assert guard_calls[0][1] == url
    channel.send.assert_awaited_once()
