"""Regression tests for the Discord outbound image fetch module seam."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

import plugins.platforms.discord.adapter as discord_adapter
import plugins.platforms.discord.outbound_image_fetch as outbound_image_fetch


def test_adapter_reexports_outbound_image_fetch_primitives():
    for name in (
        "_DiscordImageDownloadBudget",
        "_DISCORD_IMAGE_DOWNLOAD_MAX_BYTES",
        "_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES",
        "_DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT",
        "_read_response_bytes_bounded",
        "_discord_image_extension_from_bytes",
    ):
        assert getattr(discord_adapter, name) is getattr(outbound_image_fetch, name)


@pytest.mark.asyncio
async def test_adapter_redirect_wrapper_uses_adapter_url_safety_patch(monkeypatch):
    url = "https://cdn.example.test/image.png"
    safety_checks: list[str] = []

    async def record_safety_check(candidate: str) -> bool:
        safety_checks.append(candidate)
        return True

    patched_async_is_safe_url = AsyncMock(side_effect=record_safety_check)

    class Response:
        status_code = 200
        headers: dict[str, str] = {}

        async def __aenter__(self) -> "Response":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def aiter_bytes(self, *, chunk_size=None):
            yield b"\x89PNG\r\n\x1a\n"

    class Client:
        def stream(self, method: str, candidate: str, **kwargs: Any) -> Response:
            assert method == "GET"
            assert candidate == url
            assert kwargs["follow_redirects"] is False
            return Response()

    monkeypatch.setattr(
        discord_adapter, "async_is_safe_url", patched_async_is_safe_url
    )
    module_async_is_safe_url = AsyncMock(return_value=False)
    monkeypatch.setattr(
        outbound_image_fetch, "async_is_safe_url", module_async_is_safe_url
    )

    status, body, _headers = await discord_adapter._read_url_image_with_redirect_guard(
        Client(),
        url,
        timeout=30.0,
        request_kwargs={},
    )

    assert (status, body) == (200, b"\x89PNG\r\n\x1a\n")
    assert safety_checks == [url]
    patched_async_is_safe_url.assert_awaited_once_with(url)
    module_async_is_safe_url.assert_not_awaited()


def test_adapter_client_wrapper_uses_adapter_factory_patch(monkeypatch):
    sentinel_client = object()
    captured: dict[str, Any] = {}

    def patched_factory(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel_client

    monkeypatch.setattr(discord_adapter, "create_ssrf_safe_async_client", patched_factory)

    assert discord_adapter._create_discord_image_http_client("http://proxy.test") is sentinel_client
    assert captured["proxy"] == "http://proxy.test"
    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False
