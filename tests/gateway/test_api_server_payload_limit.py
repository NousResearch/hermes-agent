"""Tests for API server request body size limits.

Regression test for #13875: the API server rejected image payloads >1MB
because MAX_REQUEST_BYTES was hardcoded to 1MB and not passed to
aiohttp's web.Application(client_max_size=...).

The fix:
1. Raises the default from 1MB to 20MB (images/vision payloads)
2. Makes it configurable via API_SERVER_MAX_REQUEST_BYTES env var
3. Passes client_max_size to web.Application() so aiohttp enforces
   the same limit as the middleware
"""

from unittest.mock import MagicMock

import pytest

from gateway.platforms.api_server import MAX_REQUEST_BYTES, _parse_max_request_bytes


class TestMaxRequestBytesDefault:
    """Default MAX_REQUEST_BYTES should be large enough for images."""

    def test_default_is_10mb(self):
        assert MAX_REQUEST_BYTES == 10_000_000

    def test_default_allows_typical_image(self):
        """A base64-encoded 5MB image is ~6.7MB -- must be under the limit."""
        typical_b64_image_size = 5 * 1024 * 1024 * 4 // 3  # ~6.67 MB
        assert typical_b64_image_size < MAX_REQUEST_BYTES

    def test_old_1mb_limit_not_present(self):
        """The old hardcoded 1MB limit must not be the active value."""
        assert MAX_REQUEST_BYTES != 1_000_000, (
            "MAX_REQUEST_BYTES is still 1MB -- the old limit that blocked images"
        )


class TestParseMaxRequestBytes:
    """The env var parser should handle all input gracefully."""

    def test_returns_default_when_unset(self):
        assert _parse_max_request_bytes() == 10_000_000

    def test_custom_default(self):
        assert _parse_max_request_bytes(default=5_000_000) == 5_000_000

    def test_valid_env_override(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_MAX_REQUEST_BYTES", "50000000")
        assert _parse_max_request_bytes() == 50_000_000

    def test_env_can_reduce_limit(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_MAX_REQUEST_BYTES", "1000000")
        assert _parse_max_request_bytes() == 1_000_000

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        """Non-numeric env var should not crash — fall back to default."""
        monkeypatch.setenv("API_SERVER_MAX_REQUEST_BYTES", "20mb")
        result = _parse_max_request_bytes()
        assert result == 10_000_000

    def test_empty_string_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_MAX_REQUEST_BYTES", "")
        result = _parse_max_request_bytes()
        assert result == 10_000_000


class TestBodyLimitMiddleware:
    """The body_limit_middleware should reject oversized requests."""

    @pytest.fixture
    def middleware(self):
        from gateway.platforms.api_server import body_limit_middleware
        if body_limit_middleware is None:
            pytest.skip("aiohttp not available")
        return body_limit_middleware

    @pytest.mark.asyncio
    async def test_allows_request_under_limit(self, middleware):
        """A 5MB request (under 10MB limit) should pass through."""
        from aiohttp import web
        from unittest.mock import AsyncMock

        request = MagicMock()
        request.method = "POST"
        request.headers = {"Content-Length": str(5_000_000)}

        handler = AsyncMock(return_value=web.Response(text="ok"))
        response = await middleware(request, handler)

        handler.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_rejects_request_over_limit(self, middleware):
        """A request exceeding MAX_REQUEST_BYTES should get 413."""
        request = MagicMock()
        request.method = "POST"
        request.headers = {"Content-Length": str(MAX_REQUEST_BYTES + 1)}

        handler = MagicMock()
        response = await middleware(request, handler)

        handler.assert_not_called()
        assert response.status == 413

    @pytest.mark.asyncio
    async def test_allows_get_regardless_of_size(self, middleware):
        """GET requests should not be checked for body size."""
        from aiohttp import web
        from unittest.mock import AsyncMock

        request = MagicMock()
        request.method = "GET"
        request.headers = {"Content-Length": str(999_999_999)}

        handler = AsyncMock(return_value=web.Response(text="ok"))
        response = await middleware(request, handler)

        handler.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_rejects_invalid_content_length(self, middleware):
        """Non-numeric Content-Length should get 400."""
        request = MagicMock()
        request.method = "POST"
        request.headers = {"Content-Length": "not-a-number"}

        handler = MagicMock()
        response = await middleware(request, handler)

        assert response.status == 400


class TestApplicationClientMaxSize:
    """web.Application must receive client_max_size so aiohttp enforces
    the limit on chunked transfers (not just Content-Length header)."""

    def test_source_passes_client_max_size(self):
        """Guard: client_max_size must be set in Application constructor."""
        from pathlib import Path
        source = (Path(__file__).parent.parent.parent
                  / "gateway" / "platforms" / "api_server.py").read_text()
        assert "client_max_size=MAX_REQUEST_BYTES" in source, (
            "web.Application() must pass client_max_size=MAX_REQUEST_BYTES "
            "so aiohttp enforces the same limit as the middleware"
        )
