"""Tests for bounded HTTP response reads in Discord image/download paths.

Companion to #60122, #60112 (REST body bounding) — extends the same
resource-limiting pattern to image/animation/attachment downloads
in the Discord adapter that were left unbounded.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from plugins.platforms.discord.adapter import (
    _read_response_bytes_bounded,
    _DISCORD_IMAGE_DOWNLOAD_MAX_BYTES,
    _DISCORD_ATTACHMENT_DOWNLOAD_MAX_BYTES,
)


class AsyncContextManagerMock:
    """Mock that supports ``async with``."""

    def __init__(self, return_value):
        self._return_value = return_value

    async def __aenter__(self):
        return self._return_value

    async def __aexit__(self, *args):
        pass


class TestReadResponseBytesBounded:
    def test_reads_within_limit(self):
        resp = MagicMock()
        resp.content = MagicMock()
        resp.content.read = AsyncMock(return_value=b"x" * 100)

        result = pytest.run_sync(
            _read_response_bytes_bounded(resp, 200)
        )
        assert result == b"x" * 100
        resp.content.read.assert_awaited_once_with(201)

    def test_raises_on_overflow(self):
        resp = MagicMock()
        resp.content = MagicMock()
        resp.content.read = AsyncMock(return_value=b"x" * 101)
        resp.close = MagicMock()

        with pytest.raises(ValueError, match="exceeded 100 bytes"):
            pytest.run_sync(
                _read_response_bytes_bounded(resp, 100)
            )
        resp.close.assert_called_once()

    def test_exact_limit_passes(self):
        resp = MagicMock()
        resp.content = MagicMock()
        resp.content.read = AsyncMock(return_value=b"x" * 100)

        result = pytest.run_sync(
            _read_response_bytes_bounded(resp, 100)
        )
        assert result == b"x" * 100


class TestImageDownloadLimits:
    def test_constants_are_reasonable(self):
        assert _DISCORD_IMAGE_DOWNLOAD_MAX_BYTES == 50 * 1024 * 1024
        assert _DISCORD_ATTACHMENT_DOWNLOAD_MAX_BYTES == 100 * 1024 * 1024

    def test_image_limit_greater_than_zero(self):
        assert _DISCORD_IMAGE_DOWNLOAD_MAX_BYTES > 0
        assert _DISCORD_ATTACHMENT_DOWNLOAD_MAX_BYTES > 0
