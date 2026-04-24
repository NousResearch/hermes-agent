# -*- coding: utf-8 -*-
"""Unit tests for gateway.platforms.qqbot.core.media_loader.

Replaces the old test_qqbot_media.py.
All imports reference core/ directly; no hermes symbols used.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from gateway.platforms.qqbot.core.media_loader import MediaLoader, MediaUploader
from gateway.platforms.qqbot.core.api_client import QQApiClient


# ── MediaLoader.load ──────────────────────────────────────────────────

class TestMediaLoaderLoad:
    def test_url_passthrough(self):
        data, ct, name = MediaLoader.load("https://example.com/img.jpg")
        assert data == "https://example.com/img.jpg"
        assert "image" in ct
        assert name == "img.jpg"

    def test_url_with_file_name_override(self):
        __, __, name = MediaLoader.load("https://example.com/img.jpg", file_name="custom.jpg")
        assert name == "custom.jpg"

    def test_local_file(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        data, ct, name = MediaLoader.load(str(f))
        assert len(data) > 0
        assert "image" in ct
        assert name == "test.png"

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="required"):
            MediaLoader.load("")

    def test_placeholder_source_raises(self):
        with pytest.raises(ValueError, match="placeholder"):
            MediaLoader.load("<path>")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            MediaLoader.load("/nonexistent/file.jpg")


# ── MediaLoader.is_url ────────────────────────────────────────────────

class TestMediaLoaderIsUrl:
    def test_http(self):
        assert MediaLoader.is_url("http://example.com") is True

    def test_https(self):
        assert MediaLoader.is_url("https://example.com") is True

    def test_local_path(self):
        assert MediaLoader.is_url("/path/to/file.jpg") is False

    def test_empty(self):
        assert MediaLoader.is_url("") is False


# ── MediaUploader.upload ──────────────────────────────────────────────

class TestMediaUploader:
    def _make(self):
        api = QQApiClient("app", "secret")
        api._access_token = "tok"
        api._token_expires_at = 9999999999.0
        return MediaUploader(api, log_tag="test"), api

    @pytest.mark.asyncio
    async def test_upload_c2c_success(self):
        uploader, api = self._make()
        api.upload_c2c_file = mock.AsyncMock(return_value={"file_info": "fi-abc"})
        file_info = await uploader.upload("c2c", "uid-1", "https://x.com/img.jpg", 1)
        assert file_info == "fi-abc"
        api.upload_c2c_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_group_success(self):
        uploader, api = self._make()
        api.upload_group_file = mock.AsyncMock(return_value={"file_info": "fi-group"})
        file_info = await uploader.upload("group", "grp-1", "https://x.com/img.jpg", 1)
        assert file_info == "fi-group"

    @pytest.mark.asyncio
    async def test_invalid_chat_type_raises(self):
        uploader, api = self._make()
        with pytest.raises(ValueError, match="Unsupported chat_type"):
            await uploader.upload("guild", "ch-1", "https://x.com/img.jpg", 1)

    @pytest.mark.asyncio
    async def test_missing_file_info_raises(self):
        uploader, api = self._make()
        api.upload_c2c_file = mock.AsyncMock(return_value={})
        with pytest.raises(RuntimeError, match="no file_info"):
            await uploader.upload("c2c", "uid-1", "https://x.com/img.jpg", 1)

    @pytest.mark.asyncio
    async def test_fatal_error_no_retry(self):
        uploader, api = self._make()
        api.upload_c2c_file = mock.AsyncMock(side_effect=RuntimeError("400 Bad Request"))
        with pytest.raises(RuntimeError):
            await uploader.upload("c2c", "uid-1", "https://x.com/img.jpg", 1)
        # Should not retry on fatal errors
        assert api.upload_c2c_file.call_count == 1
