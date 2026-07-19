"""
Tests for API server file upload support (API_UPLOAD_FILES_URL).

Covers:
- Config properties (extra dict, env vars)
- _process_response_files — fallback when no upload URL
- _process_response_files — MEDIA: tag extraction + in-place replacement
- _process_response_files — bare path extraction
- _process_response_files — download URL template substitution
- _upload_file_to_server — successful upload
- _upload_file_to_server — error resilience
- Annotations — url_citation vs file_citation
- Annotations — start_index / end_index correctness
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("aiohttp")

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(**extra):
    """Build an APIServerAdapter with the given extra config and a dummy key."""
    cfg = PlatformConfig(
        enabled=True,
        extra={"key": "sk-test", **extra},
    )
    return APIServerAdapter(cfg)


def _write_temp_file(suffix: str = ".pdf", content: bytes = b"test content"):
    """Create a real temp file the adapter can stat/read."""
    d = Path(tempfile.mkdtemp(prefix="hermes_upload_test"))
    p = d / f"test{suffix}"
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestUploadConfig:
    def test_defaults_empty(self):
        adapter = _make_adapter()
        assert adapter._upload_files_url == ""
        assert adapter._upload_files_key == ""
        assert adapter._upload_files_download_url == ""

    def test_from_extra(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
            upload_files_key="sk-upload-key",
            upload_files_download_url="https://api.example.com/v1/files/{file_id}/content",
        )
        assert adapter._upload_files_url == "https://api.example.com/v1/files"
        assert adapter._upload_files_key == "sk-upload-key"
        assert (
            adapter._upload_files_download_url
            == "https://api.example.com/v1/files/{file_id}/content"
        )

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("API_UPLOAD_FILES_URL", "https://env.example.com/v1/files")
        monkeypatch.setenv("API_UPLOAD_FILES_KEY", "sk-env-key")
        monkeypatch.setenv(
            "API_UPLOAD_FILES_DOWNLOAD_URL",
            "https://env.example.com/files/{file_id}/content",
        )
        adapter = _make_adapter()
        assert adapter._upload_files_url == "https://env.example.com/v1/files"
        assert adapter._upload_files_key == "sk-env-key"
        assert (
            adapter._upload_files_download_url
            == "https://env.example.com/files/{file_id}/content"
        )

    def test_extra_overrides_env(self, monkeypatch):
        monkeypatch.setenv("API_UPLOAD_FILES_URL", "https://env.example.com/v1/files")
        adapter = _make_adapter(
            upload_files_url="https://extra.example.com/v1/files",
        )
        assert adapter._upload_files_url == "https://extra.example.com/v1/files"


# ---------------------------------------------------------------------------
# _process_response_files — fallback (no upload URL)
# ---------------------------------------------------------------------------


class TestProcessResponseFilesFallback:
    def test_empty_text_returns_empty(self):
        adapter = _make_adapter()
        text, items = asyncio.run(adapter._process_response_files(""))
        assert text == ""
        assert items == []

    def test_no_upload_url_strips_media_tags(self):
        """Without API_UPLOAD_FILES_URL, MEDIA: tags are stripped and
        images are inlined if possible; non-images are left as bare
        text for reference."""
        adapter = _make_adapter()
        png = _write_temp_file(".png")
        pdf = _write_temp_file(".pdf")
        text = f"Before\nMEDIA:{png}\nMEDIA:{pdf}\nAfter"
        cleaned, items = asyncio.run(adapter._process_response_files(text))
        assert items == []
        assert "MEDIA:" not in cleaned
        assert "Before" in cleaned
        assert "After" in cleaned

    def test_returns_empty_items(self):
        adapter = _make_adapter()
        _, items = asyncio.run(
            adapter._process_response_files("MEDIA:/tmp/foo.pdf")
        )
        assert items == []


# ---------------------------------------------------------------------------
# _process_response_files — upload path
# ---------------------------------------------------------------------------


class TestProcessResponseFilesUpload:
    def test_media_tag_replaced_in_place(self):
        """MEDIA: tag is replaced at its original position, not appended."""
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
            upload_files_download_url="https://api.example.com/files/{file_id}/content",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {
                "id": "file_abc123",
                "filename": pdf.name,
                "bytes": 1234,
                "created_at": 1750000000,
                "purpose": "user_data",
            }

        adapter._upload_file_to_server = _fake_upload
        text = f"Report ready.\nMEDIA:{pdf}\nKey findings: ..."
        cleaned, items = asyncio.run(adapter._process_response_files(text))

        # Tag replaced in-place — text flows naturally.
        assert "MEDIA:" not in cleaned
        assert f"[{pdf.name}]" in cleaned
        assert "Key findings:" in cleaned
        assert cleaned.startswith("Report ready.")
        # Not appended at the very end.
        assert not cleaned.endswith(f"[{pdf.name}]")
        assert len(items) == 1
        assert items[0]["id"] == "file_abc123"
        assert items[0]["object"] == "file"
        assert items[0]["filename"] == pdf.name

    def test_bare_path_replaced(self):
        """Bare absolute paths without MEDIA: prefix are also detected
        and replaced."""
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {
                "id": "file_xyz",
                "filename": pdf.name,
                "bytes": 5678,
                "created_at": 1750000001,
                "purpose": "user_data",
            }

        adapter._upload_file_to_server = _fake_upload
        text = f"Saved to {pdf}"
        cleaned, items = asyncio.run(adapter._process_response_files(text))

        assert str(pdf) not in cleaned
        assert f"[{pdf.name}]" in cleaned
        assert len(items) == 1
        assert items[0]["id"] == "file_xyz"

    def test_download_url_template(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
            upload_files_download_url="https://cdn.example.com/dl/{file_id}",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {"id": "file_abc123", "filename": pdf.name}

        adapter._upload_file_to_server = _fake_upload
        text = f"MEDIA:{pdf}"
        cleaned, items = asyncio.run(adapter._process_response_files(text))

        url = items[0]["download_url"]
        assert url == "https://cdn.example.com/dl/file_abc123"
        assert url in cleaned

    def test_no_download_url_template_uses_file_scheme(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {"id": "file_abc123", "filename": pdf.name}

        adapter._upload_file_to_server = _fake_upload
        _, items = asyncio.run(
            adapter._process_response_files(f"MEDIA:{pdf}")
        )
        assert items[0]["download_url"] == "file:file_abc123"

    def test_upload_failure_skips_file(self):
        """When upload returns None the MEDIA: tag is still stripped."""
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return None  # simulate failure

        adapter._upload_file_to_server = _fake_upload
        text = f"MEDIA:{pdf}"
        cleaned, items = asyncio.run(adapter._process_response_files(text))
        assert items == []
        assert "MEDIA:" not in cleaned

    def test_multiple_files(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")
        png = _write_temp_file(".png")

        _counter = 0

        async def _fake_upload(file_path, purpose="user_data"):
            nonlocal _counter
            _counter += 1
            return {
                "id": f"file_{_counter:03d}",
                "filename": Path(file_path).name,
            }

        adapter._upload_file_to_server = _fake_upload
        text = f"MEDIA:{pdf}\nMEDIA:{png}"
        cleaned, items = asyncio.run(adapter._process_response_files(text))
        assert len(items) == 2
        assert "MEDIA:" not in cleaned

    def test_deduplicates_duplicate_paths(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")
        call_count = 0

        async def _fake_upload(file_path, purpose="user_data"):
            nonlocal call_count
            call_count += 1
            return {"id": "file_001", "filename": pdf.name}

        adapter._upload_file_to_server = _fake_upload
        text = f"MEDIA:{pdf}\nAlso: MEDIA:{pdf}"
        _, items = asyncio.run(adapter._process_response_files(text))
        # Uploaded once, one file item.
        assert call_count == 1
        assert len(items) == 1

    def test_backtick_wrapped_path_skipped(self):
        """Backtick-wrapped paths like `/tmp/file.pdf` are treated as
        inline code by extract_local_files and skipped.  The agent must
        use the MEDIA: prefix."""
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")
        call_count = 0

        async def _fake_upload(file_path, purpose="user_data"):
            nonlocal call_count
            call_count += 1
            return {"id": "file_001", "filename": pdf.name}

        adapter._upload_file_to_server = _fake_upload
        text = f"Saved `{pdf}` for you"
        _, items = asyncio.run(adapter._process_response_files(text))
        assert call_count == 0
        assert items == []

    def test_preserves_surrounding_text(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {"id": "file_abc", "filename": pdf.name}

        adapter._upload_file_to_server = _fake_upload
        text = f"## Results\n\nMEDIA:{pdf}\n\nSee above for details."
        cleaned, _ = asyncio.run(adapter._process_response_files(text))
        assert "## Results" in cleaned
        assert "See above for details." in cleaned
        assert "MEDIA:" not in cleaned


# ---------------------------------------------------------------------------
# _upload_file_to_server
# ---------------------------------------------------------------------------


class TestUploadFileToServer:
    def test_returns_none_when_url_not_configured(self):
        adapter = _make_adapter()
        result = asyncio.run(
            adapter._upload_file_to_server("/tmp/foo.pdf")
        )
        assert result is None

    def test_returns_none_for_missing_file(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        result = asyncio.run(
            adapter._upload_file_to_server("/nonexistent/path/file.pdf")
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_upload(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
            upload_files_key="sk-upload",
        )
        pdf = _write_temp_file(".pdf")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={
                "id": "file_abc123",
                "object": "file",
                "bytes": 1234,
                "created_at": 1750000000,
                "filename": pdf.name,
                "purpose": "user_data",
                "status": "uploaded",
            }
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.FormData", return_value=MagicMock()):
                result = await adapter._upload_file_to_server(str(pdf))
        assert result is not None
        assert result["id"] == "file_abc123"
        assert result["filename"] == pdf.name
        assert result["bytes"] == 1234

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Internal Server Error")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.FormData", return_value=MagicMock()):
                result = await adapter._upload_file_to_server(str(pdf))
        assert result is None

    @pytest.mark.asyncio
    async def test_network_error_returns_none(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=Exception("Connection refused")
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.FormData", return_value=MagicMock()):
                result = await adapter._upload_file_to_server(str(pdf))
        assert result is None

    @pytest.mark.asyncio
    async def test_authorization_header_sent(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
            upload_files_key="sk-upload-key",
        )
        pdf = _write_temp_file(".pdf")

        mock_formdata = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"id": "file_abc", "filename": pdf.name}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.FormData", return_value=mock_formdata):
                await adapter._upload_file_to_server(str(pdf))

        call_kwargs = mock_session.post.call_args
        headers = call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-upload-key"

    @pytest.mark.asyncio
    async def test_form_fields_set_correctly(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf", content=b"hello world")

        mock_formdata = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"id": "file_abc", "filename": pdf.name}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.FormData", return_value=mock_formdata):
                await adapter._upload_file_to_server(str(pdf))

        # Verify form.add_field calls
        add_field_calls = [
            c[0] for c in mock_formdata.add_field.call_args_list
        ]
        assert ("purpose", "user_data") in add_field_calls
        # file call: (field_name, bytes, filename=..., content_type=...)
        file_calls = [c for c in add_field_calls if c[0] == "file"]
        assert len(file_calls) == 1
        file_args = file_calls[0]
        assert file_args[1] == b"hello world"
        assert mock_formdata.add_field.call_args_list[-1].kwargs.get(
            "filename"
        ) == pdf.name


# ---------------------------------------------------------------------------
# Annotations (url_citation vs file_citation)
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_url_citation_when_download_url_configured(self):
        """When API_UPLOAD_FILES_DOWNLOAD_URL is set, annotations use
        url_citation with start_index and end_index pointing into the
        cleaned text."""
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
            upload_files_download_url="https://cdn.example.com/{file_id}/content",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {
                "id": "file_abc123",
                "filename": pdf.name,
            }

        adapter._upload_file_to_server = _fake_upload
        cleaned, items = asyncio.run(
            adapter._process_response_files(f"MEDIA:{pdf}")
        )

        # Build annotations the same way _handle_responses does.
        download_url = items[0]["download_url"]
        start = cleaned.find(download_url)
        assert start > 0, "download URL must appear in cleaned text"
        assert start + len(download_url) == len(cleaned) - 1  # ends with )

    def test_file_citation_when_no_download_url(self):
        """Without API_UPLOAD_FILES_DOWNLOAD_URL, annotations use
        file_citation with file_id and filename."""
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {
                "id": "file_abc123",
                "filename": pdf.name,
            }

        adapter._upload_file_to_server = _fake_upload
        _, items = asyncio.run(
            adapter._process_response_files(f"MEDIA:{pdf}")
        )
        assert items[0]["download_url"] == "file:file_abc123"

    def test_annotations_have_required_fields(self):
        """Every file item has the OpenAI FileObject required fields."""
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {
                "id": "file_abc123",
                "filename": pdf.name,
                "bytes": 1234,
                "created_at": 1750000000,
                "purpose": "user_data",
            }

        adapter._upload_file_to_server = _fake_upload
        _, items = asyncio.run(
            adapter._process_response_files(f"MEDIA:{pdf}")
        )
        obj = items[0]
        assert "id" in obj
        assert "object" in obj
        assert obj["object"] == "file"
        assert "bytes" in obj
        assert "created_at" in obj
        assert "filename" in obj
        assert "purpose" in obj
        assert "download_url" in obj

    def test_optional_fields_passed_through(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        pdf = _write_temp_file(".pdf")

        async def _fake_upload(file_path, purpose="user_data"):
            return {
                "id": "file_xyz",
                "filename": pdf.name,
                "status": "processed",
                "expires_at": 1760000000,
                "status_details": "ok",
            }
            adapter._upload_file_to_server = _fake_upload  # noqa — unreachable

        adapter._upload_file_to_server = _fake_upload
        _, items = asyncio.run(
            adapter._process_response_files(f"MEDIA:{pdf}")
        )
        obj = items[0]
        assert obj.get("status") == "processed"
        assert obj.get("expires_at") == 1760000000
        assert obj.get("status_details") == "ok"


# ---------------------------------------------------------------------------
# Unicode / non-ASCII paths
# ---------------------------------------------------------------------------


class TestUnicodePaths:
    def test_unicode_filename(self):
        adapter = _make_adapter(
            upload_files_url="https://api.example.com/v1/files",
        )
        d = Path(tempfile.mkdtemp(prefix="hermes_upload_test"))
        p = d / "PPT模板.pptx"
        p.write_bytes(b"unicode test")

        async def _fake_upload(file_path, purpose="user_data"):
            return {"id": "file_unicode", "filename": p.name}

        adapter._upload_file_to_server = _fake_upload
        text = f"MEDIA:{p}"
        cleaned, items = asyncio.run(adapter._process_response_files(text))
        assert len(items) == 1
        assert items[0]["filename"] == "PPT模板.pptx"
        assert "MEDIA:" not in cleaned
        assert "PPT模板.pptx" in cleaned
