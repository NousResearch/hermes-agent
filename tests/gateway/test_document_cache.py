"""
Tests for document cache utilities in gateway/platforms/base.py.

Covers: get_document_cache_dir, cache_document_from_bytes,
        cleanup_document_cache, SUPPORTED_DOCUMENT_TYPES.
"""

import os
import time
from pathlib import Path

import pytest

from gateway.platforms.base import (
    SUPPORTED_DOCUMENT_TYPES,
    cache_document_from_bytes,
    cleanup_document_cache,
    get_document_cache_dir,
)

# ---------------------------------------------------------------------------
# Fixture: redirect DOCUMENT_CACHE_DIR to a temp directory for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point the module-level DOCUMENT_CACHE_DIR to a fresh tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio_cache"
    )


# ---------------------------------------------------------------------------
# TestGetDocumentCacheDir
# ---------------------------------------------------------------------------

class TestGetDocumentCacheDir:
    def test_creates_directory(self, tmp_path):
        cache_dir = get_document_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()


# ---------------------------------------------------------------------------
# TestCacheDocumentFromBytes
# ---------------------------------------------------------------------------

class TestCacheDocumentFromBytes:
    def test_basic_caching(self):
        data = b"hello world"
        path = cache_document_from_bytes(data, "test.txt")
        assert os.path.exists(path)
        assert Path(path).read_bytes() == data

    def test_filename_preserved_in_path(self):
        path = cache_document_from_bytes(b"data", "report.pdf")
        assert "report.pdf" in os.path.basename(path)

    def test_empty_filename_uses_fallback(self):
        path = cache_document_from_bytes(b"data", "")
        assert Path(path).read_bytes() == b"data"


# ---------------------------------------------------------------------------
# TestCleanupDocumentCache
# ---------------------------------------------------------------------------

class TestCleanupDocumentCache:
    def test_removes_old_files(self, tmp_path):
        cache_dir = get_document_cache_dir()
        old_file = cache_dir / "old.txt"
        old_file.write_text("old")
        # Set modification time to 48 hours ago
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        removed = cleanup_document_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()


# ---------------------------------------------------------------------------
# TestSupportedDocumentTypes
# ---------------------------------------------------------------------------

class TestSupportedDocumentTypes:
    def test_all_extensions_have_mime_types(self):
        for ext, mime in SUPPORTED_DOCUMENT_TYPES.items():
            assert ext.startswith("."), f"{ext} missing leading dot"
            assert "/" in mime, f"{mime} is not a valid MIME type"


# ---------------------------------------------------------------------------
# TestCacheMediaBytes — the unified, platform-agnostic caching primitive
# ---------------------------------------------------------------------------

# 1x1 transparent PNG (passes cache_image_from_bytes validation)
_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6360000002000154a24f5f0000000049454e44ae426082"
)


class TestCacheMediaBytes:
    def test_pdf_routes_to_document(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(b"%PDF-1.4 body", filename="report.pdf", mime_type="application/pdf")
        assert result is not None
        assert result.kind == "document"
        assert result.media_type == "application/pdf"
        assert "report.pdf" in result.display_name
        assert os.path.exists(result.path)
        assert "report.pdf" in result.context_note()

    def test_png_routes_to_image(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(_PNG_1PX, filename="photo.png", mime_type="image/png")
        assert result is not None
        assert result.kind == "image"
        assert result.media_type == "image/png"
        assert os.path.exists(result.path)


    def test_unknown_document_cached_as_octet_stream(self):
        """Unknown file types are cached (not dropped) so the agent can inspect them.

        Authorization to message the agent is the gate, not the file extension.
        """
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(b"MZ", filename="program.exe", mime_type="application/x-msdownload")
        assert result is not None
        assert result.kind == "document"
        # Caller-supplied MIME is preserved when present.
        assert result.media_type == "application/x-msdownload"
        assert os.path.exists(result.path)




# ---------------------------------------------------------------------------
# Filename-length hardening: chat platforms can deliver display names that,
# after the doc_ uuid prefix, exceed the 255-byte filesystem component limit
# (ENAMETOOLONG dropped the whole inbound file message upstream).
# ---------------------------------------------------------------------------

class TestDocumentNameLength:
    def test_oversized_cjk_name_is_truncated_and_saved(self):
        name = "名单" * 130 + ".doc"  # 520 bytes + ext
        path = cache_document_from_bytes(b"x", name)
        comp = os.path.basename(path)
        assert len(comp.encode("utf-8")) <= 255
        assert comp.endswith(".doc")
        os.unlink(path)

    def test_percent_encoded_display_name_survives(self):
        # WeCom-style: URL-encoded CJK triples byte length.
        name = ("4-2.%E4%BB%A3%E8%A1%A8" * 24) + ".doc"
        assert len(name.encode()) > 237
        path = cache_document_from_bytes(b"x", name)
        comp = os.path.basename(path)
        assert len(comp.encode("utf-8")) <= 255 and comp.endswith(".doc")
        os.unlink(path)

    def test_truncation_never_splits_a_codepoint(self):
        name = "文" * 200 + ".pdf"
        path = cache_document_from_bytes(b"x", name)
        stem = os.path.basename(path).split("_", 2)[2].rsplit(".", 1)[0]
        stem.encode("utf-8").decode("utf-8")  # raises on a half codepoint
        assert path.endswith(".pdf")
        os.unlink(path)

    def test_long_name_without_extension_is_saved(self):
        name = "a" * 300
        path = cache_document_from_bytes(b"x", name)
        assert os.path.basename(path).count(".") == 0
        os.unlink(path)

    def test_normal_names_are_untouched(self):
        path = cache_document_from_bytes(b"x", "季度报表 2026.docx")
        assert os.path.basename(path).endswith("季度报表 2026.docx")
        os.unlink(path)
