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


# ---------------------------------------------------------------------------
# TestGetDocumentCacheDir
# ---------------------------------------------------------------------------

class TestGetDocumentCacheDir:
    def test_creates_directory(self, tmp_path):
        cache_dir = get_document_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_returns_existing_directory(self):
        first = get_document_cache_dir()
        second = get_document_cache_dir()
        assert first == second
        assert first.exists()


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
        assert "document" in os.path.basename(path)

    def test_unique_filenames(self):
        p1 = cache_document_from_bytes(b"a", "same.txt")
        p2 = cache_document_from_bytes(b"b", "same.txt")
        assert p1 != p2

    def test_path_traversal_blocked(self):
        """Malicious directory components are stripped — only the leaf name survives."""
        path = cache_document_from_bytes(b"data", "../../etc/passwd")
        basename = os.path.basename(path)
        assert "passwd" in basename
        # Must NOT contain directory separators
        assert ".." not in basename
        # File must reside inside the cache directory
        cache_dir = get_document_cache_dir()
        assert Path(path).resolve().is_relative_to(cache_dir.resolve())

    def test_null_bytes_stripped(self):
        path = cache_document_from_bytes(b"data", "file\x00.pdf")
        basename = os.path.basename(path)
        assert "\x00" not in basename
        assert "file.pdf" in basename

    def test_dot_dot_filename_handled(self):
        """A filename that is literally '..' falls back to 'document'."""
        path = cache_document_from_bytes(b"data", "..")
        basename = os.path.basename(path)
        assert "document" in basename

    def test_none_filename_uses_fallback(self):
        path = cache_document_from_bytes(b"data", None)
        assert "document" in os.path.basename(path)


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

    def test_keeps_recent_files(self):
        cache_dir = get_document_cache_dir()
        recent = cache_dir / "recent.txt"
        recent.write_text("fresh")

        removed = cleanup_document_cache(max_age_hours=24)
        assert removed == 0
        assert recent.exists()

    def test_returns_removed_count(self):
        cache_dir = get_document_cache_dir()
        old_time = time.time() - 48 * 3600
        for i in range(3):
            f = cache_dir / f"old_{i}.txt"
            f.write_text("x")
            os.utime(f, (old_time, old_time))

        assert cleanup_document_cache(max_age_hours=24) == 3

    def test_empty_cache_dir(self):
        assert cleanup_document_cache(max_age_hours=24) == 0


# ---------------------------------------------------------------------------
# TestSupportedDocumentTypes
# ---------------------------------------------------------------------------

class TestSupportedDocumentTypes:
    def test_all_extensions_have_mime_types(self):
        for ext, mime in SUPPORTED_DOCUMENT_TYPES.items():
            assert ext.startswith("."), f"{ext} missing leading dot"
            assert "/" in mime, f"{mime} is not a valid MIME type"

    @pytest.mark.parametrize(
        "ext",
        [".pdf", ".md", ".txt", ".docx", ".xlsx", ".pptx"],
    )
    def test_expected_extensions_present(self, ext):
        assert ext in SUPPORTED_DOCUMENT_TYPES
