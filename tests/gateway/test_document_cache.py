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
    resolve_document_types,
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
        [
            ".pdf",
            ".md",
            ".txt",
            ".zip",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
        ],
    )
    def test_expected_extensions_present(self, ext):
        assert ext in SUPPORTED_DOCUMENT_TYPES


class TestResolveDocumentTypes:
    def test_defaults_match_builtin_allowlist(self):
        assert resolve_document_types() == SUPPORTED_DOCUMENT_TYPES

    def test_global_config_can_add_and_remove_extensions(self):
        effective = resolve_document_types({
            "_global_document_types": {
                "add": {".epub": "application/epub+zip", "foo": "text/plain"},
                "remove": [".doc"],
            }
        })
        assert effective[".epub"] == "application/epub+zip"
        assert effective[".foo"] == "text/plain"
        assert ".doc" not in effective

    def test_platform_config_applies_after_global_config(self):
        effective = resolve_document_types({
            "_global_document_types": {
                "add": {".foo": "application/x-global"},
                "remove": [".pdf"],
            },
            "document_types": {
                "add": {".foo": "application/x-platform", ".pdf": "application/pdf"},
                "remove": [".doc"],
            },
        })
        assert effective[".foo"] == "application/x-platform"
        assert effective[".pdf"] == "application/pdf"
        assert ".doc" not in effective

    def test_invalid_extension_entries_are_ignored(self):
        effective = resolve_document_types({"document_types": {"add": {"bad/name": "text/plain", ".ok": ""}}})
        assert "bad/name" not in effective
        assert effective[".ok"] == "application/octet-stream"


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

    def test_native_photo_without_filename_uses_default_kind(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(_PNG_1PX, filename="", mime_type="", default_kind="image")
        assert result is not None
        assert result.kind == "image"

    def test_mp4_routes_to_video(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(b"\x00\x00\x00\x18ftypmp42", filename="clip.mp4", mime_type="video/mp4")
        assert result is not None
        assert result.kind == "video"
        assert result.media_type == "video/mp4"

    def test_mime_only_resolves_extension(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(b"col1,col2\n1,2", filename="", mime_type="text/csv")
        assert result is not None
        assert result.kind == "document"
        assert result.media_type == "text/csv"

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

    def test_unknown_document_no_mime_falls_back_to_octet_stream(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(b"\x00\x01\x02", filename="mystery.qux", mime_type="")
        assert result is not None
        assert result.kind == "document"
        assert result.media_type == "application/octet-stream"

    def test_custom_document_type_routes_to_document(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(
            b"PK\x03\x04 fake epub",
            filename="book.epub",
            mime_type="application/epub+zip",
            document_type_config={"document_types": {"add": {".epub": "application/epub+zip"}}},
        )
        assert result is not None
        assert result.kind == "document"
        assert result.media_type == "application/epub+zip"

    def test_removed_document_type_returns_none(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(
            b"%PDF-1.4 body",
            filename="report.pdf",
            mime_type="application/pdf",
            document_type_config={"document_types": {"remove": [".pdf"]}},
        )
        assert result is None

    def test_invalid_image_returns_none(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(b"<html>not an image</html>", filename="x.png", mime_type="image/png")
        assert result is None
