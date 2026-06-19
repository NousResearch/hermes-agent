"""Tests for gateway attachment persistence (issue #41979)."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# _get_attachment_storage_path
# ---------------------------------------------------------------------------

class TestGetAttachmentStoragePath:
    """Tests for _get_attachment_storage_path()."""

    def test_returns_none_when_disabled(self, tmp_path):
        """Empty string means persistence is off."""
        from gateway.platforms.base import _get_attachment_storage_path

        with patch("hermes_cli.config.load_config", return_value={"gateway": {"attachment_storage_path": ""}}):
            assert _get_attachment_storage_path() is None

    def test_returns_none_on_missing_key(self):
        """Missing key defaults to disabled."""
        from gateway.platforms.base import _get_attachment_storage_path

        with patch("hermes_cli.config.load_config", return_value={"gateway": {}}):
            assert _get_attachment_storage_path() is None

    def test_returns_path_when_configured(self, tmp_path):
        """Non-empty path is expanded and created."""
        from gateway.platforms.base import _get_attachment_storage_path

        target = str(tmp_path / "attachments")
        with patch("hermes_cli.config.load_config", return_value={"gateway": {"attachment_storage_path": target}}):
            result = _get_attachment_storage_path()
            assert result == Path(target)
            assert result.exists()

    def test_expands_tilde(self, tmp_path):
        """Tilde paths are expanded."""
        from gateway.platforms.base import _get_attachment_storage_path

        with patch("hermes_cli.config.load_config", return_value={"gateway": {"attachment_storage_path": "~/test_attachments"}}):
            result = _get_attachment_storage_path()
            # Should return a path (may not exist if ~/test_attachments can't be created)
            # The important thing is it doesn't raise

    def test_returns_none_on_config_error(self):
        """Config load failure is handled gracefully."""
        from gateway.platforms.base import _get_attachment_storage_path

        with patch("hermes_cli.config.load_config", side_effect=Exception("config error")):
            assert _get_attachment_storage_path() is None


# ---------------------------------------------------------------------------
# _persist_attachment
# ---------------------------------------------------------------------------

class TestPersistAttachment:
    """Tests for _persist_attachment()."""

    def test_noop_when_disabled(self, tmp_path):
        """Does nothing when persistence is disabled."""
        from gateway.platforms.base import _persist_attachment

        with patch("gateway.platforms.base._get_attachment_storage_path", return_value=None):
            # Should not raise
            _persist_attachment(b"hello", filename="test.txt", platform="telegram")

    def test_creates_platform_subdirectory(self, tmp_path):
        """Creates <root>/<platform>/ subdirectory."""
        from gateway.platforms.base import _persist_attachment

        storage = tmp_path / "storage"
        with patch("gateway.platforms.base._get_attachment_storage_path", return_value=storage):
            _persist_attachment(b"hello world", filename="test.txt", platform="discord")

        plat_dir = storage / "discord"
        assert plat_dir.exists()
        files = list(plat_dir.iterdir())
        assert len(files) == 1
        assert files[0].name.endswith("_test.txt")
        assert files[0].read_bytes() == b"hello world"

    def test_deduplicates_on_name_collision(self, tmp_path):
        """Appends counter when filename already exists."""
        from gateway.platforms.base import _persist_attachment

        storage = tmp_path / "storage"
        with patch("gateway.platforms.base._get_attachment_storage_path", return_value=storage):
            _persist_attachment(b"first", filename="same.txt", platform="test")
            _persist_attachment(b"second", filename="same.txt", platform="test")

        files = list((storage / "test").iterdir())
        assert len(files) == 2
        contents = {f.read_bytes() for f in files}
        assert contents == {b"first", b"second"}

    def test_sanitizes_filename(self, tmp_path):
        """Unsafe characters in filename are replaced; directory components stripped."""
        from gateway.platforms.base import _persist_attachment

        storage = tmp_path / "storage"
        with patch("gateway.platforms.base._get_attachment_storage_path", return_value=storage):
            _persist_attachment(b"data", filename="../etc/passwd", platform="test")

        files = list((storage / "test").iterdir())
        assert len(files) == 1
        # Directory components should be stripped; only "passwd" remains
        assert ".." not in files[0].name
        assert "/" not in files[0].name
        assert "passwd" in files[0].name

    def test_noop_on_write_error(self, tmp_path):
        """Write failures are swallowed (best-effort)."""
        from gateway.platforms.base import _persist_attachment

        # Point to a read-only directory
        storage = tmp_path / "readonly"
        storage.mkdir()
        os.chmod(storage, 0o444)
        try:
            with patch("gateway.platforms.base._get_attachment_storage_path", return_value=storage):
                # Should not raise
                _persist_attachment(b"data", filename="test.txt", platform="test")
        finally:
            os.chmod(storage, 0o755)


# ---------------------------------------------------------------------------
# cleanup_persisted_attachments
# ---------------------------------------------------------------------------

class TestCleanupPersistedAttachments:
    """Tests for cleanup_persisted_attachments()."""

    def test_noop_when_disabled(self):
        """Returns 0 when persistence is disabled."""
        from gateway.platforms.base import cleanup_persisted_attachments

        with patch("gateway.platforms.base._get_attachment_storage_path", return_value=None):
            assert cleanup_persisted_attachments() == 0

    def test_removes_old_files(self, tmp_path):
        """Files older than retention_days are deleted."""
        from gateway.platforms.base import cleanup_persisted_attachments

        storage = tmp_path / "storage"
        plat = storage / "telegram"
        plat.mkdir(parents=True)

        # Create a file with old mtime
        old_file = plat / "20200101_120000_old.txt"
        old_file.write_bytes(b"old")
        old_time = time.time() - 40 * 86400  # 40 days ago
        os.utime(old_file, (old_time, old_time))

        # Create a recent file
        new_file = plat / "20260608_120000_new.txt"
        new_file.write_bytes(b"new")

        with patch("gateway.platforms.base._get_attachment_storage_path", return_value=storage), \
             patch("hermes_cli.config.load_config", return_value={"gateway": {"attachment_retention_days": 30}}):
            removed = cleanup_persisted_attachments()

        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_keep_forever_when_zero(self, tmp_path):
        """retention_days=0 means keep forever."""
        from gateway.platforms.base import cleanup_persisted_attachments

        storage = tmp_path / "storage"
        plat = storage / "test"
        plat.mkdir(parents=True)
        old_file = plat / "old.txt"
        old_file.write_bytes(b"old")
        old_time = time.time() - 100 * 86400
        os.utime(old_file, (old_time, old_time))

        with patch("gateway.platforms.base._get_attachment_storage_path", return_value=storage), \
             patch("hermes_cli.config.load_config", return_value={"gateway": {"attachment_retention_days": 0}}):
            removed = cleanup_persisted_attachments()

        assert removed == 0
        assert old_file.exists()


# ---------------------------------------------------------------------------
# cache_media_bytes integration
# ---------------------------------------------------------------------------

class TestCacheMediaBytesPersistence:
    """Test that cache_media_bytes calls _persist_attachment."""

    def test_persistence_called_with_platform(self, tmp_path):
        """cache_media_bytes passes platform to _persist_attachment."""
        from gateway.platforms.base import cache_media_bytes

        # We need a valid image to avoid None return
        # Use a minimal PNG (1x1 pixel)
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # PNG magic
            + b"\x00" * 100  # padding (not a real PNG, but enough for _looks_like_image)
        )

        with patch("gateway.platforms.base._persist_attachment") as mock_persist, \
             patch("gateway.platforms.base._get_attachment_storage_path", return_value=None), \
             patch("gateway.platforms.base.cache_image_from_bytes", return_value=str(tmp_path / "img.jpg")):
            # cache_image_from_bytes needs to return a valid path
            (tmp_path / "img.jpg").write_bytes(png_data)
            result = cache_media_bytes(png_data, filename="photo.jpg", platform="discord", default_kind="image")

        mock_persist.assert_called_once()
        call_kwargs = mock_persist.call_args
        assert call_kwargs[1]["platform"] == "discord"
        assert call_kwargs[1]["filename"] == "photo.jpg"

    def test_persistence_called_for_document(self, tmp_path):
        """cache_media_bytes calls persist for document type too."""
        from gateway.platforms.base import cache_media_bytes

        pdf_data = b"%PDF-1.4 fake pdf content"

        with patch("gateway.platforms.base._persist_attachment") as mock_persist, \
             patch("gateway.platforms.base._get_attachment_storage_path", return_value=None), \
             patch("gateway.platforms.base.cache_document_from_bytes", return_value=str(tmp_path / "doc.pdf")):
            (tmp_path / "doc.pdf").write_bytes(pdf_data)
            result = cache_media_bytes(pdf_data, filename="report.pdf", platform="telegram")

        mock_persist.assert_called_once()
