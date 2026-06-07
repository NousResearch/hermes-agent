"""Tests for gateway.accepted_document_extensions config integration."""

from unittest.mock import patch

import pytest


class TestGetAcceptedDocumentTypes:
    """Tests for get_accepted_document_types() in gateway.platforms.base."""

    def test_returns_default_set_when_no_config(self):
        """Without user config, returns the built-in SUPPORTED_DOCUMENT_TYPES."""
        from gateway.platforms.base import get_accepted_document_types, SUPPORTED_DOCUMENT_TYPES

        with patch("hermes_cli.config.load_config", return_value={}):
            result = get_accepted_document_types()

        for ext, mime in SUPPORTED_DOCUMENT_TYPES.items():
            assert ext in result
            assert result[ext] == mime

    def test_merges_extra_extensions(self):
        """User-configured extensions are merged into the default set."""
        from gateway.platforms.base import SUPPORTED_DOCUMENT_TYPES

        config = {
            "gateway": {
                "accepted_document_extensions": [".rs", ".go", ".java"],
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.platforms.base import get_accepted_document_types
            result = get_accepted_document_types()

        # Defaults still present
        assert ".pdf" in result
        assert ".py" in result
        # New extensions added
        assert ".rs" in result
        assert ".go" in result
        assert ".java" in result
        # MIME types auto-detected (system-dependent, just verify not empty)
        assert result[".rs"]  # some MIME detected
        assert result[".java"]  # some MIME detected

    def test_does_not_override_default_extensions(self):
        """User extensions don't override built-in MIME mappings."""
        config = {
            "gateway": {
                "accepted_document_extensions": [".pdf"],  # already in defaults
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.platforms.base import get_accepted_document_types
            result = get_accepted_document_types()

        # Original MIME preserved, not overwritten
        assert result[".pdf"] == "application/pdf"

    def test_unknown_extension_gets_octet_stream(self):
        """Unknown extensions default to application/octet-stream."""
        config = {
            "gateway": {
                "accepted_document_extensions": [".xyzzy123"],
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.platforms.base import get_accepted_document_types
            result = get_accepted_document_types()

        assert ".xyzzy123" in result
        assert result[".xyzzy123"] == "application/octet-stream"

    def test_ignores_non_list_config(self):
        """Non-list config values are silently ignored."""
        config = {"gateway": {"accepted_document_extensions": "not a list"}}
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.platforms.base import get_accepted_document_types
            result = get_accepted_document_types()

        # Should still have defaults
        assert ".pdf" in result

    def test_ignores_extensions_without_dot_prefix(self):
        """Extensions without leading dot are skipped."""
        config = {
            "gateway": {
                "accepted_document_extensions": ["rs", ".go"],  # "rs" has no dot
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.platforms.base import get_accepted_document_types
            result = get_accepted_document_types()

        assert "rs" not in result
        assert ".go" in result

    def test_load_config_failure_returns_defaults(self):
        """If load_config raises, returns the default set."""
        with patch("hermes_cli.config.load_config", side_effect=Exception("boom")):
            from gateway.platforms.base import get_accepted_document_types
            result = get_accepted_document_types()

        from gateway.platforms.base import SUPPORTED_DOCUMENT_TYPES
        for ext in SUPPORTED_DOCUMENT_TYPES:
            assert ext in result

    def test_cache_media_bytes_accepts_extra_extension(self):
        """cache_media_bytes accepts a user-configured extension."""
        import os
        import tempfile

        from gateway.platforms.base import cache_media_bytes

        config = {
            "gateway": {
                "accepted_document_extensions": [".rs"],
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            result = cache_media_bytes(
                b"fn main() {}",
                filename="hello.rs",
                mime_type="text/x-rust",
            )

        assert result is not None
        assert result.kind == "document"
        assert result.path.endswith(".rs") or "document" in result.path

    def test_cache_media_bytes_rejects_unknown_extension(self):
        """cache_media_bytes rejects an extension not in defaults or config."""
        from gateway.platforms.base import cache_media_bytes

        with patch("hermes_cli.config.load_config", return_value={}):
            result = cache_media_bytes(
                b"some data",
                filename="file.xyzzy123",
                mime_type="application/octet-stream",
            )

        assert result is None
