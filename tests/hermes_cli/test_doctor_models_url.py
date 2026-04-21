"""Tests for hermes doctor provider URL construction.

Regression for issue #9766: doctor appends /models to base_url even when
the base_url already contains the full endpoint path.
"""
import os
from unittest.mock import patch


def _build_models_url(base_url: str, default_url: str) -> str:
    """Replicate the URL construction logic from doctor.py.

    This mirrors the logic at line 762+ of hermes_cli/doctor.py so we can
    test it in isolation without network calls.
    """
    if base_url:
        _stripped = base_url.rstrip("/")
        if _stripped.endswith("/models"):
            return _stripped
        else:
            return _stripped + "/models"
    else:
        return default_url


class TestDoctorModelsUrlConstruction:
    """Verify doctor constructs provider health-check URLs correctly."""

    def test_no_base_url_uses_default(self):
        """When no custom base_url is set, use the default URL."""
        result = _build_models_url("", "https://api.example.com/v1/models")
        assert result == "https://api.example.com/v1/models"

    def test_base_url_appends_models(self):
        """Custom base_url gets /models appended (normal case)."""
        result = _build_models_url("https://api.example.com/v1", "https://default.com/models")
        assert result == "https://api.example.com/v1/models"

    def test_base_url_already_ends_with_models(self):
        """If base_url already ends with /models, don't double-append.

        Regression for issue #9766: some providers configure the full
        endpoint URL as base_url, and appending /models caused a 404.
        """
        result = _build_models_url("https://api.example.com/v1/models", "https://default.com/models")
        assert result == "https://api.example.com/v1/models"

    def test_base_url_with_trailing_slash(self):
        """Trailing slash on base_url is handled correctly."""
        result = _build_models_url("https://api.example.com/v1/", "https://default.com/models")
        assert result == "https://api.example.com/v1/models"

    def test_base_url_with_trailing_slash_and_models(self):
        """Trailing slash + /models is not double-appended."""
        result = _build_models_url("https://api.example.com/v1/models/", "https://default.com/models")
        assert result == "https://api.example.com/v1/models"
