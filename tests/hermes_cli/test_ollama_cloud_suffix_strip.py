"""Regression tests for ollama-cloud :cloud/-cloud suffix stripping.

models.dev appends ``:cloud`` / ``-cloud`` suffixes to Ollama Cloud model IDs
(e.g. ``deepseek-v4-flash:cloud``).  Both ``normalize_model_for_provider`` and
``validate_requested_model`` must strip these so bare IDs match the live API
catalog.  See issue #45137.
"""

import pytest


# ---------------------------------------------------------------------------
# normalize_model_for_provider
# ---------------------------------------------------------------------------

class TestOllamaCloudSuffixStripping:
    """normalize_model_for_provider strips :cloud/-cloud for ollama-cloud."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        self.normalize = normalize_model_for_provider

    def test_colon_cloud_suffix_stripped(self):
        assert self.normalize("deepseek-v4-flash:cloud", "ollama-cloud") == "deepseek-v4-flash"

    def test_hyphen_cloud_suffix_stripped(self):
        assert self.normalize("deepseek-v4-flash-cloud", "ollama-cloud") == "deepseek-v4-flash"

    def test_bare_id_unchanged(self):
        assert self.normalize("deepseek-v4-flash", "ollama-cloud") == "deepseek-v4-flash"

    def test_provider_prefix_stripped_before_suffix(self):
        assert self.normalize("ollama-cloud/deepseek-v4-flash:cloud", "ollama-cloud") == "deepseek-v4-flash"

    def test_no_double_strip(self):
        """A bare name ending in -cloud that is NOT a suffix is preserved."""
        # "nextcloud" is a real app name; ensure we only strip the trailing
        # suffix, not arbitrary substrings.
        assert self.normalize("nextcloud", "ollama-cloud") == "nextcloud"

    def test_colon_cloud_suffix_model_with_hyphen_cloud_in_name(self):
        """Model 'something-cloud' with additional ':cloud' suffix."""
        assert self.normalize("something-cloud:cloud", "ollama-cloud") == "something-cloud"


# ---------------------------------------------------------------------------
# validate_requested_model
# ---------------------------------------------------------------------------

class TestOllamaCloudValidationSuffixStrip:
    """validate_requested_model strips :cloud/-cloud for ollama-cloud lookups."""

    @pytest.fixture(autouse=True)
    def _mock_api(self, monkeypatch):
        """Mock fetch_api_models to return clean bare IDs."""
        from hermes_cli import models as _m
        self._models = _m

        def _fake_fetch(api_key=None, base_url=None, **_kw):
            return ["deepseek-v4-flash", "kimi-k2.6", "llama-4-maverick"]

        monkeypatch.setattr(_m, "fetch_api_models", _fake_fetch)
        # Also mock fetch_ollama_cloud_models to return the same list
        monkeypatch.setattr(_m, "fetch_ollama_cloud_models", lambda **_kw: ["deepseek-v4-flash", "kimi-k2.6", "llama-4-maverick"])

    def test_colon_cloud_suffix_accepted(self):
        result = self._models.validate_requested_model("deepseek-v4-flash:cloud", "ollama-cloud")
        assert result["accepted"] is True
        assert result["recognized"] is True

    def test_hyphen_cloud_suffix_accepted(self):
        result = self._models.validate_requested_model("deepseek-v4-flash-cloud", "ollama-cloud")
        assert result["accepted"] is True
        assert result["recognized"] is True

    def test_bare_id_accepted(self):
        result = self._models.validate_requested_model("deepseek-v4-flash", "ollama-cloud")
        assert result["accepted"] is True
        assert result["recognized"] is True
