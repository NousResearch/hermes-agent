"""Tests for live+curated merge in the generic profile-based provider path.

Guards the fix for #46850: when a provider's live /v1/models endpoint
returns a stale or incomplete list, the static curated models from
``_PROVIDER_MODELS`` must still appear in the merged result.
"""

from unittest.mock import MagicMock, patch

from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids


class TestGenericProviderLiveCuratedMerge:
    """provider_model_ids merges live + curated for generic api_key providers."""

    def _make_profile(self, models=None):
        """Create a minimal mock provider profile."""
        p = MagicMock()
        p.auth_type = "api_key"
        p.base_url = "https://api.example.com/v1"
        p.fetch_models.return_value = models
        p.fallback_models = None
        return p

    def test_live_models_merged_with_curated(self):
        """Live models come first; curated-only models are appended."""
        live = ["glm-5.2", "glm-5.1", "glm-5"]
        curated = _PROVIDER_MODELS["zai"]  # includes glm-5.1, glm-5, glm-4.5, etc.
        profile = self._make_profile(live)

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={"api_key": "k", "base_url": ""}),
        ):
            result = provider_model_ids("zai")

        # Live entries first (in live order)
        assert result[0] == "glm-5.2"
        assert result[1] == "glm-5.1"
        assert result[2] == "glm-5"
        # Curated-only entries appended (e.g. glm-4.5)
        result_lower = [m.lower() for m in result]
        assert "glm-4.5" in result_lower
        assert "glm-4.5-flash" in result_lower

    def test_no_duplicate_models(self):
        """Models appearing in both live and curated are not duplicated."""
        live = ["glm-5.1", "glm-5"]
        curated = ["glm-5.1", "glm-5", "glm-4.5"]
        profile = self._make_profile(live)

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={"api_key": "k", "base_url": ""}),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"zai": curated}),
        ):
            result = provider_model_ids("zai")

        assert result.count("glm-5.1") == 1
        assert result.count("glm-5") == 1
        assert result == ["glm-5.1", "glm-5", "glm-4.5"]

    def test_case_insensitive_dedup(self):
        """Dedup is case-insensitive but preserves first occurrence casing."""
        live = ["GLM-5.1", "glm-5"]
        curated = ["glm-5.1", "GLM-5", "glm-4.5"]
        profile = self._make_profile(live)

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={"api_key": "k", "base_url": ""}),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"zai": curated}),
        ):
            result = provider_model_ids("zai")

        # Live casing preserved for duplicates
        assert result[0] == "GLM-5.1"
        assert result[1] == "glm-5"
        # Curated-only appended
        assert "glm-4.5" in result

    def test_empty_curated_returns_live_only(self):
        """When no curated list exists, live is returned as-is."""
        live = ["model-a", "model-b"]
        profile = self._make_profile(live)

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={"api_key": "k", "base_url": ""}),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"zai": []}),
        ):
            result = provider_model_ids("zai")

        assert result == ["model-a", "model-b"]

    def test_live_empty_falls_back_to_curated(self):
        """When live returns nothing, curated static list is used.

        ZAI is in _MODELS_DEV_PREFERRED so the fallback path merges with
        models.dev.  We mock _merge_with_models_dev to isolate the test.
        """
        curated = ["glm-5.1", "glm-5", "glm-4.5"]
        profile = self._make_profile([])

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={"api_key": "k", "base_url": ""}),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"zai": curated}),
            patch("hermes_cli.models._merge_with_models_dev", return_value=curated),
        ):
            result = provider_model_ids("zai")

        assert result == curated
