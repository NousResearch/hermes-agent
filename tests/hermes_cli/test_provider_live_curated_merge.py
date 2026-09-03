"""Tests for live+curated merge in the generic profile-based provider path.

Guards two contracts:

* #46850 — when a provider's live /v1/models endpoint returns a stale or
  incomplete list, the static curated models from ``_PROVIDER_MODELS`` must
  still appear in the merged result (nothing is dropped).
* #46309 / #49129 — merge *order* is per-provider. Single providers
  (kimi, zai) stay **curated-first** so a deliberately surfaced newest model
  leads even when the live API lags. ``_LIVE_FIRST_PICKER_PROVIDERS``
  (OpenCode Zen / Go) flip to **live-first** because their live API is the
  authoritative catalog and stale curated entries must not lead the picker.
"""

from unittest.mock import MagicMock, patch

from hermes_cli.models import (
    _LIVE_FIRST_PICKER_PROVIDERS,
    provider_model_ids,
)


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

    def test_curated_first_for_single_provider(self):
        """Single providers (zai) stay curated-first; live-only appended."""
        assert "zai" not in _LIVE_FIRST_PICKER_PROVIDERS
        curated = ["glm-5.2", "glm-5.1", "glm-5"]  # authoritative-intent order
        # Live API lags AND surfaces a brand-new model not yet curated.
        live = ["glm-5", "glm-6-preview"]
        profile = self._make_profile(live)

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"zai": curated}),
        ):
            result = provider_model_ids("zai")

        # Curated entries lead (commit 658ac1d86, #46309).
        assert result[: len(curated)] == curated
        # Live-only entries (glm-6-preview) still surface, appended afterwards.
        assert "glm-6-preview" in result
        assert result.index("glm-6-preview") >= len(curated)
        # No duplicates for models present in both.
        assert result.count("glm-5") == 1


    def test_live_unavailable_falls_back_to_static_catalog_plus_profile_models(self):
        """Live fetch fails/empty → static catalog must NOT be shadowed.

        Regression: the no-live path used to return ONLY the profile's
        ``fallback_models`` tuple. For zai that tuple lagged the static
        catalog (no glm-5.3), so picker users on Z.AI's Anthropic wire —
        whose /models probe returns 200 + an error body that parses as an
        EMPTY catalog — silently lost every curated-only model. Static
        entries must lead, profile-only entries append.
        """
        profile = self._make_profile(models=[])  # live probe "succeeds" empty
        profile.fallback_models = ("glm-5.2", "glm-5", "glm-4-9b")

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict(
                "hermes_cli.models._PROVIDER_MODELS",
                {"zai": ["glm-5.3", "glm-5.2", "glm-5.1"]},
            ),
        ):
            result = provider_model_ids("zai")

        assert result[:3] == ["glm-5.3", "glm-5.2", "glm-5.1"]
        assert "glm-5" in result and "glm-4-9b" in result  # profile-only appended
        assert result.count("glm-5.2") == 1  # dedup across sources

    def test_no_models_dropped_either_direction(self):
        """Every live AND curated model survives the merge for both modes."""
        live = ["a", "b"]
        # zai = curated-first
        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"zai": ["c", "b"]}),
        ):
            zai_result = set(provider_model_ids("zai"))
        assert {"a", "b", "c"} <= zai_result

        # opencode-zen = live-first
        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"opencode-zen": ["c", "b"]}),
        ):
            zen_result = set(provider_model_ids("opencode-zen"))
        assert {"a", "b", "c"} <= zen_result

