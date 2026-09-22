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

    def test_opencode_go_merge_does_not_resurrect_delisted_model(self):
        """#95914 bug class, end-to-end through provider_model_ids with the REAL curated floor:
        the Go relay (GET /zen/go/v1/models) delisted ox-alpha-free 2026-09-09. The live-first
        merge must not resurrect it from the curated floor, or the picker keeps offering a model
        that now 401s (REVERT-PROOF: a stale floor re-adds it and this fails)."""
        assert "opencode-go" in _LIVE_FIRST_PICKER_PROVIDERS
        live = ["deepseek-v4-flash", "kimi-k3", "omen-alpha"]  # current Go relay (no ox-alpha-free)

        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("opencode-go")

        assert "ox-alpha-free" not in result
        assert {"deepseek-v4-flash", "kimi-k3", "omen-alpha"} <= set(result)

    def test_opencode_zen_merge_does_not_resurrect_retired_model(self):
        """#115496 bug class, end-to-end through provider_model_ids with the REAL curated floor:
        the Zen relay (GET /zen/v1/models) retired x-preview-f-free (the picker-facing id for Ox
        Alpha) 2026-09-19. The live-first merge must not resurrect it from the curated floor
        (models_catalog_static.py still lists it first), or the picker keeps offering a model that
        now 401s (REVERT-PROOF: a stale floor re-adds it and this fails)."""
        assert "opencode-zen" in _LIVE_FIRST_PICKER_PROVIDERS
        live = ["kimi-k3", "gpt-5.6-sol", "claude-opus-5"]  # current Zen relay (no x-preview-f-free)

        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("opencode-zen")

        assert "x-preview-f-free" not in result
        assert {"kimi-k3", "gpt-5.6-sol", "claude-opus-5"} <= set(result)

    def test_alibaba_token_plan_merge_does_not_resurrect_delisted_model(self):
        """#119481: the live Token Plan catalog is authoritative for retired ids.

        ``qwen3.8-max-0902`` was once curated, but the endpoint no longer
        serves it.  A live refresh must retain every currently listed model
        without merging that retired id back into the picker.
        """
        live = ["qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus"]

        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("alibaba-token-plan")

        assert "qwen3.8-max-0902" not in result
        assert set(live) <= set(result)


    def test_opencode_zen_offline_catalog_drops_retired_model(self):
        """#115496 without a key: no live fetch, so provider_model_ids serves the curated floor merged
        with models.dev — both still carry the retired x-preview-f-free. The final rows must not."""
        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(None)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "", "base_url": ""},
            ),
            patch("agent.models_dev.list_agentic_models", return_value=["x-preview-f-free", "kimi-k3"]),
        ):
            result = provider_model_ids("opencode-zen")

        assert "x-preview-f-free" not in result
        assert "kimi-k3" in result
