"""Tests for live+curated merge in the generic profile-based provider path.

Guards three contracts:

* #46850 — when a provider's live /v1/models endpoint returns a stale or
  incomplete list, the static curated models from ``_PROVIDER_MODELS`` must
  still appear in the merged result (nothing is dropped).
* #46309 / #49129 — merge *order* is per-provider. Single providers
  (kimi, zai) stay **curated-first** so a deliberately surfaced newest model
  leads even when the live API lags. ``_LIVE_FIRST_PICKER_PROVIDERS``
  (OpenCode Zen / Go) flip to **live-first** because their live API is the
  authoritative catalog and stale curated entries must not lead the picker.
* #119481 — subscription-tier providers (``_LIVE_TERMINAL_PICKER_PROVIDERS``)
  skip the merge entirely on a successful probe: entitlements vary per plan,
  so curated-only ids are phantom rows that 404 every turn.
"""

from unittest.mock import MagicMock, patch

from hermes_cli.models import (
    _LIVE_FIRST_PICKER_PROVIDERS,
    _LIVE_TERMINAL_PICKER_PROVIDERS,
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
        the Go relay (GET /zen/go/v1/models) delisted ox-alpha-free 2026-09-09 but may keep LISTING
        it (#111749). Neither the live listing nor the curated floor may resurrect it, or the picker
        keeps offering a model that now 401s."""
        assert "opencode-go" in _LIVE_FIRST_PICKER_PROVIDERS
        live = ["deepseek-v4-flash", "kimi-k3", "omen-alpha", "ox-alpha-free"]

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


class TestLiveTerminalPickerProviders:
    """#119481: subscription-tier providers (``_LIVE_TERMINAL_PICKER_PROVIDERS``) have plan-dependent
    entitlements, so a SUCCESSFUL live /v1/models is the account's whole catalog. The curated-first
    union merge used to re-add retired ids the live endpoint no longer serves (qwen3.8-max-0902,
    whole Kimi rows on some tiers): selecting one 404s every turn, silently answered by
    fallback_providers. Terminal-on-success means the merge can no longer resurrect phantom rows
    (REVERT-PROOF: any curated-first re-add fails these)."""

    def _make_profile(self, models=None, fallback_models=None):
        p = MagicMock()
        p.auth_type = "api_key"
        p.base_url = "https://token-plan.example.com/compatible-mode/v1"
        p.fetch_models.return_value = models
        p.fallback_models = fallback_models
        return p

    def test_alibaba_token_plan_live_catalog_is_terminal(self):
        """Live tier catalog (intl sk-sp tier from #119481) replaces the curated list — no
        curated-only phantom rows (qwen3.8-max-0902, kimi-*) can re-enter."""
        assert "alibaba-token-plan" in _LIVE_TERMINAL_PICKER_PROVIDERS
        live = [
            "qwen3.6-flash", "qwen3.7-max", "qwen3.7-plus", "qwen3.8-max", "qwen3.8-flash",
            "deepseek-v4-pro", "glm-5.2", "glm-5.3",
        ]

        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("alibaba-token-plan")

        # Exactly the live catalog: every served id present…
        assert set(result) >= set(live)
        # …and not one curated-only phantom row survives (#119481's 404 trap).
        for phantom in ("qwen3.8-max-0902", "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5", "glm-5.1", "glm-5"):
            assert phantom not in result

    def test_alibaba_token_plan_cn_twin_is_terminal(self):
        """The -cn twin shares the tier-dependent endpoint shape; same terminal rule."""
        assert "alibaba-token-plan-cn" in _LIVE_TERMINAL_PICKER_PROVIDERS
        live = ["qwen3.8-max", "qwen3.8-flash"]

        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("alibaba-token-plan-cn")

        assert set(result) == set(live)

    def test_tencent_tokenplan_live_catalog_is_terminal(self):
        """The third tier-dependent plan named in #119481 gets the same treatment — through the
        REAL registered profile (plugins/model-providers/tencent/), not a mocked
        get_provider_profile: production used to have no profile for this name, which made
        the terminal arm dead code for tencent (review on #126898)."""
        from providers import get_provider_profile

        assert "tencent-tokenplan" in _LIVE_TERMINAL_PICKER_PROVIDERS
        profile = get_provider_profile("tencent-tokenplan")
        assert profile is not None and profile.auth_type == "api_key"
        live = ["hunyuan-turbo", "hunyuan-pro"]

        with (
            patch.object(profile, "fetch_models", return_value=live),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("tencent-tokenplan")

        assert set(result) == set(live)

    def test_failed_probe_still_falls_back_to_curated_floor(self):
        """Terminal only applies on a SUCCESSFUL probe: without a key the picker must still offer
        the curated floor (offline users keep a usable list)."""
        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(None)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "", "base_url": ""},
            ),
        ):
            result = provider_model_ids("alibaba-token-plan")

        assert "qwen3.8-max-0902" in result  # real curated floor still served offline
        assert result

    def test_non_terminal_providers_keep_the_union_merge(self):
        """Every other provider keeps the #46850 union contract: curated-only ids survive a lagging
        live API. zai is the sentinel (curated-first, NOT terminal)."""
        assert "zai" not in _LIVE_TERMINAL_PICKER_PROVIDERS
        live = ["glm-5"]
        curated = ["glm-5.2", "glm-5.1"]

        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"zai": curated}),
        ):
            result = provider_model_ids("zai")

        assert set(result) >= {"glm-5.2", "glm-5.1", "glm-5"}
