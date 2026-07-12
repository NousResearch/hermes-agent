"""Tests for the models.dev-preferred merge behavior in provider_model_ids
and list_authenticated_providers.

These guard the contract:

  * For providers in ``_MODELS_DEV_PREFERRED`` (opencode-go, opencode-zen,
    xiaomi, deepseek, smaller inference providers), both the CLI model
    picker path (``provider_model_ids``) and the gateway ``/model`` picker
    path (``list_authenticated_providers``) merge fresh models.dev entries
    on top of the curated static list.
  * OpenRouter and Nous Portal are NEVER merged — they keep their curated
    (OpenRouter) or live-Portal (Nous) semantics.
  * If models.dev is unreachable (offline / CI), the curated list is the
    fallback — no crash, no empty list.

Merging is what lets new models (e.g. ``mimo-v2.5-pro`` on opencode-go)
appear in ``/model`` without a Hermes release.
"""

from unittest.mock import patch


from hermes_cli.models import (
    _MODELS_DEV_PREFERRED,
    _PROVIDER_MODELS,
    _merge_with_models_dev,
    provider_model_ids,
)


class TestMergeHelper:
    def test_merge_empty_mdev_returns_curated(self):
        """When models.dev returns nothing, curated list is preserved verbatim."""
        with patch("agent.models_dev.list_agentic_models", return_value=[]):
            out = _merge_with_models_dev("opencode-go", ["mimo-v2-pro", "kimi-k2.6"])
        assert out == ["mimo-v2-pro", "kimi-k2.6"]

    def test_merge_mdev_raises_returns_curated(self):
        """Offline / broken models.dev must not break the catalog path."""
        def boom(_provider):
            raise RuntimeError("network down")

        with patch("agent.models_dev.list_agentic_models", side_effect=boom):
            out = _merge_with_models_dev("opencode-go", ["mimo-v2-pro"])
        assert out == ["mimo-v2-pro"]

    def test_merge_mdev_first_then_curated_extras(self):
        """models.dev entries come first; curated-only entries are appended."""
        mdev = ["mimo-v2.5-pro", "mimo-v2-pro", "kimi-k2.6"]
        curated = ["kimi-k2.6", "kimi-k2.5", "mimo-v2-pro"]  # kimi-k2.5 is curated-only
        with patch("agent.models_dev.list_agentic_models", return_value=mdev):
            out = _merge_with_models_dev("opencode-go", curated)
        # models.dev entries first (in order), then curated-only entries
        assert out == ["mimo-v2.5-pro", "mimo-v2-pro", "kimi-k2.6", "kimi-k2.5"]

    def test_merge_case_insensitive_dedup(self):
        """Dedup is case-insensitive but preserves the first occurrence's casing."""
        mdev = ["MiniMax-M2.7"]
        curated = ["minimax-m2.7", "minimax-m2.5"]
        with patch("agent.models_dev.list_agentic_models", return_value=mdev):
            out = _merge_with_models_dev("minimax", curated)
        # models.dev casing wins since it came first
        assert out == ["MiniMax-M2.7", "minimax-m2.5"]


class TestProviderModelIdsPreferred:
    def test_opencode_go_is_preferred(self):
        assert "opencode-go" in _MODELS_DEV_PREFERRED

    def test_opencode_go_includes_fresh_models_dev_entries(self):
        """provider_model_ids('opencode-go') adds models.dev entries on top."""
        mdev = ["mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "kimi-k2.6"]
        with patch("agent.models_dev.list_agentic_models", return_value=mdev):
            out = provider_model_ids("opencode-go")
        # Fresh models must surface (this is exactly the reported bug fix:
        # mimo-v2.5-pro should be pickable on opencode-go).
        assert "mimo-v2.5-pro" in out
        assert "mimo-v2.5" in out
        # Curated entries are still present.
        assert "mimo-v2-pro" in out
        assert "kimi-k2.6" in out

    def test_opencode_go_offline_falls_back_to_curated(self):
        """Offline models.dev → curated-only list, no crash."""
        with patch("agent.models_dev.list_agentic_models", return_value=[]):
            out = provider_model_ids("opencode-go")
        # Curated floor (see hermes_cli/models.py _PROVIDER_MODELS["opencode-go"])
        assert "mimo-v2-pro" in out
        assert "kimi-k2.6" in out

    def test_opencode_zen_includes_fresh_models(self):
        """opencode-zen follows the same pattern as opencode-go."""
        assert "opencode-zen" in _MODELS_DEV_PREFERRED
        mdev = ["claude-opus-4-7", "kimi-k2.6", "glm-5.1"]
        with patch("agent.models_dev.list_agentic_models", return_value=mdev):
            out = provider_model_ids("opencode-zen")
        assert "claude-opus-4-7" in out
        assert "kimi-k2.6" in out

    def test_kimi_coding_offline_catalog_includes_k2_7_code(self):
        """Native Kimi users must see the newest Code model without live catalog help."""
        assert "kimi-coding" not in _MODELS_DEV_PREFERRED
        with patch("agent.models_dev.list_agentic_models", return_value=[]):
            out = provider_model_ids("kimi-coding")
        assert "kimi-k2.7-code" in out

    def test_kimi_coding_live_catalog_does_not_hide_curated_k2_7_code(self):
        """Kimi /models can lag inference; live results must not replace curated."""
        with (
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "sk-test", "base_url": "https://api.moonshot.ai/v1"},
            ),
            patch("providers.base.ProviderProfile.fetch_models", return_value=["kimi-k2.6"]),
        ):
            out = provider_model_ids("kimi-coding")
        # Curated-first order; curated newest (k2.7-code) stays ahead of live.
        assert out[:2] == ["kimi-k2.7-code", "kimi-k2.6"]

    def test_kimi_setup_flow_uses_same_coding_plan_catalog(self):
        """The setup wizard must not carry a stale duplicate Kimi model list."""
        from hermes_cli.model_setup_flows import _model_flow_kimi

        captured = {}

        def fake_select(model_list, **_kwargs):
            captured["models"] = model_list
            return None

        with (
            patch("hermes_cli.main._prompt_api_key", return_value=("sk-kimi-test", False)),
            patch("hermes_cli.auth._prompt_model_selection", side_effect=fake_select),
            patch("hermes_cli.config.get_env_value", return_value=""),
            patch("hermes_cli.config.save_env_value"),
            patch(
                "hermes_cli.models.provider_model_ids",
                return_value=["kimi-k2.7-code", "kimi-k2.7-code-highspeed"],
            ) as resolver,
        ):
            _model_flow_kimi({}, current_model="")

        resolver.assert_called_once_with("kimi-coding", force_refresh=True)
        assert captured["models"] == ["kimi-k2.7-code", "kimi-k2.7-code-highspeed"]


class TestOpenRouterAndNousUnchanged:
    """Per Teknium: openrouter and nous are NEVER merged with models.dev."""

    def test_openrouter_not_in_preferred_set(self):
        assert "openrouter" not in _MODELS_DEV_PREFERRED

    def test_nous_not_in_preferred_set(self):
        assert "nous" not in _MODELS_DEV_PREFERRED

    def test_openrouter_does_not_call_merge(self):
        """openrouter takes its own live path — merge helper must NOT run."""
        with patch(
            "hermes_cli.models._merge_with_models_dev",
            side_effect=AssertionError("merge should not be called for openrouter"),
        ):
            # Even if model_ids() fails for some other reason, we just care
            # that the merge path isn't invoked.
            try:
                provider_model_ids("openrouter")
            except AssertionError:
                raise
            except Exception:
                pass  # model_ids() may fail in the hermetic test env — that's fine.
