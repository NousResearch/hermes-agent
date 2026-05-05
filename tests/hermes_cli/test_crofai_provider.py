"""Tests for CrofAI provider support — standard direct API provider."""

import types

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
)


_OTHER_PROVIDER_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
    "XAI_API_KEY", "KIMI_API_KEY", "KIMI_CN_API_KEY",
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY", "AI_GATEWAY_API_KEY",
    "KILOCODE_API_KEY", "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "TOKENHUB_API_KEY", "ARCEEAI_API_KEY",
    "GMI_API_KEY", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
)


# =============================================================================
# Provider Registry
# =============================================================================


class TestCrofAIProviderRegistry:
    def test_registered(self):
        assert "crofai" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["crofai"].name == "CrofAI"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["crofai"].auth_type == "api_key"

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["crofai"].inference_base_url == "https://crof.ai/v1"

    def test_api_key_env_vars(self):
        assert PROVIDER_REGISTRY["crofai"].api_key_env_vars == ("CROFAI_API_KEY",)

    def test_base_url_env_var(self):
        assert PROVIDER_REGISTRY["crofai"].base_url_env_var == "CROFAI_BASE_URL"


# =============================================================================
# Aliases
# =============================================================================


class TestCrofAIAliases:
    @pytest.mark.parametrize("alias", ["crofai", "crof", "crof-ai", "crof.ai"])
    def test_alias_resolves(self, alias, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS + ("OPENROUTER_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("CROFAI_API_KEY", "crof-test-12345")
        assert resolve_provider(alias) == "crofai"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("crof") == "crofai"
        assert normalize_provider("crof-ai") == "crofai"
        assert normalize_provider("crof.ai") == "crofai"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("crof") == "crofai"
        assert normalize_provider("crof-ai") == "crofai"
        assert normalize_provider("crof.ai") == "crofai"


# =============================================================================
# Credentials
# =============================================================================


class TestCrofAICredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("CROFAI_API_KEY", "crof-test")
        status = get_api_key_provider_status("crofai")
        assert status["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("CROFAI_API_KEY", raising=False)
        status = get_api_key_provider_status("crofai")
        assert not status["configured"]

    def test_openrouter_key_does_not_make_crofai_configured(self, monkeypatch):
        """OpenRouter users should NOT see CrofAI as configured."""
        monkeypatch.delenv("CROFAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("crofai")
        assert not status["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("CROFAI_API_KEY", "crof-direct-key")
        monkeypatch.delenv("CROFAI_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("crofai")
        assert creds["api_key"] == "crof-direct-key"
        assert creds["base_url"] == "https://crof.ai/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("CROFAI_API_KEY", "crof-x")
        monkeypatch.setenv("CROFAI_BASE_URL", "https://custom.crof.example/v1")
        creds = resolve_api_key_provider_credentials("crofai")
        assert creds["base_url"] == "https://custom.crof.example/v1"


# =============================================================================
# Model catalog
# =============================================================================


class TestCrofAIModelCatalog:
    def test_static_model_list(self):
        """CrofAI has a static _PROVIDER_MODELS catalog entry. Specific model
        names change with releases and don't belong in tests — assert the
        relationship (catalog populated, model IDs look like simple slugs)
        instead of snapshotting names. See AGENTS.md change-detector rule.
        """
        from hermes_cli.models import _PROVIDER_MODELS
        assert "crofai" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["crofai"]
        assert len(models) >= 1
        # Invariant: CrofAI catalog ships bare model slugs (no provider/ prefix).
        for m in models:
            assert "/" not in m, (
                f"CrofAI catalog model {m!r} unexpectedly contains '/'; "
                "the API returns bare model IDs (e.g. 'kimi-k2.5')."
            )

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "crofai" in slugs


# =============================================================================
# Live /v1/models fetch path — fetch is best-effort; static fallback is the
# contract this test guards.
# =============================================================================


class TestCrofAILiveModelsFallback:
    def test_provider_model_ids_falls_back_to_static_when_unauthenticated(self, monkeypatch):
        from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids

        for key in _OTHER_PROVIDER_KEYS + ("CROFAI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        ids = provider_model_ids("crofai")
        assert ids == _PROVIDER_MODELS["crofai"]


# =============================================================================
# Model normalization
# =============================================================================


class TestCrofAINormalization:
    def test_in_matching_prefix_strip_set(self):
        from hermes_cli.model_normalize import _MATCHING_PREFIX_STRIP_PROVIDERS
        assert "crofai" in _MATCHING_PREFIX_STRIP_PROVIDERS

    def test_strips_prefix(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        assert normalize_model_for_provider("crofai/kimi-k2.5", "crofai") == "kimi-k2.5"

    def test_bare_name_unchanged(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        assert normalize_model_for_provider("kimi-k2.5", "crofai") == "kimi-k2.5"


# =============================================================================
# URL mapping
# =============================================================================


class TestCrofAIURLMapping:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("crof.ai") == "crofai"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "crofai" in _PROVIDER_PREFIXES
        assert "crof" in _PROVIDER_PREFIXES
        assert "crof-ai" in _PROVIDER_PREFIXES

    def test_trajectory_compressor_detects_crofai(self):
        import trajectory_compressor as tc
        comp = tc.TrajectoryCompressor.__new__(tc.TrajectoryCompressor)
        comp.config = types.SimpleNamespace(base_url="https://crof.ai/v1")
        assert comp._detect_provider() == "crofai"


# =============================================================================
# providers.py overlay + label
# =============================================================================


class TestCrofAIProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "crofai" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["crofai"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "CROFAI_BASE_URL"
        assert overlay.is_aggregator

    def test_label(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert _PROVIDER_LABELS["crofai"] == "CrofAI"


# =============================================================================
# Auxiliary client — cheapest/fastest model in the catalog
# =============================================================================


class TestCrofAIAuxiliary:
    def test_aux_model_is_in_catalog(self):
        """The auxiliary default must be a model CrofAI actually serves; if a
        future change retires the chosen aux, this test catches the dangling
        reference instead of failing at first vision/compression call.
        """
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        from hermes_cli.models import _PROVIDER_MODELS

        aux = _API_KEY_PROVIDER_AUX_MODELS.get("crofai")
        assert aux, "CrofAI should have an auxiliary-model default"
        assert aux in _PROVIDER_MODELS["crofai"]
