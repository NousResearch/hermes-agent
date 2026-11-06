"""Tests for Xiaomi MiMo provider support."""

import pytest

from hermes_cli.auth import (
    resolve_provider,
    resolve_api_key_provider_credentials,
)

# =============================================================================
# Provider Registry
# =============================================================================

# =============================================================================
# Aliases
# =============================================================================

class TestXiaomiAliases:
    """All aliases should resolve to 'xiaomi'."""

    @pytest.mark.parametrize("alias", [
        "xiaomi", "mimo", "xiaomi-mimo",
    ])
    def test_alias_resolves(self, alias, monkeypatch):
        # Clear env to avoid auto-detection interfering
        for key in ("XIAOMI_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-key-12345678")
        assert resolve_provider(alias) == "xiaomi"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("mimo") == "xiaomi"
        assert normalize_provider("xiaomi-mimo") == "xiaomi"

# =============================================================================
# Auto-detection
# =============================================================================

class TestXiaomiAutoDetection:
    """Setting XIAOMI_API_KEY should auto-detect the provider."""

    def test_auto_detect(self, monkeypatch):
        # Clear all other provider env vars
        for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                     "DASHSCOPE_API_KEY", "XAI_API_KEY", "KIMI_API_KEY",
                     "MINIMAX_API_KEY", "AI_GATEWAY_API_KEY", "KILOCODE_API_KEY",
                     "HF_TOKEN", "GLM_API_KEY", "COPILOT_GITHUB_TOKEN",
                     "GH_TOKEN", "GITHUB_TOKEN", "MINIMAX_CN_API_KEY",
                     "TOKENHUB_API_KEY", "ARCEEAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-xiaomi-test-12345678")
        provider = resolve_provider("auto")
        assert provider == "xiaomi"

# =============================================================================
# Credentials
# =============================================================================

class TestXiaomiCredentials:
    """Test credential resolution for the xiaomi provider."""

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test-12345678")
        monkeypatch.delenv("XIAOMI_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("xiaomi")
        assert creds["api_key"] == "sk-test-12345678"
        assert creds["base_url"] == "https://api.xiaomimimo.com/v1"

    def test_resolve_credentials_reads_home_external_secret_scope(
        self, tmp_path, monkeypatch
    ):
        """BWS-injected keys belong in the profile scope that loaded them."""
        from agent import secret_scope as ss
        from hermes_cli import config as config_module
        from hermes_cli import env_loader

        home = tmp_path / "hermes"
        home.mkdir()
        (home / ".env").write_text("", encoding="utf-8")
        monkeypatch.setattr(config_module, "get_env_path", lambda: home / ".env")
        config_module.invalidate_env_cache()

        monkeypatch.delenv("XIAOMI_BASE_URL", raising=False)
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            str(home.resolve()),
            {"XIAOMI_API_KEY": "sk-bws-xiaomi-12345678"},
        )

        ss.set_multiplex_active(True)
        token = ss.set_secret_scope(ss.build_profile_secret_scope(home))
        try:
            creds = resolve_api_key_provider_credentials("xiaomi")
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert creds["api_key"] == "sk-bws-xiaomi-12345678"
        assert creds["source"] == "XIAOMI_API_KEY"

# =============================================================================
# Model catalog (dynamic — no static list)
# =============================================================================

class TestXiaomiModelCatalog:
    """Xiaomi uses dynamic model discovery via models.dev."""

    def test_static_model_list_fallback(self):
        """Static _PROVIDER_MODELS fallback must exist for model picker.

        We only assert the provider key is present — the specific model
        names are data that changes with upstream releases and doesn't
        belong in tests.
        """
        from hermes_cli.models import _PROVIDER_MODELS
        assert "xiaomi" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["xiaomi"]) >= 1

    def test_list_agentic_models_mock(self, monkeypatch):
        """When models.dev returns Xiaomi data, list_agentic_models should return models."""
        from agent import models_dev as md

        fake_data = {
            "xiaomi": {
                "name": "Xiaomi",
                "api": "https://api.xiaomimimo.com/v1",
                "env": ["XIAOMI_API_KEY"],
                "models": {
                    "mimo-v2-pro": {
                        "limit": {"context": 1000000},
                        "tool_call": True,
                    },
                    "mimo-v2-omni": {
                        "limit": {"context": 256000},
                        "tool_call": True,
                    },
                    "mimo-v2-flash": {
                        "limit": {"context": 256000},
                        "tool_call": True,
                    },
                },
            }
        }
        monkeypatch.setattr(md, "fetch_models_dev", lambda: fake_data)

        result = md.list_agentic_models("xiaomi")
        assert "mimo-v2-pro" in result
        assert "mimo-v2-flash" in result

# =============================================================================
# Normalization
# =============================================================================

class TestXiaomiNormalization:
    """Model name normalization — Xiaomi is a direct provider."""

    def test_lowercase_subset_of_matching_prefix(self):
        """_LOWERCASE_MODEL_PROVIDERS must be a subset of _MATCHING_PREFIX_STRIP_PROVIDERS.

        Otherwise the .lower() code path is unreachable dead code — the
        provider check at line 422 gates entry to the block.
        """
        from hermes_cli.model_normalize import (
            _LOWERCASE_MODEL_PROVIDERS,
            _MATCHING_PREFIX_STRIP_PROVIDERS,
        )
        assert _LOWERCASE_MODEL_PROVIDERS.issubset(_MATCHING_PREFIX_STRIP_PROVIDERS), (
            f"_LOWERCASE_MODEL_PROVIDERS has entries not in _MATCHING_PREFIX_STRIP_PROVIDERS: "
            f"{_LOWERCASE_MODEL_PROVIDERS - _MATCHING_PREFIX_STRIP_PROVIDERS}"
        )

    @pytest.mark.parametrize("input_name,expected", [
        ("MiMo-V2.5-Pro", "mimo-v2.5-pro"),
        ("mimo-v2.5-pro", "mimo-v2.5-pro"),     # already lowercase
    ])
    def test_normalize_lowercases_mixed_case(self, input_name, expected):
        """Xiaomi's API requires lowercase model IDs — mixed case from docs must be lowered."""
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider(input_name, "xiaomi")
        assert result == expected

# =============================================================================
# URL mapping
# =============================================================================

class TestXiaomiURLMapping:
    """Test URL → provider inference for Xiaomi endpoints."""

    def test_infer_from_regional_urls(self):
        """Regional token-plan endpoints should also resolve to xiaomi."""
        from agent.model_metadata import _infer_provider_from_url
        assert _infer_provider_from_url("https://token-plan-ams.xiaomimimo.com/v1") == "xiaomi"
        assert _infer_provider_from_url("https://token-plan-cn.xiaomimimo.com/v1") == "xiaomi"
        assert _infer_provider_from_url("https://token-plan-sgp.xiaomimimo.com/v1") == "xiaomi"

# =============================================================================
# providers.py
# =============================================================================

# =============================================================================
# Auxiliary client
# =============================================================================

# =============================================================================
# Agent init (no SyntaxError, correct api_mode)
# =============================================================================
