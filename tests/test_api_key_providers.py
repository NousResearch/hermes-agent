"""Tests for API-key provider support (z.ai/GLM, Kimi, MiniMax, AI Gateway)."""

import os
import sys
import types

import pytest

# Ensure dotenv doesn't interfere
if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    ProviderConfig,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    get_auth_status,
    AuthError,
    KIMI_CODE_BASE_URL,
    _resolve_kimi_base_url,
)


# =============================================================================
# Provider Registry tests
# =============================================================================

class TestProviderRegistry:
    """Test that new providers are correctly registered."""

    @pytest.mark.parametrize("provider_id,name,auth_type", [
        ("zai", "Z.AI / GLM", "api_key"),
        ("kimi-coding", "Kimi / Moonshot", "api_key"),
        ("minimax", "MiniMax", "api_key"),
        ("minimax-cn", "MiniMax (China)", "api_key"),
        ("ai-gateway", "AI Gateway", "api_key"),
        ("kilocode", "Kilo Code", "api_key"),
    ])
    def test_provider_registered(self, provider_id, name, auth_type):
        assert provider_id in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY[provider_id]
        assert pconfig.name == name
        assert pconfig.auth_type == auth_type
        assert pconfig.inference_base_url  # must have a default base URL

    def test_zai_env_vars(self):
        pconfig = PROVIDER_REGISTRY["zai"]
        assert pconfig.api_key_env_vars == ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY")
        assert pconfig.base_url_env_var == "GLM_BASE_URL"

    def test_kimi_env_vars(self):
        pconfig = PROVIDER_REGISTRY["kimi-coding"]
        assert pconfig.api_key_env_vars == ("KIMI_API_KEY",)
        assert pconfig.base_url_env_var == "KIMI_BASE_URL"

    def test_minimax_env_vars(self):
        pconfig = PROVIDER_REGISTRY["minimax"]
        assert pconfig.api_key_env_vars == ("MINIMAX_API_KEY",)
        assert pconfig.base_url_env_var == "MINIMAX_BASE_URL"

    def test_minimax_cn_env_vars(self):
        pconfig = PROVIDER_REGISTRY["minimax-cn"]
        assert pconfig.api_key_env_vars == ("MINIMAX_CN_API_KEY",)
        assert pconfig.base_url_env_var == "MINIMAX_CN_BASE_URL"

    def test_ai_gateway_env_vars(self):
        pconfig = PROVIDER_REGISTRY["ai-gateway"]
        assert pconfig.api_key_env_vars == ("AI_GATEWAY_API_KEY",)
        assert pconfig.base_url_env_var == "AI_GATEWAY_BASE_URL"

    def test_kilocode_env_vars(self):
        pconfig = PROVIDER_REGISTRY["kilocode"]
        assert pconfig.api_key_env_vars == ("KILOCODE_API_KEY",)
        assert pconfig.base_url_env_var == "KILOCODE_BASE_URL"

    def test_base_urls(self):
        assert PROVIDER_REGISTRY["zai"].inference_base_url == "https://api.z.ai/api/paas/v4"
        assert PROVIDER_REGISTRY["kimi-coding"].inference_base_url == "https://api.moonshot.ai/v1"
        assert PROVIDER_REGISTRY["minimax"].inference_base_url == "https://api.minimax.io/v1"
        assert PROVIDER_REGISTRY["minimax-cn"].inference_base_url == "https://api.minimaxi.com/v1"
        assert PROVIDER_REGISTRY["ai-gateway"].inference_base_url == "https://ai-gateway.vercel.sh/v1"
        assert PROVIDER_REGISTRY["kilocode"].inference_base_url == "https://api.kilo.ai/api/gateway"

    def test_oauth_providers_unchanged(self):
        """Ensure we didn't break the existing OAuth providers."""
        assert "nous" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["nous"].auth_type == "oauth_device_code"
        assert "openai-codex" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["openai-codex"].auth_type == "oauth_external"


# =============================================================================
# Provider Resolution tests
# =============================================================================

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY",
    "KIMI_API_KEY", "KIMI_BASE_URL", "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY",
    "AI_GATEWAY_API_KEY", "AI_GATEWAY_BASE_URL",
    "KILOCODE_API_KEY", "KILOCODE_BASE_URL",
    "DASHSCOPE_API_KEY", "OPENCODE_ZEN_API_KEY", "OPENCODE_GO_API_KEY",
    "NOUS_API_KEY",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in PROVIDER_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})


class TestResolveProvider:
    """Test resolve_provider() with new providers."""

    def test_explicit_zai(self):
        assert resolve_provider("zai") == "zai"

    def test_explicit_kimi_coding(self):
        assert resolve_provider("kimi-coding") == "kimi-coding"

    def test_explicit_minimax(self):
        assert resolve_provider("minimax") == "minimax"

    def test_explicit_minimax_cn(self):
        assert resolve_provider("minimax-cn") == "minimax-cn"

    def test_explicit_ai_gateway(self):
        assert resolve_provider("ai-gateway") == "ai-gateway"

    def test_alias_glm(self):
        assert resolve_provider("glm") == "zai"

    def test_alias_z_ai(self):
        assert resolve_provider("z-ai") == "zai"

    def test_alias_zhipu(self):
        assert resolve_provider("zhipu") == "zai"

    def test_alias_kimi(self):
        assert resolve_provider("kimi") == "kimi-coding"

    def test_alias_moonshot(self):
        assert resolve_provider("moonshot") == "kimi-coding"

    def test_alias_minimax_underscore(self):
        assert resolve_provider("minimax_cn") == "minimax-cn"

    def test_alias_aigateway(self):
        assert resolve_provider("aigateway") == "ai-gateway"

    def test_alias_vercel(self):
        assert resolve_provider("vercel") == "ai-gateway"

    def test_explicit_kilocode(self):
        assert resolve_provider("kilocode") == "kilocode"

    def test_alias_kilo(self):
        assert resolve_provider("kilo") == "kilocode"

    def test_alias_kilo_code(self):
        assert resolve_provider("kilo-code") == "kilocode"

    def test_alias_kilo_gateway(self):
        assert resolve_provider("kilo-gateway") == "kilocode"

    def test_alias_case_insensitive(self):
        assert resolve_provider("GLM") == "zai"
        assert resolve_provider("Z-AI") == "zai"
        assert resolve_provider("Kimi") == "kimi-coding"

    def test_unknown_provider_raises(self):
        with pytest.raises(AuthError):
            resolve_provider("nonexistent-provider-xyz")

    def test_auto_detects_glm_key(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-glm-key")
        assert resolve_provider("auto") == "zai"

    def test_auto_detects_zai_key(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
        assert resolve_provider("auto") == "zai"

    def test_auto_detects_z_ai_key(self, monkeypatch):
        monkeypatch.setenv("Z_AI_API_KEY", "test-z-ai-key")
        assert resolve_provider("auto") == "zai"

    def test_auto_detects_kimi_key(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
        assert resolve_provider("auto") == "kimi-coding"

    def test_auto_detects_minimax_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-mm-key")
        assert resolve_provider("auto") == "minimax"

    def test_auto_detects_minimax_cn_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-mm-cn-key")
        assert resolve_provider("auto") == "minimax-cn"

    def test_auto_detects_ai_gateway_key(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-gw-key")
        assert resolve_provider("auto") == "ai-gateway"

    def test_auto_detects_kilocode_key(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "test-kilo-key")
        assert resolve_provider("auto") == "kilocode"

    def test_openrouter_takes_priority_over_glm(self, monkeypatch):
        """OpenRouter API key should win over GLM in auto-detection."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        assert resolve_provider("auto") == "openrouter"


# =============================================================================
# API Key Provider Status tests
# =============================================================================

class TestApiKeyProviderStatus:

    def test_unconfigured_provider(self):
        status = get_api_key_provider_status("zai")
        assert status["configured"] is False
        assert status["logged_in"] is False

    def test_configured_provider(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-key-123")
        status = get_api_key_provider_status("zai")
        assert status["configured"] is True
        assert status["logged_in"] is True
        assert status["key_source"] == "GLM_API_KEY"
        assert "z.ai" in status["base_url"].lower() or "api.z.ai" in status["base_url"]

    def test_fallback_env_var(self, monkeypatch):
        """ZAI_API_KEY should work when GLM_API_KEY is not set."""
        monkeypatch.setenv("ZAI_API_KEY", "zai-fallback-key")
        status = get_api_key_provider_status("zai")
        assert status["configured"] is True
        assert status["key_source"] == "ZAI_API_KEY"

    def test_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kimi-key")
        monkeypatch.setenv("KIMI_BASE_URL", "https://custom.kimi.example/v1")
        status = get_api_key_provider_status("kimi-coding")
        assert status["base_url"] == "https://custom.kimi.example/v1"

    def test_get_auth_status_dispatches_to_api_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
        status = get_auth_status("minimax")
        assert status["configured"] is True
        assert status["provider"] == "minimax"

    def test_non_api_key_provider(self):
        status = get_api_key_provider_status("nous")
        assert status["configured"] is False


# =============================================================================
# Credential Resolution tests
# =============================================================================

class TestResolveApiKeyProviderCredentials:

    def test_resolve_zai_with_key(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-secret-key")
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["provider"] == "zai"
        assert creds["api_key"] == "glm-secret-key"
        assert creds["base_url"] == "https://api.z.ai/api/paas/v4"
        assert creds["source"] == "GLM_API_KEY"

    def test_resolve_kimi_with_key(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kimi-secret-key")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["provider"] == "kimi-coding"
        assert creds["api_key"] == "kimi-secret-key"
        assert creds["base_url"] == "https://api.moonshot.ai/v1"

    def test_resolve_minimax_with_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-secret-key")
        creds = resolve_api_key_provider_credentials("minimax")
        assert creds["provider"] == "minimax"
        assert creds["api_key"] == "mm-secret-key"
        assert creds["base_url"] == "https://api.minimax.io/v1"

    def test_resolve_minimax_cn_with_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "mmcn-secret-key")
        creds = resolve_api_key_provider_credentials("minimax-cn")
        assert creds["provider"] == "minimax-cn"
        assert creds["api_key"] == "mmcn-secret-key"
        assert creds["base_url"] == "https://api.minimaxi.com/v1"

    def test_resolve_ai_gateway_with_key(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-secret-key")
        creds = resolve_api_key_provider_credentials("ai-gateway")
        assert creds["provider"] == "ai-gateway"
        assert creds["api_key"] == "gw-secret-key"
        assert creds["base_url"] == "https://ai-gateway.vercel.sh/v1"

    def test_resolve_kilocode_with_key(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "kilo-secret-key")
        creds = resolve_api_key_provider_credentials("kilocode")
        assert creds["provider"] == "kilocode"
        assert creds["api_key"] == "kilo-secret-key"
        assert creds["base_url"] == "https://api.kilo.ai/api/gateway"

    def test_resolve_kilocode_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "kilo-key")
        monkeypatch.setenv("KILOCODE_BASE_URL", "https://custom.kilo.example/v1")
        creds = resolve_api_key_provider_credentials("kilocode")
        assert creds["base_url"] == "https://custom.kilo.example/v1"

    def test_resolve_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        monkeypatch.setenv("GLM_BASE_URL", "https://custom.glm.example/v4")
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://custom.glm.example/v4"

    def test_resolve_without_key_returns_empty(self):
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["api_key"] == ""
        assert creds["source"] == "default"

    def test_resolve_invalid_provider_raises(self):
        with pytest.raises(AuthError):
            resolve_api_key_provider_credentials("nous")

    def test_glm_key_priority(self, monkeypatch):
        """GLM_API_KEY takes priority over ZAI_API_KEY."""
        monkeypatch.setenv("GLM_API_KEY", "primary")
        monkeypatch.setenv("ZAI_API_KEY", "secondary")
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["api_key"] == "primary"
        assert creds["source"] == "GLM_API_KEY"

    def test_zai_key_fallback(self, monkeypatch):
        """ZAI_API_KEY used when GLM_API_KEY not set."""
        monkeypatch.setenv("ZAI_API_KEY", "secondary")
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["api_key"] == "secondary"
        assert creds["source"] == "ZAI_API_KEY"


# =============================================================================
# Runtime Provider Resolution tests
# =============================================================================

class TestRuntimeProviderResolution:

    def test_runtime_zai(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="zai")
        assert result["provider"] == "zai"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "glm-key"
        assert "z.ai" in result["base_url"] or "api.z.ai" in result["base_url"]

    def test_runtime_kimi(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kimi-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="kimi-coding")
        assert result["provider"] == "kimi-coding"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "kimi-key"

    def test_runtime_minimax(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="minimax")
        assert result["provider"] == "minimax"
        assert result["api_key"] == "mm-key"

    def test_runtime_ai_gateway(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="ai-gateway")
        assert result["provider"] == "ai-gateway"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "gw-key"
        assert "ai-gateway.vercel.sh" in result["base_url"]

    def test_runtime_kilocode(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "kilo-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="kilocode")
        assert result["provider"] == "kilocode"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "kilo-key"
        assert "kilo.ai" in result["base_url"]

    def test_runtime_auto_detects_api_key_provider(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "auto-kimi-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="auto")
        assert result["provider"] == "kimi-coding"
        assert result["api_key"] == "auto-kimi-key"


# =============================================================================
# _has_any_provider_configured tests
# =============================================================================

class TestHasAnyProviderConfigured:

    def test_glm_key_counts(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_module
        monkeypatch.setenv("GLM_API_KEY", "test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True

    def test_minimax_key_counts(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_module
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True


# =============================================================================
# Kimi Code auto-detection tests
# =============================================================================

MOONSHOT_DEFAULT_URL = "https://api.moonshot.ai/v1"


class TestResolveKimiBaseUrl:
    """Test _resolve_kimi_base_url() helper for key-prefix auto-detection."""

    def test_sk_kimi_prefix_routes_to_kimi_code(self):
        url = _resolve_kimi_base_url("sk-kimi-abc123", MOONSHOT_DEFAULT_URL, "")
        assert url == KIMI_CODE_BASE_URL

    def test_legacy_key_uses_default(self):
        url = _resolve_kimi_base_url("sk-abc123", MOONSHOT_DEFAULT_URL, "")
        assert url == MOONSHOT_DEFAULT_URL

    def test_empty_key_uses_default(self):
        url = _resolve_kimi_base_url("", MOONSHOT_DEFAULT_URL, "")
        assert url == MOONSHOT_DEFAULT_URL

    def test_env_override_wins_over_sk_kimi(self):
        """KIMI_BASE_URL env var should always take priority."""
        custom = "https://custom.example.com/v1"
        url = _resolve_kimi_base_url("sk-kimi-abc123", MOONSHOT_DEFAULT_URL, custom)
        assert url == custom

    def test_env_override_wins_over_legacy(self):
        custom = "https://custom.example.com/v1"
        url = _resolve_kimi_base_url("sk-abc123", MOONSHOT_DEFAULT_URL, custom)
        assert url == custom


class TestKimiCodeStatusAutoDetect:
    """Test that get_api_key_provider_status auto-detects sk-kimi- keys."""

    def test_sk_kimi_key_gets_kimi_code_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test-key-123")
        status = get_api_key_provider_status("kimi-coding")
        assert status["configured"] is True
        assert status["base_url"] == KIMI_CODE_BASE_URL

    def test_legacy_key_gets_moonshot_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-legacy-test-key")
        status = get_api_key_provider_status("kimi-coding")
        assert status["configured"] is True
        assert status["base_url"] == MOONSHOT_DEFAULT_URL

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test-key")
        monkeypatch.setenv("KIMI_BASE_URL", "https://override.example/v1")
        status = get_api_key_provider_status("kimi-coding")
        assert status["base_url"] == "https://override.example/v1"


class TestKimiCodeCredentialAutoDetect:
    """Test that resolve_api_key_provider_credentials auto-detects sk-kimi- keys."""

    def test_sk_kimi_key_gets_kimi_code_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-secret-key")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["api_key"] == "sk-kimi-secret-key"
        assert creds["base_url"] == KIMI_CODE_BASE_URL

    def test_legacy_key_gets_moonshot_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-legacy-secret-key")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["api_key"] == "sk-legacy-secret-key"
        assert creds["base_url"] == MOONSHOT_DEFAULT_URL

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-secret-key")
        monkeypatch.setenv("KIMI_BASE_URL", "https://override.example/v1")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["base_url"] == "https://override.example/v1"

    def test_non_kimi_providers_unaffected(self, monkeypatch):
        """Ensure the auto-detect logic doesn't leak to other providers."""
        monkeypatch.setenv("GLM_API_KEY", "sk-kimi-looks-like-kimi-but-isnt")
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://api.z.ai/api/paas/v4"


# =============================================================================
# Kimi / Moonshot model list isolation tests
# =============================================================================

class TestKimiMoonshotModelListIsolation:
    """Moonshot (legacy) users must not see Coding Plan-only models."""

    def test_moonshot_list_excludes_coding_plan_only_models(self):
        from hermes_cli.main import _PROVIDER_MODELS
        moonshot_models = _PROVIDER_MODELS["moonshot"]
        coding_plan_only = {"kimi-for-coding", "kimi-k2-thinking-turbo"}
        leaked = set(moonshot_models) & coding_plan_only
        assert not leaked, f"Moonshot list contains Coding Plan-only models: {leaked}"

    def test_moonshot_list_contains_shared_models(self):
        from hermes_cli.main import _PROVIDER_MODELS
        moonshot_models = _PROVIDER_MODELS["moonshot"]
        assert "kimi-k2.5" in moonshot_models
        assert "kimi-k2-thinking" in moonshot_models

    def test_coding_plan_list_contains_plan_specific_models(self):
        from hermes_cli.main import _PROVIDER_MODELS
        coding_models = _PROVIDER_MODELS["kimi-coding"]
        assert "kimi-for-coding" in coding_models
        assert "kimi-k2-thinking-turbo" in coding_models
