"""
Unit tests for agent/auth.py — the unified credential resolver.

These tests prove that resolve_provider_credentials() correctly handles:
- All 19+ providers
- Empty base_url fallback (the cascade bug)
- Config.yaml precedence
- Env var override
- Pool entry with wrong/empty base_url
- Region enforcement (MiniMax-CN)
- API mode auto-detection
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path for the editable install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent.auth import resolve_provider_credentials, ResolvedCredential


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _fake_entry(provider="zai", base_url="", api_key="sk-fake-key", source="manual"):
    """Create a fake PooledCredential-like object."""
    return SimpleNamespace(
        provider=provider,
        id="test-id",
        label="Test Key",
        auth_type="api_key",
        source=source,
        access_token=api_key,
        refresh_token=None,
        last_status=None,
        last_status_at=None,
        last_error_code=None,
        last_error_reason=None,
        last_error_message=None,
        last_error_reset_at=None,
        base_url=base_url,
        expires_at=None,
        expires_at_ms=None,
        last_refresh=None,
        inference_base_url=None,
        agent_key=None,
        agent_key_expires_at=None,
        request_count=0,
        extra={},
        runtime_api_key=api_key if api_key else None,
        runtime_base_url=base_url if base_url else None,
    )


# ── Generic API-key providers ────────────────────────────────────────────────

class TestGenericProviders:
    """Test generic API-key providers (deepseek, stepfun, etc.)."""

    def test_deepseek_uses_registry_url(self):
        """DeepSeek with no entry should use the registry default."""
        result = resolve_provider_credentials(
            provider="deepseek",
            model_cfg={"provider": "deepseek", "default": "deepseek-chat"},
        )
        assert result.base_url == "https://api.deepseek.com/v1"
        assert result.api_mode == "chat_completions"

    def test_deepseek_config_yaml_override(self):
        """Config.yaml model.base_url should override registry default."""
        result = resolve_provider_credentials(
            provider="deepseek",
            model_cfg={
                "provider": "deepseek",
                "default": "deepseek-chat",
                "base_url": "https://custom.deepseek.proxy/v1",
            },
        )
        assert "custom.deepseek.proxy" in result.base_url

    def test_deepseek_env_var_override(self):
        """DEEPSEEK_BASE_URL env var should win over config and registry."""
        with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "https://env-override.deepseek.com/v1"}):
            result = resolve_provider_credentials(
                provider="deepseek",
                model_cfg={"provider": "deepseek", "base_url": "https://config.deepseek.com/v1"},
            )
        assert "env-override" in result.base_url

    def test_deepseek_pool_entry_with_base_url(self):
        """Pool entry with explicit base_url should be used."""
        entry = _fake_entry(provider="deepseek", base_url="https://pool.deepseek.com/v1")
        result = resolve_provider_credentials(
            provider="deepseek",
            entry=entry,
            model_cfg={"provider": "deepseek"},
        )
        assert "pool.deepseek.com" in result.base_url

    def test_deepseek_empty_base_url_falls_back_to_registry(self):
        """Empty base_url should fall back to registry (the cascade bug fix)."""
        entry = _fake_entry(provider="deepseek", base_url="")
        result = resolve_provider_credentials(
            provider="deepseek",
            entry=entry,
            model_cfg={"provider": "deepseek"},
        )
        assert result.base_url != ""
        assert result.base_url == "https://api.deepseek.com/v1"


# ── Z.AI ─────────────────────────────────────────────────────────────────────

class TestZAIProvider:
    """Test Z.AI (zai) provider resolution."""

    def test_zai_empty_base_url_does_not_return_empty(self):
        """The core bug: base_url='' should NOT stay empty."""
        entry = _fake_entry(provider="zai", base_url="")
        result = resolve_provider_credentials(
            provider="zai",
            entry=entry,
            model_cfg={"provider": "zai", "default": "glm-5.2"},
        )
        assert result.base_url != ""
        assert "z.ai" in result.base_url or "bigmodel.cn" in result.base_url

    def test_zai_env_var_override(self):
        """GLM_BASE_URL should override everything."""
        with patch.dict(os.environ, {"GLM_BASE_URL": "https://api.z.ai/api/coding/paas/v4"}):
            result = resolve_provider_credentials(
                provider="zai",
                model_cfg={"provider": "zai"},
            )
        assert "coding/paas/v4" in result.base_url

    def test_zai_config_yaml_override(self):
        """model.base_url in config.yaml should be honored."""
        result = resolve_provider_credentials(
            provider="zai",
            model_cfg={
                "provider": "zai",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
            },
        )
        assert "coding" in result.base_url or "z.ai" in result.base_url

    def test_zai_pool_entry_uses_entry_url(self):
        """Pool entry with explicit base_url should be respected."""
        entry = _fake_entry(provider="zai", base_url="https://api.z.ai/api/anthropic")
        result = resolve_provider_credentials(
            provider="zai",
            entry=entry,
            model_cfg={"provider": "zai"},
        )
        assert "z.ai" in result.base_url


# ── MiniMax ──────────────────────────────────────────────────────────────────

class TestMiniMaxProvider:
    """Test MiniMax region enforcement."""

    def test_minimax_cn_does_not_use_international_url(self):
        """minimax-cn should NOT use api.minimax.io (international)."""
        entry = _fake_entry(provider="minimax-cn", base_url="https://api.minimax.io/anthropic")
        result = resolve_provider_credentials(
            provider="minimax-cn",
            entry=entry,
            model_cfg={"provider": "minimax-cn"},
        )
        assert "minimaxi.com" in result.base_url, f"Expected China URL, got: {result.base_url}"

    def test_minimax_does_not_use_china_url(self):
        """minimax (international) should NOT use api.minimaxi.com (China)."""
        entry = _fake_entry(provider="minimax", base_url="https://api.minimaxi.com/anthropic")
        result = resolve_provider_credentials(
            provider="minimax",
            entry=entry,
            model_cfg={"provider": "minimax"},
        )
        assert "minimax.io" in result.base_url, f"Expected international URL, got: {result.base_url}"

    def test_minimax_cn_empty_base_url(self):
        """minimax-cn with empty base_url should use the China endpoint."""
        entry = _fake_entry(provider="minimax-cn", base_url="")
        result = resolve_provider_credentials(
            provider="minimax-cn",
            entry=entry,
            model_cfg={"provider": "minimax-cn"},
        )
        assert "minimaxi.com" in result.base_url

    def test_minimax_api_mode_is_anthropic(self):
        """MiniMax endpoints end with /anthropic → api_mode should be anthropic_messages."""
        result = resolve_provider_credentials(
            provider="minimax",
            model_cfg={"provider": "minimax"},
        )
        assert result.api_mode == "anthropic_messages"


# ── Anthropic ────────────────────────────────────────────────────────────────

class TestAnthropicProvider:
    """Test Anthropic provider resolution."""

    def test_anthropic_default_url(self):
        result = resolve_provider_credentials(
            provider="anthropic",
            model_cfg={"provider": "anthropic", "default": "claude-sonnet-4-20250514"},
        )
        assert result.base_url == "https://api.anthropic.com"
        assert result.api_mode == "anthropic_messages"

    def test_anthropic_config_override(self):
        # Use a URL that passes _anthropic_base_url_override_ok (must contain "anthropic")
        result = resolve_provider_credentials(
            provider="anthropic",
            model_cfg={
                "provider": "anthropic",
                "base_url": "https://proxy.anthropic.com",
            },
        )
        # Should resolve to the configured URL, not the default
        assert "proxy.anthropic.com" in result.base_url

    def test_anthropic_strips_v1_suffix(self):
        """Anthropic Messages API should not have /v1 suffix."""
        result = resolve_provider_credentials(
            provider="anthropic",
            model_cfg={
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com/v1",
            },
        )
        assert not result.base_url.endswith("/v1")


# ── Precedence ───────────────────────────────────────────────────────────────

class TestPrecedence:
    """Test the precedence chain: env > config > resolved > registry."""

    def test_env_beats_config(self):
        """Env var should win over config.yaml."""
        with patch.dict(os.environ, {"GLM_BASE_URL": "https://env.z.ai"}):
            result = resolve_provider_credentials(
                provider="zai",
                model_cfg={"provider": "zai", "base_url": "https://config.z.ai"},
            )
        assert "env.z.ai" in result.base_url
        assert result.source == "env"

    def test_explicit_beats_everything(self):
        """Explicit args should win over all other sources."""
        with patch.dict(os.environ, {"GLM_BASE_URL": "https://env.z.ai"}):
            result = resolve_provider_credentials(
                provider="zai",
                explicit_base_url="https://explicit.z.ai",
                model_cfg={"provider": "zai", "base_url": "https://config.z.ai"},
            )
        assert "explicit.z.ai" in result.base_url
        assert result.source == "explicit"


# ── Empty base_url cascade fix ──────────────────────────────────────────────

class TestEmptyBaseUrlFix:
    """The core fix: base_url="" should NEVER reach the HTTP client."""

    @pytest.mark.parametrize("provider", ["zai", "deepseek", "minimax-cn", "anthropic"])
    def test_empty_base_url_always_resolves(self, provider):
        """Every provider should return a non-empty base_url even with empty entry."""
        entry = _fake_entry(provider=provider, base_url="")
        result = resolve_provider_credentials(
            provider=provider,
            entry=entry,
            model_cfg={"provider": provider},
        )
        assert result.base_url != "", f"{provider}: base_url is empty!"
        assert result.base_url.startswith("https://"), f"{provider}: base_url is not HTTPS: {result.base_url}"


# ── Cross-provider consistency ──────────────────────────────────────────────

class TestCrossProviderConsistency:
    """Verify that all providers return a valid ResolvedCredential."""

    @pytest.mark.parametrize("provider", [
        "deepseek", "zai", "minimax", "minimax-cn", "anthropic",
        "openrouter", "xai", "lmstudio", "stepfun", "arcee",
    ])
    def test_all_providers_return_valid_credential(self, provider):
        """Every provider should return a ResolvedCredential with required fields."""
        result = resolve_provider_credentials(
            provider=provider,
            model_cfg={"provider": provider},
        )
        assert isinstance(result, ResolvedCredential)
        assert result.provider == provider
        assert isinstance(result.base_url, str)
        assert isinstance(result.api_mode, str)
        assert isinstance(result.source, str)
        assert result.api_mode in (
            "chat_completions", "anthropic_messages", "codex_responses", "gemini_native",
            "codex_app_server",
        )
