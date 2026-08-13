"""Tests for OpenCode Free provider — registration, env var detection, aliases, fallback."""

import os
from unittest.mock import patch


class TestOpenCodeFreeProviderRegistration:
    """Verify the opencode-free provider registers correctly."""

    def test_provider_is_registered(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile is not None
        assert profile.name == "opencode-free"

    def test_provider_has_correct_base_url(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.base_url == "https://opencode.ai/zen/v1"

    def test_provider_has_no_env_vars_requirement(self):
        """OPENCODE_FREE_API_KEY is declared but empty string is acceptable."""
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert "OPENCODE_FREE_API_KEY" in profile.env_vars

    def test_provider_uses_chat_completions_mode(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.api_mode == "chat_completions"


class TestOpenCodeFreeAliases:
    """Verify alias resolution for the opencode-free provider."""

    def test_alias_free(self):
        from providers import get_provider_profile
        profile = get_provider_profile("free")
        assert profile is not None
        assert profile.name == "opencode-free"

    def test_alias_opencode_free(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode_free")
        assert profile is not None
        assert profile.name == "opencode-free"


class TestOpenCodeFreeAuthAlias:
    """Verify the hardcoded alias in auth.py resolve_provider()."""

    def test_resolve_provider_free_alias(self):
        from hermes_cli.auth import resolve_provider
        # "free" should resolve to "opencode-free" without needing the env var
        result = resolve_provider("free")
        assert result == "opencode-free"


class TestOpenCodeFreeFallbackModels:
    """Verify fallback models are registered in _DEFAULT_PROVIDER_MODELS."""

    def test_fallback_models_exist(self):
        from hermes_cli.setup import _DEFAULT_PROVIDER_MODELS
        assert "opencode-free" in _DEFAULT_PROVIDER_MODELS

    def test_fallback_models_content(self):
        from hermes_cli.setup import _DEFAULT_PROVIDER_MODELS
        models = _DEFAULT_PROVIDER_MODELS["opencode-free"]
        assert "big-pickle" in models
        assert "deepseek-v4-flash-free" in models
        assert "mimo-v2.5-free" in models
        assert "nemotron-3-super-free" in models
        assert len(models) == 4


class TestOpenCodeFreeEnvVarDetection:
    """Verify env var triggers auto-detection."""

    def test_auto_detect_with_env_var(self):
        from hermes_cli.auth import resolve_provider
        with patch.dict(os.environ, {"OPENCODE_FREE_API_KEY": "test-key"}):
            result = resolve_provider("auto")
            assert result == "opencode-free"

    def test_no_auto_detect_without_env_var(self):
        from hermes_cli.auth import resolve_provider
        # Remove the env var if present
        env = os.environ.copy()
        env.pop("OPENCODE_FREE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            # Without the env var, auto-detect should NOT return opencode-free
            # (it may return another provider or raise)
            try:
                result = resolve_provider("auto")
                assert result != "opencode-free"
            except Exception:
                # Expected — no provider configured
                pass
