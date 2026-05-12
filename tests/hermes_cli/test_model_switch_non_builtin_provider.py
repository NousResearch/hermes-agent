"""Regression: /model on a user-defined provider (providers: dict) must not
auto-detect and switch to another provider.

When the current provider is NOT a built-in provider (e.g. sensenova defined
in config.yaml providers:), detect_provider_for_model() should be skipped —
the model likely belongs to the current custom provider, not OpenRouter or
any other aggregator.

See: https://github.com/NousResearch/hermes-agent/issues/8470
     https://github.com/NousResearch/hermes-agent/pull/13764
"""

import pytest
from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _make_runtime(base_url="https://token.sensenova.cn/v1", api_key="sk-test"):
    """Return a mock resolve_runtime_provider result."""
    return {
        "api_key": api_key,
        "base_url": base_url,
        "api_mode": "chat_completions",
    }


class TestUserProviderModelSwitchNoLeak:
    """Non-built-in providers must not trigger detect_provider_for_model()."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch):
        # Prevent live network calls
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kw: _make_runtime(),
        )
        monkeypatch.setattr(
            "hermes_cli.models.validate_requested_model",
            lambda *a, **k: _MOCK_VALIDATION,
        )
        monkeypatch.setattr(
            "hermes_cli.model_switch.get_model_info", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None
        )
        monkeypatch.setattr(
            "hermes_cli.model_switch.list_provider_models", lambda *a, **k: None
        )

    def test_non_builtin_provider_does_not_leak_to_openrouter(self, monkeypatch):
        """sensenova (user-defined) should NOT auto-detect to OpenRouter.

        Even if OpenRouter catalog has 'sensenova/sensenova-6.7-flash-lite',
        the switch must stay on the user's current provider.
        """
        detect_calls = []

        def _fake_detect(model, provider):
            detect_calls.append((model, provider))
            # Simulate: OpenRouter catalog has this model
            return ("openrouter", "sensenova/sensenova-6.7-flash-lite")

        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model", _fake_detect
        )

        result = switch_model(
            raw_input="sensenova-6.7-flash-lite",
            current_provider="sensenova",
            current_model="sensenova-6.7-flash-lite",
            current_base_url="https://token.sensenova.cn/v1",
            current_api_key="sk-test",
            user_providers={
                "sensenova": {
                    "base_url": "https://token.sensenova.cn/v1",
                    "api_key": "sk-test",
                }
            },
        )

        assert result.success is True, f"Switch failed: {result.error_message}"
        # Provider must remain sensenova, NOT openrouter
        assert result.target_provider == "sensenova", (
            f"Provider leaked to {result.target_provider}, expected sensenova"
        )
        assert result.new_model == "sensenova-6.7-flash-lite"
        # detect_provider_for_model should NOT have been called at all
        assert detect_calls == [], (
            f"detect_provider_for_model was called {detect_calls}, "
            "should be skipped for non-built-in providers"
        )

    def test_custom_provider_does_not_leak(self, monkeypatch):
        """Provider named 'custom' should not leak (already covered by is_custom)."""
        detect_calls = []

        def _fake_detect(model, provider):
            detect_calls.append((model, provider))
            return ("openrouter", "meta-llama/llama-3-70b")

        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model", _fake_detect
        )

        result = switch_model(
            raw_input="llama-3-70b",
            current_provider="custom",
            current_model="llama-3-70b",
            current_base_url="http://localhost:11434/v1",
            current_api_key="",
            user_providers={},
        )

        assert result.success is True
        assert result.target_provider == "custom"
        assert detect_calls == [], (
            "detect_provider_for_model should be skipped for 'custom' provider"
        )

    def test_local_provider_does_not_leak(self, monkeypatch):
        """Provider named 'local' should not leak (already covered by is_custom)."""
        detect_calls = []

        def _fake_detect(model, provider):
            detect_calls.append((model, provider))
            return ("openrouter", "qwen/qwen3-coder")

        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model", _fake_detect
        )

        result = switch_model(
            raw_input="qwen3-coder",
            current_provider="local",
            current_model="qwen3-coder",
            current_base_url="http://127.0.0.1:11434/v1",
            current_api_key="",
            user_providers={},
        )

        assert result.success is True
        assert result.target_provider == "local"
        assert detect_calls == [], (
            "detect_provider_for_model should be skipped for 'local' provider"
        )

    def test_localhost_base_url_does_not_leak(self, monkeypatch):
        """Provider with localhost base_url should not leak (already covered)."""
        detect_calls = []

        def _fake_detect(model, provider):
            detect_calls.append((model, provider))
            return ("openrouter", "deepseek/deepseek-chat")

        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model", _fake_detect
        )

        result = switch_model(
            raw_input="deepseek-chat",
            current_provider="custom",
            current_model="deepseek-chat",
            current_base_url="http://localhost:8080/v1",
            current_api_key="",
            user_providers={},
        )

        assert result.success is True
        assert detect_calls == [], (
            "detect_provider_for_model should be skipped for localhost base_url"
        )

    def test_builtin_openrouter_still_works(self, monkeypatch):
        """OpenRouter itself (a built-in aggregator) should still work normally."""
        result = switch_model(
            raw_input="deepseek-chat",
            current_provider="openrouter",
            current_model="gpt-5.4",
            current_base_url="https://openrouter.ai/api/v1",
            current_api_key="sk-or",
            user_providers={},
        )

        assert result.success is True

    def test_builtin_deepseek_still_works(self, monkeypatch):
        """DeepSeek (built-in) should still work normally."""
        result = switch_model(
            raw_input="deepseek-chat",
            current_provider="deepseek",
            current_model="deepseek-chat",
            current_base_url="https://api.deepseek.com/v1",
            current_api_key="sk-ds",
            user_providers={},
        )

        assert result.success is True

    def test_builtin_zai_still_works(self, monkeypatch):
        """Z.AI (built-in) should still work normally."""
        result = switch_model(
            raw_input="glm-4-flash",
            current_provider="zai",
            current_model="glm-4-flash",
            current_base_url="https://api.z.ai/api/paas/v4",
            current_api_key="sk-zai",
            user_providers={},
        )

        assert result.success is True

    def test_builtin_anthropic_still_works(self, monkeypatch):
        """Anthropic (built-in) should still work normally."""
        result = switch_model(
            raw_input="claude-sonnet-4.6",
            current_provider="anthropic",
            current_model="claude-sonnet-4.6",
            current_base_url="https://api.anthropic.com",
            current_api_key="sk-ant",
            user_providers={},
        )

        assert result.success is True


class TestIsCustomGuardLogic:
    """Verify the get_provider() check correctly identifies non-built-in providers."""

    def test_sensenova_is_not_builtin(self):
        from hermes_cli.providers import get_provider
        assert get_provider("sensenova") is None

    def test_openrouter_is_builtin(self):
        from hermes_cli.providers import get_provider
        assert get_provider("openrouter") is not None

    def test_deepseek_is_builtin(self):
        from hermes_cli.providers import get_provider
        assert get_provider("deepseek") is not None

    def test_zai_is_builtin(self):
        from hermes_cli.providers import get_provider
        assert get_provider("zai") is not None

    def test_arbitrary_name_is_not_builtin(self):
        from hermes_cli.providers import get_provider
        assert get_provider("my-custom-endpoint") is None
        assert get_provider("some-random-provider") is None
        assert get_provider("totally-not-a-provider") is None
