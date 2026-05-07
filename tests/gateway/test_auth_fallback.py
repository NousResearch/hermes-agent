"""Test that AuthError triggers fallback provider resolution (#7230)."""

import os
from unittest.mock import patch, MagicMock

import pytest


class TestResolveRuntimeAgentKwargsAuthFallback:
    """_resolve_runtime_agent_kwargs should try fallback on AuthError."""

    def test_auth_error_tries_fallback(self, tmp_path, monkeypatch):
        """When primary provider raises AuthError, fallback is attempted."""
        from hermes_cli.auth import AuthError

        # Create a config with fallback
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  provider: openai-codex\n"
            "fallback_model:\n  provider: openrouter\n"
            "  model: meta-llama/llama-4-maverick\n"
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        call_count = {"n": 0}

        def _mock_resolve(**kwargs):
            call_count["n"] += 1
            requested = kwargs.get("requested", "")
            if requested and "codex" in str(requested).lower():
                raise AuthError("Codex token refresh failed with status 401")
            return {
                "api_key": "fallback-key",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "openai_chat",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openai-codex")

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert result["api_key"] == "fallback-key"
        # Should have been called at least twice (primary + fallback)
        assert call_count["n"] >= 2

    def test_auth_error_no_fallback_raises(self, tmp_path, monkeypatch):
        """When primary fails and no fallback configured, RuntimeError is raised."""
        from hermes_cli.auth import AuthError

        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  provider: openai-codex\n")

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openai-codex")

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=AuthError("token expired"),
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            with pytest.raises(RuntimeError):
                _resolve_runtime_agent_kwargs()

    def test_empty_openrouter_key_tries_fallback(self, tmp_path, monkeypatch):
        """When primary OpenRouter runtime resolves with an empty key, gateway should try fallback."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_providers:\n"
            "  - provider: my-custom\n"
            "    model: gpt-4\n"
            "    base_url: https://api.example.com/v1\n"
            "    api_key: sk-valid-key\n",
            encoding="utf-8",
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")

        calls = []

        def _mock_resolve(**kwargs):
            calls.append(kwargs)
            if kwargs.get("requested") == "my-custom":
                return {
                    "api_key": "sk-valid-key",
                    "base_url": "https://api.example.com/v1",
                    "provider": "custom",
                    "api_mode": "chat_completions",
                    "command": None,
                    "args": [],
                    "credential_pool": None,
                }
            return {
                "api_key": "",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs

            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "custom"
        assert result["api_key"] == "sk-valid-key"
        assert [call.get("requested") for call in calls] == ["openrouter", "my-custom"]

    def test_empty_openrouter_key_without_fallback_raises_clear_error(self, tmp_path, monkeypatch):
        """When primary OpenRouter runtime resolves with an empty key and no fallback exists, raise a clear error."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        ):
            from gateway.run import _resolve_runtime_agent_kwargs

            with pytest.raises(RuntimeError, match="OpenRouter runtime resolved without a usable API key"):
                _resolve_runtime_agent_kwargs()

    def test_empty_non_openrouter_key_does_not_trigger_gateway_fallback(self, tmp_path, monkeypatch):
        """Empty keys for non-OpenRouter runtimes should not hit this gateway-specific fallback path."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_providers:\n"
            "  - provider: my-custom\n"
            "    model: gpt-4\n"
            "    base_url: https://api.example.com/v1\n"
            "    api_key: sk-valid-key\n",
            encoding="utf-8",
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "custom")

        calls = []

        def _mock_resolve(**kwargs):
            calls.append(kwargs)
            return {
                "api_key": "",
                "base_url": "https://api.example.com/v1",
                "provider": "custom",
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs

            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "custom"
        assert result["api_key"] == ""
        assert calls == [{"requested": "custom"}]
