"""Test that AuthError triggers fallback provider resolution (#7230)."""

from unittest.mock import patch

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
            # First call = primary path (gateway reads model.provider from
            # config.yaml internally; we simulate the auth failure here).
            # Second call = fallback path with explicit_api_key + explicit_base_url
            # supplied by gateway from fallback_model config.
            if call_count["n"] == 1:
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
        assert calls[0] == {}
        assert [call.get("requested") for call in calls[1:]] == ["my-custom"]

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
        assert calls == [{}]

    def test_legacy_fallback_is_appended_after_fallback_providers(self, tmp_path, monkeypatch):
        """When both keys exist, the legacy entry still participates in resolution."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: anthropic/claude-sonnet-4.6\n"
            "fallback_model:\n"
            "  provider: nous\n"
            "  model: Hermes-4\n"
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        calls = []

        def _mock_resolve(**kwargs):
            requested = kwargs.get("requested")
            calls.append(requested)
            if requested == "openrouter":
                raise RuntimeError("openrouter unavailable")
            return {
                "api_key": "nous-key",
                "base_url": "https://portal.nousresearch.com/v1",
                "provider": "nous",
                "api_mode": "chat_completions",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _try_resolve_fallback_provider

            result = _try_resolve_fallback_provider()

        assert calls == ["openrouter", "nous"]
        assert result["provider"] == "nous"
        assert result["model"] == "Hermes-4"

    def test_empty_openrouter_fallback_is_skipped_real_config_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Real HERMES_HOME config resolution skips empty OpenRouter entries."""
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  provider: openrouter\n"
            "  default: openrouter/auto\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: anthropic/claude-sonnet-4.6\n"
            "  - provider: custom\n"
            "    model: gpt-4\n"
            "    base_url: https://api.example.com/v1\n"
            "    api_key: sk-valid-key\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr("gateway.run._hermes_home", hermes_home)
        for name in (
            "HERMES_INFERENCE_PROVIDER",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_BASE_URL",
            "OPENAI_BASE_URL",
            "CUSTOM_BASE_URL",
        ):
            monkeypatch.delenv(name, raising=False)

        from gateway.run import _resolve_runtime_agent_kwargs

        result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "custom"
        assert result["api_key"] == "sk-valid-key"
        assert result["base_url"] == "https://api.example.com/v1"
        assert result["model"] == "gpt-4"
