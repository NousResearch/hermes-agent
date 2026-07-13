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


class TestGatewayFallbackEnvExpansion:
    def test_load_fallback_model_expands_legacy_env_api_key(self, tmp_path, monkeypatch):
        """Gateway fallback loader should expand ${VAR} refs in legacy fallback_model."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_model:\n"
            "  provider: custom\n"
            "  model: deepseek-chat\n"
            "  api_key: ${HERMES_TEST_FB_KEY}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_TEST_FB_KEY", "secret-123")
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        from gateway.run import GatewayRunner

        fb = GatewayRunner._load_fallback_model()

        assert fb[0]["api_key"] == "secret-123"

    def test_try_resolve_fallback_provider_expands_env_api_key(self, tmp_path, monkeypatch):
        """Gateway runtime fallback resolution should expand ${VAR} refs before provider lookup."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_providers:\n"
            "  - provider: custom\n"
            "    model: deepseek-chat\n"
            "    base_url: https://example.invalid/v1\n"
            "    api_key: ${HERMES_TEST_FB_KEY}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_TEST_FB_KEY", "secret-456")
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        seen = {}

        def _mock_resolve(**kwargs):
            seen["explicit_api_key"] = kwargs.get("explicit_api_key")
            return {
                "api_key": kwargs.get("explicit_api_key"),
                "base_url": kwargs.get("explicit_base_url"),
                "provider": kwargs.get("requested"),
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _try_resolve_fallback_provider

            fb = _try_resolve_fallback_provider()

        assert seen["explicit_api_key"] == "secret-456"
        assert fb["api_key"] == "secret-456"

    def test_refresh_fallback_model_expands_legacy_env_api_key(self, tmp_path, monkeypatch):
        """Gateway refresh path should expand ${VAR} refs in legacy fallback_model."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_model:\n"
            "  provider: custom\n"
            "  model: deepseek-chat\n"
            "  api_key: ${HERMES_TEST_FB_KEY}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_TEST_FB_KEY", "secret-refresh-legacy")
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._fallback_model = None

        fb = runner._refresh_fallback_model()

        assert fb[0]["api_key"] == "secret-refresh-legacy"
        assert runner._fallback_model == fb

    def test_refresh_fallback_model_expands_list_env_api_key(self, tmp_path, monkeypatch):
        """Gateway refresh path should expand ${VAR} refs in fallback_providers."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_providers:\n"
            "  - provider: custom\n"
            "    model: deepseek-chat\n"
            "    base_url: https://example.invalid/v1\n"
            "    api_key: ${HERMES_TEST_FB_KEY}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_TEST_FB_KEY", "secret-refresh-list")
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._fallback_model = None

        fb = runner._refresh_fallback_model()

        assert fb[0]["api_key"] == "secret-refresh-list"
        assert fb[0]["base_url"] == "https://example.invalid/v1"
        assert runner._fallback_model == fb
