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

    def test_auth_error_manual_mode_does_not_auto_resolve_fallback(
        self, tmp_path, monkeypatch
    ):
        """Before an agent exists there is no interactive surface, so fail closed."""
        from hermes_cli.auth import AuthError

        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: openai-codex\n"
            "fallback:\n  auto_activate: false\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: meta-llama/llama-4-maverick\n"
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=AuthError("token expired"),
        ) as resolve:
            from gateway.run import _resolve_runtime_agent_kwargs

            with pytest.raises(RuntimeError):
                _resolve_runtime_agent_kwargs()

        resolve.assert_called_once()

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
