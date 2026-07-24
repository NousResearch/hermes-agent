"""Test that AuthError triggers fallback provider resolution (#7230)."""

from unittest.mock import patch

import pytest


class TestResolveRuntimeAgentKwargsAuthFallback:
    """_resolve_runtime_agent_kwargs should try fallback on AuthError."""

    def test_auth_error_tries_fallback(self, tmp_path, monkeypatch):
        """When primary provider raises AuthError, fallback is attempted."""
        from hermes_cli import runtime_provider as rp
        from hermes_cli.auth import AuthError

        # Create a config with fallback
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  provider: openai-codex\n"
            "fallback_model:\n  provider: kimi-coding\n"
            "  model: k3\n"
            "  base_url: https://api.kimi.com/coding\n"
            "  api_mode: chat_completions\n"
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        calls = []
        real_resolve = rp.resolve_runtime_provider
        monkeypatch.setattr(
            rp,
            "_resolve_runtime_provider_impl",
            lambda **_kwargs: {
                "api_key": "fallback-key",
                "base_url": "https://api.kimi.com/coding",
                "provider": "kimi-coding",
                "api_mode": "anthropic_messages",
                "command": None,
                "args": None,
                "credential_pool": None,
            },
        )

        def _mock_resolve(**kwargs):
            calls.append(kwargs)
            # First call = primary path (gateway reads model.provider from
            # config.yaml internally; we simulate the auth failure here).
            # Second call = fallback path through the real runtime wrapper.
            if len(calls) == 1:
                raise AuthError("Codex token refresh failed with status 401")
            return real_resolve(**kwargs)

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "kimi-coding"
        assert result["api_key"] == "fallback-key"
        assert result["api_mode"] == "chat_completions"
        assert result["base_url"] == "https://api.kimi.com/coding/v1"
        assert len(calls) >= 2
        assert calls[1]["explicit_api_mode"] == "chat_completions"
        assert calls[1]["target_model"] == "k3"

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
