"""Regression tests for #29285 — provider precedence in resolve_provider("auto").

Explicit user intent (config.yaml model.provider, env-var API keys) must win
over a stale logged-in OAuth `active_provider` in auth.json. Before the fix,
`active_provider` sat above the env/config checks and silently overrode an
explicit choice — e.g. a user OAuth-logged-into Anthropic but with
OPENAI_API_KEY exported (or model.provider set) got routed to Anthropic.
"""
import logging

import pytest

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, AuthError


def _login(monkeypatch, provider_id):
    """Simulate a logged-in OAuth active_provider in auth.json."""
    monkeypatch.setattr("hermes_cli.auth._load_auth_store",
                        lambda: {"active_provider": provider_id})
    monkeypatch.setattr("hermes_cli.auth.get_auth_status",
                        lambda p: {"logged_in": p == provider_id})


def _config(monkeypatch, model_cfg):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": model_cfg})


def _no_aws(monkeypatch):
    # Neutralize any ambient AWS creds so Bedrock auto-detect can't interfere.
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: False)


def _clear_provider_env(monkeypatch):
    provider_key_vars = {
        env_var
        for provider in PROVIDER_REGISTRY.values()
        for env_var in provider.api_key_env_vars
    }
    for var in provider_key_vars | {
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "HERMES_INFERENCE_PROVIDER",
    }:
        monkeypatch.delenv(var, raising=False)


class TestProviderPrecedence:
    def test_config_provider_beats_stale_oauth(self, monkeypatch):
        """config.yaml model.provider wins over a logged-in OAuth active_provider."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")           # stale OAuth login
        _config(monkeypatch, {"provider": "zai", "default": "glm-4.6"})
        assert resolve_provider("auto") == "zai"

    def test_config_provider_alias_is_normalized(self, monkeypatch):
        """config.yaml model.provider accepts the same aliases as explicit input."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _config(monkeypatch, {"provider": " GO "})
        assert resolve_provider("auto") == "opencode-go"

    def test_config_read_failure_is_warning(self, monkeypatch, caplog):
        """A config read failure must be visible without debug logging enabled."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: (_ for _ in ()).throw(OSError("unreadable config")),
        )

        with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
            try:
                resolve_provider("auto")
            except AuthError:
                pass

        assert any(
            "Could not read config.yaml model.provider" in record.message
            for record in caplog.records
        )

    def test_warns_when_multiple_provider_api_keys_are_detected(
        self, monkeypatch, caplog
    ):
        """Multiple provider keys warn while registry order selects the provider."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _config(monkeypatch, {})
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
        monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-go-key")

        with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
            assert resolve_provider("auto") == "alibaba"

        warning_messages = [
            record.message
            for record in caplog.records
            if "Multiple provider API keys detected" in record.message
        ]
        assert len(warning_messages) == 1
        assert "alibaba (DASHSCOPE_API_KEY)" in warning_messages[0]
        assert "opencode-go (OPENCODE_GO_API_KEY)" in warning_messages[0]

    def test_shared_provider_api_key_does_not_emit_multi_key_warning(
        self, monkeypatch, caplog
    ):
        """A shared env var is one candidate even when two providers accept it."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _config(monkeypatch, {})
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")

        with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
            assert resolve_provider("auto") == "alibaba"

        assert not any(
            "Multiple provider API keys detected" in record.message
            for record in caplog.records
        )

    def test_single_provider_dual_env_vars_not_multi_key(
        self, monkeypatch, caplog
    ):
        """A provider with multiple env vars (e.g. GLM_API_KEY + ZAI_API_KEY)
        counts as one candidate, not two."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _config(monkeypatch, {})
        monkeypatch.setenv("GLM_API_KEY", "test-glm-key")
        monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")

        with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
            assert resolve_provider("auto") == "zai"

        assert not any(
            "Multiple provider API keys detected" in record.message
            for record in caplog.records
        )

    def test_env_key_beats_stale_oauth(self, monkeypatch):
        """An exported provider API key wins over a logged-in OAuth active_provider."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {"default": "some-model"})  # dict, NO provider key
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        assert resolve_provider("auto") == "openrouter"


    def test_oauth_used_as_last_resort(self, monkeypatch):
        """With NO config provider and NO env keys, the logged-in OAuth provider
        is still used (it's the last-resort fallback, not removed)."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {})  # empty model config, no provider
        assert resolve_provider("auto") == "anthropic"


    def test_warns_on_silent_oauth_fallthrough(self, monkeypatch, caplog):
        """A populated model dict lacking `provider` that falls through to OAuth
        emits a WARN so the silent override is visible (#29285)."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {"default": "claude-x"})  # populated, no provider
        with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
            assert resolve_provider("auto") == "anthropic"
        assert any("no `provider` key" in r.message for r in caplog.records)


    def test_openrouter_pool_beats_stale_oauth(self, monkeypatch):
        """An OpenRouter credential-pool entry (no env var) wins over a logged-in
        OAuth provider — the pool rung sits above OAuth (#42130 + #29285)."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {})

        class _Pool:
            def has_credentials(self):
                return True

        monkeypatch.setattr("agent.credential_pool.load_pool", lambda name: _Pool())
        assert resolve_provider("auto") == "openrouter"
