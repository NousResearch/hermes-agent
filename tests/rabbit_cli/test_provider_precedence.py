"""Regression tests for #29285 — provider precedence in resolve_provider("auto").

Explicit user intent (config.yaml model.provider, env-var API keys) must win
over a stale logged-in OAuth `active_provider` in auth.json. Before the fix,
`active_provider` sat above the env/config checks and silently overrode an
explicit choice — e.g. a user OAuth-logged-into Anthropic but with
OPENAI_API_KEY exported (or model.provider set) got routed to Anthropic.
"""
import pytest

from rabbit_cli.auth import resolve_provider, AuthError


def _login(monkeypatch, provider_id):
    """Simulate a logged-in OAuth active_provider in auth.json."""
    monkeypatch.setattr("rabbit_cli.auth._load_auth_store",
                        lambda: {"active_provider": provider_id})
    monkeypatch.setattr("rabbit_cli.auth.get_auth_status",
                        lambda p: {"logged_in": p == provider_id})


def _config(monkeypatch, model_cfg):
    monkeypatch.setattr("rabbit_cli.config.load_config", lambda: {"model": model_cfg})


def _no_aws(monkeypatch):
    # Neutralize any ambient AWS creds so Bedrock auto-detect can't interfere.
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: False)


def _clear_provider_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GLM_API_KEY", "ZAI_API_KEY",
                "KIMI_API_KEY", "MINIMAX_API_KEY", "RABBIT_INFERENCE_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


class TestProviderPrecedence:
    def test_config_provider_beats_stale_oauth(self, monkeypatch):
        """config.yaml model.provider wins over a logged-in OAuth active_provider."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")           # stale OAuth login
        _config(monkeypatch, {"provider": "zai", "default": "glm-4.6"})
        assert resolve_provider("auto") == "zai"

    def test_env_key_beats_stale_oauth(self, monkeypatch):
        """An exported provider API key wins over a logged-in OAuth active_provider."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {"default": "some-model"})  # dict, NO provider key
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        assert resolve_provider("auto") == "openai-api"


    def test_oauth_used_as_last_resort(self, monkeypatch):
        """With NO config provider and NO env keys, the logged-in OAuth provider
        is still used (it's the last-resort fallback, not removed)."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {})  # empty model config, no provider
        assert resolve_provider("auto") == "anthropic"




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


def _logged_out(monkeypatch):
    monkeypatch.setattr("rabbit_cli.auth._load_auth_store", lambda: {})
    monkeypatch.setattr("rabbit_cli.auth.get_auth_status", lambda p: {"logged_in": False})


class TestImplicitHostCredentialPrecedence:
    """A leftover ~/.aws profile must not pre-empt explicit intent, and ambient AWS
    credentials carry inference when nothing explicit exists (the free-tier rung
    that once sat above this chain was removed with the hosted service)."""

    @pytest.mark.parametrize("env_key, login, expected", [
        (None, None, "bedrock"),                        # implicit AWS chain answers
        ("OPENAI_API_KEY", None, "openai-api"),         # env key still wins
        (None, "anthropic", "anthropic"),               # a sign-in still wins
    ])
    def test_implicit_bedrock_chain_precedence(self, monkeypatch, env_key, login, expected):
        _clear_provider_env(monkeypatch)
        _config(monkeypatch, "")
        if login:
            _login(monkeypatch, login)
        else:
            _logged_out(monkeypatch)
        if env_key:
            monkeypatch.setenv(env_key, "sk-test-key")
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: True)
        assert resolve_provider("auto") == expected
