"""Regression tests for #29285 — provider precedence in resolve_provider("auto").

Explicit user intent (config.yaml model.provider, env-var API keys) must win
over a stale logged-in OAuth `active_provider` in auth.json. Before the fix,
`active_provider` sat above the env/config checks and silently overrode an
explicit choice — e.g. a user OAuth-logged-into Anthropic but with
OPENAI_API_KEY exported (or model.provider set) got routed to Anthropic.
"""
import pytest

from hermes_cli.auth import resolve_provider, AuthError


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
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GLM_API_KEY", "ZAI_API_KEY",
                "KIMI_API_KEY", "MINIMAX_API_KEY", "HERMES_INFERENCE_PROVIDER"):
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
        import logging
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


def _logged_out(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})
    monkeypatch.setattr("hermes_cli.auth.get_auth_status", lambda p: {"logged_in": False})


def _free_tier(monkeypatch, *, on=True, identity=False):
    """Free tier switch + whether a free-tier identity already exists; counts mints."""
    monkeypatch.setattr("hermes_cli.anon_auth.guest_enabled", lambda: on)
    monkeypatch.setattr("hermes_cli.anon_auth.has_guest", lambda: identity)
    calls = {"mint": 0}

    def _ensure(*, blocking=True, timeout_seconds=None):
        calls["mint"] += 1
        return {"auth_method": "anonymous"} if on else None

    monkeypatch.setattr("hermes_cli.anon_auth.ensure_portal_identity", _ensure)
    return calls


class TestFreeTierBeatsImplicitHostCredentials:
    """NS-829: a leftover ~/.aws profile must not pre-empt the free tier on a fresh install.

    Two invariants. (1) The ladder: explicit intent still wins, the free tier sits above the
    implicit Bedrock chain, and the free tier off restores Bedrock. (2) A failed mint, whether it
    returns None or raises, falls through to Bedrock."""

    @pytest.mark.parametrize("free_tier_on, identity, env_key, login, expected, mints", [
        (True, True, None, None, "nous", 0),          # existing identity beats the AWS chain
        (True, False, None, None, "nous", 1),         # fresh install mints before Bedrock
        (False, False, None, None, "bedrock", 0),     # free tier off: Bedrock as before
        (True, True, "OPENAI_API_KEY", None, "openrouter", 0),  # env key still wins
        (True, True, None, "anthropic", "anthropic", 0),        # a sign-in still wins
    ])
    def test_free_tier_sits_above_the_bedrock_chain(self, monkeypatch, free_tier_on, identity,
                                                     env_key, login, expected, mints):
        _clear_provider_env(monkeypatch)
        _config(monkeypatch, "")
        if login:
            _login(monkeypatch, login)
        else:
            _logged_out(monkeypatch)
        if env_key:
            monkeypatch.setenv(env_key, "sk-test-key")
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: True)
        calls = _free_tier(monkeypatch, on=free_tier_on, identity=identity)
        assert resolve_provider("auto") == expected
        assert calls["mint"] == mints

    @pytest.mark.parametrize("failure", ["returns_none", "raises"])
    def test_failed_mint_falls_through_to_bedrock(self, monkeypatch, failure):
        _clear_provider_env(monkeypatch)
        _config(monkeypatch, "")
        _logged_out(monkeypatch)
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: True)
        monkeypatch.setattr("hermes_cli.anon_auth.guest_enabled", lambda: True)
        monkeypatch.setattr("hermes_cli.anon_auth.has_guest", lambda: False)

        def _mint(**kw):
            if failure == "raises":
                raise RuntimeError("portal 429")
            return None

        monkeypatch.setattr("hermes_cli.anon_auth.ensure_portal_identity", _mint)
        assert resolve_provider("auto") == "bedrock"
