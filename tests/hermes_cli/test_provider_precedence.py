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


def _config(monkeypatch, model_cfg, **extra):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": model_cfg, **extra})


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

    def test_config_runtime_provider_counts_as_inference_route(self, monkeypatch):
        """Dashboard readiness accepts custom routes but keeps the auto sentinel unresolved."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _logged_out(monkeypatch)
        _config(monkeypatch, {
            "provider": "custom", "default": "local-model", "base_url": "http://127.0.0.1:8080/v1",
        })
        assert resolve_provider("auto", skip_free_tier=True) == "custom"

        _config(
            monkeypatch,
            {"provider": "custom:llama-local", "default": "local-model"},
            providers={"llama-local": {"base_url": "http://127.0.0.1:8080/v1"}},
        )
        assert resolve_provider("auto", skip_free_tier=True) == "custom:llama-local"

        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {"provider": "auto", "default": "some-model"})
        assert resolve_provider("auto") == "anthropic"

    @pytest.mark.parametrize("model_cfg", [
        {"provider": "custom", "default": "local-model"},
        {"provider": "custom:missing", "default": "local-model"},
    ])
    def test_incomplete_custom_provider_is_not_an_inference_route(self, monkeypatch, model_cfg):
        """Bootstrap readiness rejects custom identities that runtime resolution cannot build."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _logged_out(monkeypatch)
        _config(monkeypatch, model_cfg, providers={})

        with pytest.raises(AuthError, match="No inference provider configured"):
            resolve_provider("auto", skip_free_tier=True)

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
    """Free tier switch + whether a free-tier identity already exists. The resolver is a READ: any
    call into the creator from inside it is a bug, so the stub fails loudly."""
    monkeypatch.setattr("hermes_cli.anon_auth.guest_enabled", lambda: on)
    monkeypatch.setattr("hermes_cli.anon_auth.has_guest", lambda: identity)
    monkeypatch.setattr("hermes_cli.anon_auth.ensure_portal_identity",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("resolve_provider must not mint")))


class TestFreeTierBeatsImplicitHostCredentials:
    """NS-829: a leftover ~/.aws profile must not pre-empt the free tier on a fresh install.

    The ladder: explicit intent still wins, an EXISTING free-tier identity sits above the implicit
    Bedrock chain, the free tier off (or its identity absent) restores Bedrock. The resolver never
    creates the identity; the boot bootstrap does, before any turn asks."""

    @pytest.mark.parametrize("free_tier_on, identity, env_key, login, expected", [
        (True, True, None, None, "nous"),                 # existing identity beats the AWS chain
        (True, False, None, None, "bedrock"),             # no identity yet: Bedrock, nothing minted
        (False, True, None, None, "bedrock"),             # free tier off: Bedrock as before
        (True, True, "OPENAI_API_KEY", None, "openrouter"),  # env key still wins
        (True, True, None, "anthropic", "anthropic"),        # a sign-in still wins
    ])
    def test_free_tier_sits_above_the_bedrock_chain(self, monkeypatch, free_tier_on, identity,
                                                     env_key, login, expected):
        _clear_provider_env(monkeypatch)
        _config(monkeypatch, "")
        if login:
            _login(monkeypatch, login)
        else:
            _logged_out(monkeypatch)
        if env_key:
            monkeypatch.setenv(env_key, "sk-test-key")
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: True)
        _free_tier(monkeypatch, on=free_tier_on, identity=identity)
        assert resolve_provider("auto") == expected

    def test_skip_free_tier_answers_what_else_would_carry_inference(self, monkeypatch):
        """The bootstrap's question: with the free tier hidden, an existing identity is not an
        answer and the ladder falls through to the next real rung."""
        _clear_provider_env(monkeypatch)
        _config(monkeypatch, "")
        _logged_out(monkeypatch)
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: False)
        _free_tier(monkeypatch, on=True, identity=True)
        assert resolve_provider("auto") == "nous"
        with pytest.raises(AuthError):
            resolve_provider("auto", skip_free_tier=True)
