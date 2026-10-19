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
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda *_a, **_kw: False)


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


@pytest.fixture
def portal(monkeypatch, tmp_path):
    from tests.hermes_cli.anon_portal import install_portal
    return install_portal(monkeypatch, tmp_path)


def _never_mint(monkeypatch):
    """The resolver is a READ: any call into the creator from inside it is a bug."""
    monkeypatch.setattr("hermes_cli.anon_auth.ensure_portal_identity",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("resolve_provider must not mint")))


def _claimed_trial_with_aws(monkeypatch):
    """A claimed welcome trial on the desktop's local backend, then a leftover ~/.aws profile."""
    from tests.hermes_cli.anon_portal import claimed_trial
    _clear_provider_env(monkeypatch)
    claimed_trial(monkeypatch)
    _never_mint(monkeypatch)
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda *_a, **_kw: True)


class TestFreeTierBeatsImplicitHostCredentials:
    """NS-829: a leftover ~/.aws profile must not pre-empt the welcome trial on a fresh install.

    A claimed trial on the desktop's local backend sits above the implicit Bedrock chain; explicit
    intent still beats it; off the desktop backend a claimed guest never resolves to nous. The
    resolver never creates the identity."""

    def test_a_claimed_trial_on_the_desktop_backend_beats_the_bedrock_chain(self, portal, monkeypatch):
        _claimed_trial_with_aws(monkeypatch)
        assert resolve_provider("auto") == "nous"

    @pytest.mark.parametrize("env_key, config_provider, expected", [
        ("OPENAI_API_KEY", None, "openai-api"),  # an exported key wins
        (None, "zai", "zai"),                     # a config pin wins
    ], ids=["env-key", "config-provider"])
    def test_explicit_providers_beat_a_claimed_trial(self, portal, monkeypatch, env_key, config_provider,
                                                     expected):
        _claimed_trial_with_aws(monkeypatch)
        if env_key:
            monkeypatch.setenv(env_key, "sk-test-key")
        if config_provider:
            _config(monkeypatch, {"provider": config_provider, "default": "some-model"})
        assert resolve_provider("auto") == expected

    def test_on_the_cli_surface_a_claimed_guest_never_resolves_to_nous(self, portal, monkeypatch):
        _claimed_trial_with_aws(monkeypatch)
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)
        monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
        assert resolve_provider("auto") == "bedrock"
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda *_a, **_kw: False)
        with pytest.raises(AuthError):
            resolve_provider("auto")

    def test_skip_free_tier_answers_what_else_would_carry_inference(self, portal, monkeypatch):
        """With the free tier hidden, a claimed trial is not an answer and the ladder falls through
        to the next real rung."""
        _claimed_trial_with_aws(monkeypatch)
        monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda *_a, **_kw: False)
        assert resolve_provider("auto") == "nous"
        with pytest.raises(AuthError):
            resolve_provider("auto", skip_free_tier=True)
