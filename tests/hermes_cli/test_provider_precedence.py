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
    # ANTHROPIC_*/CLAUDE_CODE_OAUTH_TOKEN included: on a developer machine
    # with a live Claude Code login these are exported, and without clearing
    # them the auto-detect tests below pass or fail for the wrong reason.
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GLM_API_KEY", "ZAI_API_KEY",
                "KIMI_API_KEY", "MINIMAX_API_KEY", "HERMES_INFERENCE_PROVIDER",
                "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_OAUTH_TOKEN"):
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

    def test_provider_specific_env_key_beats_stale_oauth(self, monkeypatch):
        """A provider-specific env key (GLM) wins over a logged-in OAuth provider."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {})
        monkeypatch.setenv("GLM_API_KEY", "test-glm-key")
        assert resolve_provider("auto") == "zai"

    def test_oauth_used_as_last_resort(self, monkeypatch):
        """With NO config provider and NO env keys, the logged-in OAuth provider
        is still used (it's the last-resort fallback, not removed)."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")
        _config(monkeypatch, {})  # empty model config, no provider
        assert resolve_provider("auto") == "anthropic"

    def test_explicit_request_unaffected(self, monkeypatch):
        """An explicit requested provider short-circuits everything."""
        _clear_provider_env(monkeypatch)
        _login(monkeypatch, "anthropic")
        assert resolve_provider("zai") == "zai"

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

    def test_warns_when_env_key_preempts_oauth(self, monkeypatch, caplog):
        """When an exported API key preempts a logged-in OAuth provider, a WARN
        makes the silent routing switch visible (#29285)."""
        import logging
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        _login(monkeypatch, "anthropic")           # OAuth into anthropic
        _config(monkeypatch, {})
        monkeypatch.setenv("GLM_API_KEY", "test-glm-key")  # unrelated key present
        with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
            assert resolve_provider("auto") == "zai"
        assert any("preempting your" in r.message for r in caplog.records)

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


class TestImplicitOAuthTokenNotAutoSelected:
    """CLAUDE_CODE_OAUTH_TOKEN is written by Claude Code's own login, not by
    the user configuring Hermes. The is_provider_explicitly_configured
    doctrine (PR #7156) already excludes it from explicit-config detection;
    the auto-detect scan must apply the same posture — ambient presence must
    not route inference onto the Anthropic HTTP lane (metered overage per
    Anthropic's third-party policy; see #58546, #70039)."""

    @staticmethod
    def _no_oauth_login(monkeypatch):
        monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})
        monkeypatch.setattr("hermes_cli.auth.get_auth_status",
                            lambda p: {"logged_in": False})

    def test_token_alone_does_not_auto_select_anthropic(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        self._no_oauth_login(monkeypatch)
        _config(monkeypatch, {})
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ambient")
        with pytest.raises(AuthError) as exc_info:
            resolve_provider("auto")
        # The error must tell the user the token was seen and how to use it
        # deliberately — silence here would just look broken.
        assert "CLAUDE_CODE_OAUTH_TOKEN" in str(exc_info.value)

    def test_explicit_anthropic_key_still_auto_selects(self, monkeypatch):
        """No over-exclusion: ANTHROPIC_API_KEY is a user-exported credential
        and keeps auto-selecting anthropic."""
        _clear_provider_env(monkeypatch)
        _no_aws(monkeypatch)
        self._no_oauth_login(monkeypatch)
        _config(monkeypatch, {})
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
        assert resolve_provider("auto") == "anthropic"

    def test_explicit_provider_still_uses_the_token(self, monkeypatch):
        """Explicit selection is untouched: provider=anthropic with only the
        Claude Code token resolves to anthropic (credential resolution then
        finds the token via get_anthropic_key as before)."""
        _clear_provider_env(monkeypatch)
        self._no_oauth_login(monkeypatch)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ambient")
        assert resolve_provider("anthropic") == "anthropic"
