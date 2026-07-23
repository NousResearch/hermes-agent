"""Credential-scope isolation proof for resolve_anthropic_token() (#51603).

resolve_anthropic_token() is the fallback path agent_init.py uses when no
explicit api_key is configured for the runtime provider. Every other
credential reader in the multiplex gateway (hermes_cli.runtime_provider,
tools.mcp_tool) resolves through agent.secret_scope.get_secret() so a
profile's own .env wins over whatever happens to be sitting in process-global
os.environ. resolve_anthropic_token() was never migrated and read os.getenv
directly for ANTHROPIC_TOKEN / CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY,
so a multiplexed gateway serving Profile A could silently use Profile B's (or
the process default's) Anthropic credential. See test_multiplex_credential_
isolation.py for the equivalent proof already in place for runtime_provider.
"""
import pytest

from agent import secret_scope as ss
from agent import anthropic_adapter


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ss.set_multiplex_active(False)
    # Isolate resolve_anthropic_token() from any real Claude Code credential
    # file / keychain / credential_pool state on the machine running the
    # tests — sources 3 and 4 in the priority order are out of scope here.
    monkeypatch.setattr(anthropic_adapter, "read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(anthropic_adapter, "_resolve_anthropic_pool_token", lambda: None)
    yield
    ss.set_multiplex_active(False)


class TestResolveAnthropicTokenUsesScope:
    def test_scoped_api_key_used_over_environ(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-WRONG-OTHER-PROFILE")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-PROFILE-A"})
        try:
            assert anthropic_adapter.resolve_anthropic_token() == "sk-ant-api-PROFILE-A"
        finally:
            ss.reset_secret_scope(tok)

    def test_two_profiles_return_different_keys(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-SHARED-ENVIRON-WRONG")
        ss.set_multiplex_active(True)

        tok_a = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-A"})
        try:
            assert anthropic_adapter.resolve_anthropic_token() == "sk-ant-api-A"
        finally:
            ss.reset_secret_scope(tok_a)

        tok_b = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-B"})
        try:
            assert anthropic_adapter.resolve_anthropic_token() == "sk-ant-api-B"
        finally:
            ss.reset_secret_scope(tok_b)

    def test_anthropic_token_env_does_not_shadow_scoped_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat-WRONG-OTHER-PROFILE")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-PROFILE-A"})
        try:
            assert anthropic_adapter.resolve_anthropic_token() == "sk-ant-api-PROFILE-A"
        finally:
            ss.reset_secret_scope(tok)

    def test_anthropic_token_in_scope_is_used(self, monkeypatch):
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_TOKEN": "sk-ant-oat-PROFILE-A"})
        try:
            assert anthropic_adapter.resolve_anthropic_token() == "sk-ant-oat-PROFILE-A"
        finally:
            ss.reset_secret_scope(tok)

    def test_cc_oauth_env_does_not_shadow_scoped_api_key(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-cc-WRONG-OTHER-PROFILE")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-ant-api-PROFILE-A"})
        try:
            assert anthropic_adapter.resolve_anthropic_token() == "sk-ant-api-PROFILE-A"
        finally:
            ss.reset_secret_scope(tok)

    def test_unscoped_call_in_multiplex_mode_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-leak")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            anthropic_adapter.resolve_anthropic_token()

    def test_unscoped_call_outside_multiplex_reads_environ(self, monkeypatch):
        # multiplex inactive (single-profile / non-gateway callers, e.g. the
        # `hermes` CLI itself) must keep behaving exactly like plain os.getenv.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-legacy")
        assert anthropic_adapter.resolve_anthropic_token() == "sk-ant-api-legacy"
