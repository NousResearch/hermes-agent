"""Anthropic pool-vs-env precedence (issue #136800).

A borrowed Claude Code OAuth grant seeded into the anthropic credential pool
must not shadow an explicitly set ANTHROPIC_API_KEY: the pool rung runs before
the anthropic env rung in the provider ladder, so the pool rung yields when the
operator set an env key.
"""

import pytest

from hermes_cli import runtime_provider as rp


class _BorrowedClaudeCodeEntry:
    access_token = "oauth-token-from-claude-code"
    source = "claude_code"
    base_url = ""


class _BorrowedPool:
    def has_credentials(self):
        return True

    def select(self, **_kwargs):
        return _BorrowedClaudeCodeEntry()


@pytest.fixture
def _borrowed_pool_env(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _BorrowedPool())
    monkeypatch.setattr(rp, "_get_model_config", dict)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)


def test_anthropic_env_api_key_beats_borrowed_pool_oauth(monkeypatch, _borrowed_pool_env):
    """Explicit ANTHROPIC_API_KEY wins over the borrowed pool OAuth grant."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-explicit-key")

    resolved = rp.resolve_runtime_provider(requested="anthropic")

    assert resolved["provider"] == "anthropic"
    assert resolved["api_key"] == "sk-ant-explicit-key"
    assert resolved["source"] == "env"


def test_anthropic_borrowed_pool_oauth_still_used_without_env_key(monkeypatch, _borrowed_pool_env):
    """Without an explicit env key, the borrowed pool entry keeps resolving."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="anthropic")

    assert resolved["provider"] == "anthropic"
    assert resolved["api_key"] == "oauth-token-from-claude-code"
