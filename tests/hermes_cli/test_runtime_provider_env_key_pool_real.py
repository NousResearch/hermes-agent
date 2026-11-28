"""``env_key_pool`` against a real ``load_pool`` on the test's sandboxed HERMES_HOME (PR #106833)."""

import pytest

from hermes_cli import runtime_provider as rp

KEY = "real-env-key-0123456789abcdef"
FOUNDRY = "https://res.services.ai.azure.com/anthropic"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "AZURE_ANTHROPIC_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)


@pytest.mark.parametrize("provider,url", [("anthropic", "https://api.anthropic.com"),
                                          ("openrouter", "https://openrouter.ai/api/v1")])
def test_env_key_member_gets_the_pool_non_member_does_not(provider, url):
    pool = rp.env_key_pool(provider, KEY, url)
    assert pool is not None and pool.provider == provider
    assert rp.env_key_pool(provider, "not-a-member", url) is None


def test_foundry_session_never_gets_the_unfiltered_anthropic_pool(monkeypatch):
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "anthropic", "default": "claude-opus-4-7", "base_url": FOUNDRY})
    resolved = rp.resolve_runtime_provider(requested="anthropic")
    assert resolved["base_url"].startswith(FOUNDRY) and resolved["api_key"] == KEY
    assert resolved["credential_pool"] is None  # main attaches only Foundry-bound rows (#154242)
