"""A benched env ANTHROPIC_TOKEN must not hide a healthy pooled account.

Session start resolves the Anthropic token with the requested model. When the env
token is benched for that model (e.g. its account hit the weekly cap), the resolver
returned None and the session fell through to the fallback provider even though
another pooled account was available for the same model.
"""
import json
import time

import pytest

CAPPED = "sk-ant-oat01-synthetic-capped-account-0000"
HEALTHY = "sk-ant-oat01-synthetic-healthy-account-0000"
MODEL = "claude-opus-5-5"
OTHER = "claude-haiku-5-5"


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]

    def write(healthy_cooldowns):
        (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [
            {"id": "capped", "label": "capped", "auth_type": "oauth", "priority": 0, "source": "manual",
             "access_token": CAPPED, "model_cooldowns": {MODEL: time.time() + 4 * 86400}},
            {"id": "healthy", "label": "healthy", "auth_type": "oauth", "priority": 1, "source": "manual",
             "access_token": HEALTHY, "model_cooldowns": healthy_cooldowns},
        ]}}))
    monkeypatch.setenv("ANTHROPIC_TOKEN", CAPPED)
    return write


def test_benched_env_token_yields_a_pooled_account_free_for_the_model(store):
    from agent.anthropic_credentials import resolve_anthropic_token
    store({})
    assert resolve_anthropic_token(model=MODEL) == HEALTHY


def test_env_token_still_wins_when_it_is_not_benched(store):
    from agent.anthropic_credentials import resolve_anthropic_token
    store({})
    assert resolve_anthropic_token(model=OTHER) == CAPPED
    assert resolve_anthropic_token() == CAPPED


def test_none_when_every_pooled_account_is_benched_for_the_model(store):
    from agent.anthropic_credentials import resolve_anthropic_token
    store({MODEL: time.time() + 300})
    assert resolve_anthropic_token(model=MODEL) is None


def test_a_bench_on_another_model_does_not_hide_the_pooled_account(store):
    from agent.anthropic_credentials import resolve_anthropic_token
    store({OTHER: time.time() + 300})
    assert resolve_anthropic_token(model=MODEL) == HEALTHY
