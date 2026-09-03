"""Regression: a suppressed ``ANTHROPIC_API_KEY`` must not shadow the pool.

``hermes auth remove anthropic <env-key-entry>`` records
``env:ANTHROPIC_API_KEY`` under ``suppressed_sources`` and tells the user
"Hermes will ignore ANTHROPIC_API_KEY". ``load_pool()`` honours that, but
``resolve_anthropic_token()`` read the variable straight from the environment
ahead of the pool, so every request went out authenticated with the removed
key (observed as ``HTTP 401: invalid x-api-key`` with a bogus value, and as
the Console "credit balance is too low" 400 with a real one) while the
healthy OAuth entry sat unused.
"""

import json

import pytest


def _write_store(tmp_path, *, suppressed: bool) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    store = {
        "version": 1,
        "credential_pool": {
            "anthropic": [
                {
                    "id": "oauth-1",
                    "label": "anthropic-oauth-1",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual:hermes_pkce",
                    "access_token": "pool-oauth-token",
                }
            ]
        },
    }
    if suppressed:
        store["suppressed_sources"] = {"anthropic": ["env:ANTHROPIC_API_KEY"]}
    (home / "auth.json").write_text(json.dumps(store))


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    # Keep the host's Claude Code login out of the resolver and the pool seed.
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials", lambda: None
    )
    return tmp_path


def test_suppressed_env_key_yields_to_pool_oauth(isolated_home):
    _write_store(isolated_home, suppressed=True)
    from agent.anthropic_credentials import resolve_anthropic_token

    assert resolve_anthropic_token() == "pool-oauth-token"


def test_unsuppressed_env_key_still_wins(isolated_home):
    _write_store(isolated_home, suppressed=False)
    from agent.anthropic_credentials import resolve_anthropic_token

    assert resolve_anthropic_token() == "sk-ant-env-key"
