"""Regression tests for #63737: sk-ant-oat pool entries are OAuth."""

import json
from pathlib import Path

from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)




def test_anthropic_real_api_key_unchanged():
    entry = PooledCredential.from_dict(
        "anthropic",
        {"auth_type": "api_key", "access_token": "sk-ant-api-EXAMPLE"},
    )
    assert entry.auth_type == AUTH_TYPE_API_KEY








def test_load_heals_legacy_row_and_exposes_it_to_resolver(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: None,
    )
    token = "sk-ant-oat-legacy-manual"
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps({
        "version": 1,
        "credential_pool": {
            "anthropic": [{
                "id": "legacy-oat",
                "label": "Legacy setup token",
                "auth_type": AUTH_TYPE_API_KEY,
                "priority": 0,
                "source": "manual",
                "access_token": token,
            }],
        },
    }))

    from agent.anthropic_adapter import resolve_anthropic_token
    from agent.credential_pool import load_pool

    entry = load_pool("anthropic").entries()[0]
    persisted = json.loads(auth_file.read_text())
    assert entry.auth_type == AUTH_TYPE_OAUTH
    assert persisted["credential_pool"]["anthropic"][0]["auth_type"] == AUTH_TYPE_OAUTH
    assert resolve_anthropic_token() == token


