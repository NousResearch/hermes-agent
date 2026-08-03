"""Regression tests for #63737: sk-ant-oat pool entries are OAuth."""

import json
from pathlib import Path

from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)


def test_manual_anthropic_oat_normalized_to_oauth():
    # Pool auth_type gates OAuth-only resolver and refresh paths.
    entry = PooledCredential.from_dict(
        "anthropic",
        {
            "label": "MainKey",
            "source": "manual",
            "auth_type": "api_key",
            "access_token": "sk-ant-oat-EXAMPLE",
        },
    )
    assert entry.auth_type == AUTH_TYPE_OAUTH


def test_anthropic_real_api_key_unchanged():
    entry = PooledCredential.from_dict(
        "anthropic",
        {"auth_type": "api_key", "access_token": "sk-ant-api-EXAMPLE"},
    )
    assert entry.auth_type == AUTH_TYPE_API_KEY


def test_anthropic_admin_key_unchanged():
    entry = PooledCredential.from_dict(
        "anthropic",
        {"auth_type": "api_key", "access_token": "sk-ant-admin-EXAMPLE"},
    )
    assert entry.auth_type == AUTH_TYPE_API_KEY


def test_non_anthropic_provider_unchanged():
    entry = PooledCredential.from_dict(
        "openrouter",
        {"auth_type": "api_key", "access_token": "sk-ant-oat-WHATEVER"},
    )
    assert entry.auth_type == AUTH_TYPE_API_KEY


def test_add_entry_normalizes_before_persisting(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    pool = CredentialPool("anthropic", [])
    entry = pool.add_entry(PooledCredential(
        provider="anthropic",
        id="manual-oat",
        label="Manual setup token",
        auth_type=AUTH_TYPE_API_KEY,
        priority=0,
        source="manual",
        access_token="sk-ant-oat-manual-entry",
    ))

    persisted = json.loads((hermes_home / "auth.json").read_text())
    assert entry.auth_type == AUTH_TYPE_OAUTH
    assert persisted["credential_pool"]["anthropic"][0]["auth_type"] == AUTH_TYPE_OAUTH


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


def test_profile_global_fallback_normalizes_in_memory_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_root = tmp_path / ".hermes"
    global_root.mkdir()
    profile_home = global_root / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    token = "sk-ant-oat-global-fallback"
    global_auth = global_root / "auth.json"
    global_auth.write_text(json.dumps({
        "version": 1,
        "credential_pool": {
            "anthropic": [{
                "id": "global-oat",
                "label": "Global setup token",
                "auth_type": AUTH_TYPE_API_KEY,
                "priority": 0,
                "source": "manual",
                "access_token": token,
            }],
        },
    }))

    from agent.credential_pool import load_pool

    entry = load_pool("anthropic").entries()[0]
    persisted = json.loads(global_auth.read_text())
    assert entry.auth_type == AUTH_TYPE_OAUTH
    assert persisted["credential_pool"]["anthropic"][0]["auth_type"] == AUTH_TYPE_API_KEY
    assert not (profile_home / "auth.json").exists()
