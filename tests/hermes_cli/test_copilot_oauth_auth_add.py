"""Tests for adding GitHub Copilot credentials through OAuth device-code flow."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_auth_add_copilot_oauth_runs_device_code_and_persists_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.credential_pool import AUTH_TYPE_OAUTH
    from hermes_cli.auth import read_credential_pool
    from hermes_cli.auth_commands import auth_add_command

    with patch("hermes_cli.copilot_auth.copilot_device_code_login", return_value="gho_test_token") as login:
        auth_add_command(SimpleNamespace(
            provider="copilot",
            auth_type="oauth",
            label="test-copilot",
            api_key=None,
            timeout=123,
        ))

    login.assert_called_once_with(timeout_seconds=123)
    entries = read_credential_pool("copilot")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["label"] == "test-copilot"
    assert entry["auth_type"] == AUTH_TYPE_OAUTH
    assert entry["source"] == "manual:device_code"
    assert entry["access_token"] == ""
    assert entry["refresh_token"] == "gho_test_token"
    assert entry.get("base_url") is None


def test_pool_only_runtime_exchanges_source_token_and_uses_enterprise_endpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
    from hermes_cli import runtime_provider

    pool = load_pool("copilot")
    pool.add_entry(PooledCredential(
        provider="copilot",
        id="source1",
        label="test-copilot",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:device_code",
        access_token="",
        refresh_token="gho_source_token",
    ))

    exchanged = {}

    def fake_exchange(source_token):
        exchanged["source"] = source_token
        return (
            "tid=test;exp=9999999999;runtime=synthetic",
            9_999_999_999.0,
            "https://copilot.enterprise.example.com",
        )

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.exchange_copilot_token", fake_exchange
    )
    monkeypatch.setattr(runtime_provider, "resolve_provider", lambda *_a, **_kw: "copilot")
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {
        "provider": "copilot",
        "default": "gpt-4o",
    })

    resolved = runtime_provider.resolve_runtime_provider(
        requested="copilot", target_model="gpt-4o"
    )

    assert exchanged == {"source": "gho_source_token"}
    assert resolved["api_key"] == "tid=test;exp=9999999999;runtime=synthetic"
    assert resolved["api_key"] != "gho_source_token"
    assert resolved["base_url"] == "https://copilot.enterprise.example.com"


def test_interactive_add_offers_oauth_for_copilot(monkeypatch):
    from hermes_cli import auth_commands

    answers = iter(["copilot", "2", "test-copilot"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    captured = {}

    def fake_auth_add(args):
        captured["provider"] = args.provider
        captured["auth_type"] = args.auth_type
        captured["label"] = args.label

    monkeypatch.setattr(auth_commands, "auth_add_command", fake_auth_add)

    auth_commands._interactive_add()

    assert captured == {
        "provider": "copilot",
        "auth_type": "oauth",
        "label": "test-copilot",
    }
