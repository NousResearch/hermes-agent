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
    assert entry["access_token"] == "gho_test_token"
    assert entry["base_url"] == "https://api.githubcopilot.com"


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
