"""Tests for Anthropic OAuth setup flow behavior."""

import shutil
import subprocess

import pytest

from hermes_cli.config import load_env, save_env_value


def test_explicit_claude_setup_reenables_suppressed_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import anthropic_credentials as ac
    from hermes_cli.auth import is_source_suppressed, suppress_credential_source

    suppress_credential_source("anthropic", "claude_code")
    monkeypatch.setattr(shutil, "which", lambda _name: "/fake/claude")
    monkeypatch.setattr(ac.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0))
    reads = iter([
        {"accessToken": "old-access", "refreshToken": "old-refresh", "expiresAt": 9_999_999_999_999},
        {"accessToken": "fresh-access", "refreshToken": "fresh-refresh", "expiresAt": 9_999_999_999_999},
    ])
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_keychain", lambda: next(reads))
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_file", lambda: None)

    assert ac.run_oauth_setup_token() == "fresh-access"
    assert is_source_suppressed("anthropic", "claude_code") is False


def test_failed_claude_setup_keeps_suppression(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import anthropic_credentials as ac
    from hermes_cli.auth import is_source_suppressed, suppress_credential_source

    suppress_credential_source("anthropic", "claude_code")
    monkeypatch.setattr(shutil, "which", lambda _name: "/fake/claude")
    monkeypatch.setattr(ac.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1))
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_keychain", lambda: {
        "accessToken": "old-access", "refreshToken": "old-refresh", "expiresAt": 9_999_999_999_999,
    })
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_file", lambda: None)

    assert ac.run_oauth_setup_token() is None
    assert is_source_suppressed("anthropic", "claude_code") is True


def test_corrupt_auth_store_blocks_setup_before_external_access(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text("{not-json")
    from agent import anthropic_credentials as ac

    monkeypatch.setattr(shutil, "which", lambda _name: "/fake/claude")

    def forbidden(*_args, **_kwargs):
        pytest.fail("corrupt auth state allowed Claude setup or credential access")

    monkeypatch.setattr(ac.subprocess, "run", forbidden)
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_keychain", forbidden)
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_file", forbidden)

    assert ac.run_oauth_setup_token() is None
    assert (tmp_path / "auth.json").read_text() == "{not-json"


def test_run_anthropic_oauth_flow_prefers_claude_code_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "agent.anthropic_credentials.run_oauth_setup_token",
        lambda: "sk-ant-oat01-from-claude-setup",
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials",
        lambda: {
            "accessToken": "cc-access-token",
            "refreshToken": "cc-refresh-token",
            "expiresAt": 9999999999999,
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.is_claude_code_token_valid",
        lambda creds: True,
    )

    from hermes_cli.main_provider_setup import _run_anthropic_oauth_flow

    save_env_value("ANTHROPIC_TOKEN", "stale-env-token")
    assert _run_anthropic_oauth_flow(save_env_value) is True

    env_vars = load_env()
    assert env_vars["ANTHROPIC_TOKEN"] == ""
    assert env_vars["ANTHROPIC_API_KEY"] == ""
    output = capsys.readouterr().out
    assert "Claude Code credentials linked" in output


def test_run_anthropic_oauth_flow_manual_token_still_persists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.anthropic_credentials.run_oauth_setup_token", lambda: None)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_credentials.is_claude_code_token_valid", lambda creds: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "sk-ant-oat01-manual-token")
    monkeypatch.setattr(
        "hermes_cli.secret_prompt.masked_secret_prompt",
        lambda _prompt="": "sk-ant-oat01-manual-token",
    )

    from hermes_cli.main_provider_setup import _run_anthropic_oauth_flow

    assert _run_anthropic_oauth_flow(save_env_value) is True

    env_vars = load_env()
    assert env_vars["ANTHROPIC_TOKEN"] == "sk-ant-oat01-manual-token"
    output = capsys.readouterr().out
    assert "Setup-token saved" in output
