"""Tests for Anthropic OAuth setup flow behavior."""

import json

from hermes_cli.config import load_env, save_env_value


def test_run_anthropic_oauth_flow_prefers_claude_code_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "agent.anthropic_adapter.run_oauth_setup_token",
        lambda: "sk-ant-oat01-from-claude-setup",
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: {
            "accessToken": "cc-access-token",
            "refreshToken": "cc-refresh-token",
            "expiresAt": 9999999999999,
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.is_claude_code_token_valid",
        lambda creds: True,
    )

    from hermes_cli.main import _run_anthropic_oauth_flow

    save_env_value("ANTHROPIC_TOKEN", "stale-env-token")
    assert _run_anthropic_oauth_flow(save_env_value) is True

    env_vars = load_env()
    assert env_vars["ANTHROPIC_TOKEN"] == ""
    assert env_vars["ANTHROPIC_API_KEY"] == ""
    output = capsys.readouterr().out
    assert "Claude Code credentials linked" in output


def test_run_anthropic_oauth_flow_manual_token_still_persists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.anthropic_adapter.run_oauth_setup_token", lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.is_claude_code_token_valid", lambda creds: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "sk-ant-oat01-manual-token")
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": "sk-ant-oat01-manual-token")

    from hermes_cli.main import _run_anthropic_oauth_flow

    assert _run_anthropic_oauth_flow(save_env_value) is True

    env_vars = load_env()
    assert env_vars["ANTHROPIC_TOKEN"] == "sk-ant-oat01-manual-token"
    output = capsys.readouterr().out
    assert "Setup-token saved" in output


def test_model_flow_anthropic_oauth_persists_complete_auth_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "agent.anthropic_adapter.run_oauth_setup_token",
        lambda: "sk-ant-oat01-from-claude-setup",
    )
    monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.is_claude_code_token_valid", lambda creds: False)
    monkeypatch.setattr("agent.anthropic_adapter._is_oauth_token", lambda key: key.startswith("sk-ant-"))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "1")
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *_args, **_kwargs: "claude-test")

    from hermes_cli.main import _model_flow_anthropic

    _model_flow_anthropic({})

    payload = json.loads((tmp_path / "auth.json").read_text())
    assert payload["active_provider"] == "anthropic"

    pool_entry = payload["credential_pool"]["anthropic"][0]
    assert pool_entry["source"] == "env:ANTHROPIC_TOKEN"
    assert pool_entry["auth_type"] == "oauth"
    assert pool_entry["access_token"] == "sk-ant-oat01-from-claude-setup"
    assert pool_entry["base_url"] == "https://api.anthropic.com"

    provider_state = payload["providers"]["anthropic"]
    assert provider_state["source"] == pool_entry["source"]
    assert provider_state["auth_type"] == pool_entry["auth_type"]
    assert provider_state["access_token"] == pool_entry["access_token"]
    assert provider_state["base_url"] == pool_entry["base_url"]
