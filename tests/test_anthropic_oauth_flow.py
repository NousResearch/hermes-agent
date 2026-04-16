"""Tests for Anthropic OAuth setup flow behavior."""

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

    from hermes_cli.main import _run_anthropic_oauth_flow

    assert _run_anthropic_oauth_flow(save_env_value) is True

    env_vars = load_env()
    assert env_vars["ANTHROPIC_TOKEN"] == "sk-ant-oat01-manual-token"
    output = capsys.readouterr().out
    assert "Setup-token saved" in output
