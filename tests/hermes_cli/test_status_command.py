from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.status import show_status


def test_status_shows_claude_cli_backend_details(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    args = SimpleNamespace(all=False, deep=False)
    with patch("hermes_cli.status.load_config", return_value={"model": {"provider": "claude-cli", "default": "claude-cli/claude-sonnet-4-6"}}), \
         patch("hermes_cli.status.resolve_requested_provider", return_value="claude-cli"), \
         patch("hermes_cli.status.get_env_value", return_value=""), \
         patch("hermes_cli.status.get_nous_subscription_features") as _features, \
         patch("hermes_cli.status.managed_nous_tools_enabled", return_value=False), \
         patch("hermes_cli.auth.get_nous_auth_status", return_value={}), \
         patch("hermes_cli.auth.get_codex_auth_status", return_value={}), \
         patch("hermes_cli.auth.get_external_process_provider_status", side_effect=[
             {"configured": False, "logged_in": False},
             {"configured": True, "logged_in": True, "resolved_command": "/usr/local/bin/claude", "command": "claude", "args": ["--debug"], "base_url": "claude-cli://local"},
         ]):
        show_status(args)

    out = capsys.readouterr().out
    assert "Claude CLI" in out
    assert "/usr/local/bin/claude" in out
    assert "claude-cli://local" in out
