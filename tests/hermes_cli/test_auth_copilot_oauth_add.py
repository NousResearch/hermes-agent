from types import SimpleNamespace
from unittest.mock import patch

from agent.credential_pool import AUTH_TYPE_OAUTH, load_pool
from hermes_cli.auth_commands import auth_add_command


def test_auth_add_copilot_oauth_device_code_stores_pool_entry(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for name in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value=None), patch(
        "hermes_cli.copilot_auth.copilot_device_code_login",
        return_value="gho_testtoken",
    ) as login:
        auth_add_command(
            SimpleNamespace(
                provider="copilot",
                auth_type="oauth",
                label="github-main",
                timeout=1,
            )
        )

    login.assert_called_once_with(timeout_seconds=1)
    with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value=None):
        entries = load_pool("copilot").entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.provider == "copilot"
    assert entry.auth_type == AUTH_TYPE_OAUTH
    assert entry.source == "manual:device_code"
    assert entry.access_token == "gho_testtoken"
    assert entry.label == "github-main"
