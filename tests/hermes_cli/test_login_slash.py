from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.login_slash import normalize_login_provider, run_login_slash


def test_login_provider_aliases_cover_common_oauth_names():
    assert normalize_login_provider("anthropic") == "anthropic"
    assert normalize_login_provider("claude") == "anthropic"
    assert normalize_login_provider("codex") == "openai-codex"
    assert normalize_login_provider("github") == "copilot"
    assert normalize_login_provider("github-copilot") == "copilot"


def test_login_anthropic_defaults_to_oauth():
    with patch("hermes_cli.login_slash.auth_add_command") as add:
        run_login_slash("/login anthropic --label main --no-browser")

    args = add.call_args.args[0]
    assert args.provider == "anthropic"
    assert args.auth_type == "oauth"
    assert args.label == "main"
    assert args.no_browser is True


@pytest.mark.parametrize(
    ("command", "provider"),
    [
        ("/login codex", "openai-codex"),
        ("/login github", "copilot"),
        ("/login add github", "copilot"),
    ],
)
def test_login_normalizes_codex_and_github_aliases(command, provider):
    with patch("hermes_cli.login_slash.auth_add_command") as add:
        run_login_slash(command)

    args = add.call_args.args[0]
    assert args.provider == provider
    assert args.auth_type == "oauth"


def test_login_status_lists_pool_for_optional_provider():
    with patch("hermes_cli.login_slash.auth_list_command") as status, patch(
        "agent.credential_pool.load_pool"
    ) as load_pool:
        load_pool.return_value.has_credentials.return_value = True
        run_login_slash("/login status anthropic")

    status.assert_called_once()
    assert status.call_args.args[0].provider == "anthropic"


def test_login_logout_requires_credential_target():
    with pytest.raises(SystemExit) as exc:
        run_login_slash("/login logout anthropic")

    assert "Credential target required" in str(exc.value)


def test_login_logout_removes_target():
    with patch("hermes_cli.login_slash.auth_remove_command") as remove:
        run_login_slash("/login logout github 1")

    args = remove.call_args.args[0]
    assert args.provider == "copilot"
    assert args.target == "1"


def test_login_refresh_resets_provider_status(capsys):
    with patch("hermes_cli.login_slash.auth_reset_command") as reset:
        run_login_slash("/login refresh codex")

    args = reset.call_args.args[0]
    assert args.provider == "openai-codex"
    assert "runtime token will refresh" in capsys.readouterr().out
