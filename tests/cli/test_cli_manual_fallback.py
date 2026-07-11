"""CLI bootstrap behavior for manual provider fallback."""

from unittest.mock import MagicMock, patch

from hermes_cli.auth import AuthError
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _FakeCLI(CLIAgentSetupMixin):
    requested_provider = "openai-codex"
    _explicit_api_key = None
    _explicit_base_url = None
    _fallback_model = [{"provider": "openrouter", "model": "gpt-5.4"}]
    _fallback_auto_activate = False
    model = "gpt-5.4"


def test_manual_mode_auth_failure_does_not_auto_resolve_fallback():
    cli = _FakeCLI()

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        side_effect=AuthError("token expired"),
    ) as resolve, patch("cli.ChatConsole", return_value=MagicMock()):
        assert cli._ensure_runtime_credentials() is False

    resolve.assert_called_once_with(
        requested="openai-codex",
        explicit_api_key=None,
        explicit_base_url=None,
    )
    assert cli.requested_provider == "openai-codex"
    assert cli.model == "gpt-5.4"
