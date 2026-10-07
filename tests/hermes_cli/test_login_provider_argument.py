"""Shared CLI/TUI /login provider routing (#74728)."""

from unittest.mock import MagicMock

import pytest

from agent import i18n
from cli import HermesCLI
from hermes_cli import anon_auth
from hermes_cli.commands import resolve_command
from tui_gateway.slash_worker import _run


@pytest.mark.parametrize("surface", ["cli", "tui"])
@pytest.mark.parametrize("arg", ["codex", "other", "nous extra"])
def test_provider_argument_never_starts_the_wrong_sign_in(monkeypatch, surface, arg):
    cli = object.__new__(HermesCLI)
    cli._slash_metrics_surface = None
    cli._app = None
    cli.console = MagicMock()
    cli._side_worker = MagicMock()
    output = []
    monkeypatch.setattr("cli._cprint", output.append)
    monkeypatch.setattr("cli.get_skill_commands", lambda: {})
    monkeypatch.setattr("hermes_cli.plugins.fire_pre_command_hook", lambda **kwargs: None)
    flow = MagicMock(return_value=iter([anon_auth.AlreadySignedIn()]))
    monkeypatch.setattr(anon_auth, "run_sign_in", flow)

    if surface == "tui":
        text = _run(cli, f"/login {arg}")
    else:
        assert cli.process_command(f"/login {arg}")
        text = "\n".join(output)

    flow.assert_not_called()
    cli._side_worker.assert_not_called()
    assert len(text.strip().splitlines()) == 1
    if arg == "codex":
        assert "hermes auth add openai-codex" in text
    else:
        assert text.strip() == "Usage: /login [nous|codex]"

    # Both supported Nous forms still use the original composition and timeout.
    for command in ("/login", "/login nous"):
        flow.return_value = iter([anon_auth.AlreadySignedIn()])
        cli.process_command(command)
    assert flow.call_count == 2
    flow.assert_called_with(timeout_seconds=8.0)


def test_login_help_describes_both_advertised_providers():
    command = resolve_command("login")
    assert "Codex" in command.describe()
    assert command.args_hint == "[nous|codex]"


def test_codex_terminal_pointer_uses_the_profile_locale(monkeypatch, tmp_path):
    home = tmp_path / "profile"
    (home / "locales").mkdir(parents=True)
    (home / "locales" / "fr.yaml").write_text(
        'cli:\n  commands:\n    login:\n      codex_terminal: "Terminal local: hermes auth add openai-codex"\n')
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_LANGUAGE", "fr")
    i18n.reset_language_cache()
    cli = object.__new__(HermesCLI)
    output = []
    monkeypatch.setattr("cli._cprint", output.append)
    flow = MagicMock(return_value=iter([anon_auth.AlreadySignedIn()]))
    monkeypatch.setattr(anon_auth, "run_sign_in", flow)
    try:
        cli._handle_login_command("/login codex")
        flow.assert_not_called()
        assert output == ["  Terminal local: hermes auth add openai-codex"]
    finally:
        i18n.reset_language_cache()
