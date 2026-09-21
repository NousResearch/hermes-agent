"""Copilot auth status and catalog must agree on the CLI's JSONC store."""
import json
import sys

import pytest


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    for name in ("HOME", "USERPROFILE", "HERMES_HOME", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(tmp_path))
    for name in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", sys.executable)
    path = tmp_path / ".copilot" / "config.json"
    path.parent.mkdir()
    return path


def status():
    from hermes_cli.auth import get_external_process_provider_status
    return get_external_process_provider_status("copilot-acp")


@pytest.mark.parametrize("comment", ["", "// full line\n", "/* block\ncomment */"])
@pytest.mark.parametrize("inline", ["", "// active login\n", "/* active login */"])
def test_both_readers_accept_comments(config_file, comment, inline):
    from hermes_cli.models import _copilot_cli_config_tokens
    from hermes_cli.copilot_auth import load_copilot_cli_config
    token = 'gho_quote"_slash\\_//_/*literal*/'
    value = {"https://github.com:user": token}
    config_file.write_text(
        '{\n' + comment + '"copilotTokens":' + json.dumps(value) + ', ' + inline
        + '"lastLoggedInUser":{"login":"user"}}', encoding="utf-8")
    assert load_copilot_cli_config()["copilotTokens"] == value
    assert _copilot_cli_config_tokens() == [token]
    assert status()["auth_verified"] is True
    assert status()["auth_source"] == "~/.copilot/config.json"


@pytest.mark.parametrize("text", [None, "", " \n", "// comment only", "/* only */", "{}", '{"copilotTokens": []}'])
def test_empty_store(config_file, text):
    from hermes_cli.models import _copilot_cli_config_tokens
    if text is not None:
        config_file.write_text(text, encoding="utf-8")
    assert _copilot_cli_config_tokens() == []
    assert status()["auth_verified"] is False


@pytest.mark.parametrize("text", ['{"copilotTokens":', '{} /* unterminated', '{"x": 1/* comment */2}'])
def test_invalid_config_preserves_hosts_fallback(config_file, text):
    from hermes_cli.models import _copilot_cli_config_tokens
    config_file.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        _copilot_cli_config_tokens()
    assert status()["auth_verified"] is False
    hosts = config_file.parent.parent / ".config" / "github-copilot" / "hosts.json"
    hosts.parent.mkdir(parents=True)
    hosts.write_text('{"github.com": {}}', encoding="utf-8")
    assert status()["auth_source"] == "~/.config/github-copilot/hosts.json"


def test_env_precedence(config_file, monkeypatch):
    config_file.write_text("not JSON", encoding="utf-8")
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_synthetic")
    assert status()["auth_source"] == "env: COPILOT_GITHUB_TOKEN"


def test_io_failure_preserves_status_fallback(config_file, monkeypatch):
    from hermes_cli import copilot_auth
    config_file.write_text("{}", encoding="utf-8")
    def denied(*args, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(copilot_auth.Path, "read_text", denied)
    assert status()["auth_verified"] is False
