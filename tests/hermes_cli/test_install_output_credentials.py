"""A token embedded in a source URL (``https://user:TOKEN@host/...``) never reaches install output."""
from __future__ import annotations

import pytest
from rich.console import Console

TOKEN = "s3cr3tTOKEN"


@pytest.mark.parametrize("url", [f"https://user:{TOKEN}@127.0.0.1:9/x/y.git", f"https://{TOKEN}@127.0.0.1:9/x/y.git"])
def test_plugin_install_failure_never_prints_the_url_token(tmp_path, monkeypatch, capsys, url):
    from hermes_cli.plugins_cmd import cmd_install

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    with pytest.raises(SystemExit):
        cmd_install(url)  # real git clone against a closed local port
    out = capsys.readouterr().out
    assert "Could not download the plugin from https://127.0.0.1:9/x/y.git" in out
    assert TOKEN not in out


def test_skills_hub_messages_never_print_the_url_token(tmp_path, monkeypatch):
    from hermes_cli.skills_hub import do_install, do_tap

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    console = Console(record=True, width=200)
    url = f"https://user:{TOKEN}@127.0.0.1:9/s/SKILL.md"
    do_install(url, console=console, source_id="url", skip_confirm=True)  # real fetch, closed port
    do_tap("add", url, console=console)
    out = console.export_text()
    assert "Could not download 'https://user:***@127.0.0.1:9/s/SKILL.md'" in out
    assert "Added tap: https://user:***@127.0.0.1:9/s/SKILL.md" in out
    assert TOKEN not in out
