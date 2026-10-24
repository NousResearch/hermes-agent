"""Startup noise switches for the interactive CLI: ``display.compact`` reaches the banner,
``display.banner: false`` drops the intro, ``display.composer_hints: false`` drops the idle hints."""
import importlib
import sys
import types

import pytest


def _import_cli():
    for name in list(sys.modules):
        if name in ("cli", "run_agent", "tools") or name.startswith("tools."):
            sys.modules.pop(name, None)
    if "firecrawl" not in sys.modules:
        sys.modules["firecrawl"] = types.SimpleNamespace(Firecrawl=object)
    return importlib.import_module("cli")


@pytest.fixture
def cli_with_display(monkeypatch):
    def build(**display):
        cli = _import_cli()
        monkeypatch.setitem(cli.CLI_CONFIG, "display", {**cli.CLI_CONFIG["display"], **display})
        return cli
    return build


def test_display_compact_applies_when_no_flag_is_passed(cli_with_display):
    cli = cli_with_display(compact=True)
    assert cli.HermesCLI(max_turns=1).compact is True


def test_banner_off_prints_no_intro(cli_with_display, monkeypatch):
    cli = cli_with_display(banner=False)
    shell = cli.HermesCLI(max_turns=1)
    printed = []
    monkeypatch.setattr(shell, "_console_print", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(shell.console, "clear", lambda: None)
    shell.show_banner()
    text = " ".join(str(a) for row in printed for a in row)
    assert "Available Tools" not in text
    assert "AI Agent Framework" not in text


def test_banner_default_still_prints_intro(cli_with_display, monkeypatch):
    cli = cli_with_display(compact=True)
    shell = cli.HermesCLI(max_turns=1)
    printed = []
    monkeypatch.setattr(shell, "_console_print", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(shell, "_show_status", lambda: None)
    monkeypatch.setattr(shell.console, "clear", lambda: None)
    shell.show_banner()
    assert printed


def test_composer_hints_off_hides_idle_hints_only(cli_with_display):
    from hermes_cli.cli_modern_layout import hint_pairs
    cli = cli_with_display(composer_hints=False)
    shell = cli.HermesCLI(max_turns=1)
    assert hint_pairs(shell) == []
    shell._agent_running = True
    assert [key for key, _ in hint_pairs(shell)][-1] == "ctrl+c"


def test_composer_hints_default_shows_idle_hints(cli_with_display):
    from hermes_cli.cli_modern_layout import hint_pairs
    shell = cli_with_display().HermesCLI(max_turns=1)
    assert [key for key, _ in hint_pairs(shell)] == ["/", "@", "!"]
