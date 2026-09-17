"""Swedish command presentation must not change dispatch or profile isolation."""
from dataclasses import asdict

import pytest
from prompt_toolkit.document import Document

from agent import i18n
from hermes_cli.commands import COMMAND_REGISTRY, COMMANDS, gateway_help_lines, resolve_command
from hermes_cli.commands_completion import SlashCommandCompleter
from hermes_cli.commands_i18n import (
    SWEDISH_DESCRIPTIONS, cli_command_description, command_description,
)


@pytest.fixture(autouse=True)
def isolated_language(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


def test_registry_coverage_and_dispatch_identity(monkeypatch):
    before = [asdict(cmd) for cmd in COMMAND_REGISTRY]
    assert SWEDISH_DESCRIPTIONS.keys() == {cmd.name for cmd in COMMAND_REGISTRY}
    monkeypatch.setenv("HERMES_LANGUAGE", "sv_FI")
    for cmd in COMMAND_REGISTRY:
        assert command_description(cmd) == SWEDISH_DESCRIPTIONS[cmd.name]
        for alias in cmd.aliases:
            assert resolve_command(alias) is cmd
            assert f"/{cmd.name}" in cli_command_description(alias, "fallback")
        if cmd.args_hint:
            assert f"/{cmd.name} {cmd.args_hint}" in cli_command_description(cmd.name, "fallback")
    assert [asdict(cmd) for cmd in COMMAND_REGISTRY] == before
    assert cli_command_description("/external-fixture", "Custom plugin copy") == "Custom plugin copy"


def test_cached_registry_follows_real_profile_language(monkeypatch, tmp_path):
    profiles = []
    for lang in ("sv-SE", "en"):
        home = tmp_path / lang
        home.mkdir()
        (home / "config.yaml").write_text(f"display:\n  language: {lang}\n", encoding="utf-8")
        profiles.append(home)
    cmd = resolve_command("model")
    for home, expected in [(profiles[0], "Byt modell"), (profiles[1], "Switch model"), (profiles[0], "Byt modell")]:
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert command_description(cmd).startswith(expected)
    monkeypatch.setenv("HERMES_LANGUAGE", "unknown")
    assert command_description(cmd) == cmd.description
    assert cli_command_description("/model", COMMANDS["/model"]) == COMMANDS["/model"]


def test_real_completer_and_gateway_help_preserve_command_text(monkeypatch):
    completer = SlashCommandCompleter(skill_commands_provider=lambda: {}, skill_bundles_provider=lambda: {})
    document = Document("/mod", cursor_position=4)
    english = list(completer.get_completions(document, None))
    monkeypatch.setenv("HERMES_LANGUAGE", "sv")
    swedish = list(completer.get_completions(document, None))
    assert [(c.text, c.start_position) for c in english] == [(c.text, c.start_position) for c in swedish]
    assert any("Byt modell" in c.display_meta_text for c in swedish)
    assert any("Byt modell" in line and "`/model " in line for line in gateway_help_lines())


def test_gateway_catalog_changes_only_descriptions(monkeypatch, tmp_path):
    from tui_gateway import server

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    english = server._methods["commands.catalog"]("en", {})["result"]
    monkeypatch.setenv("HERMES_LANGUAGE", "sv")
    swedish = server._methods["commands.catalog"]("sv", {})["result"]
    assert dict(swedish["pairs"])["/model"].startswith("Byt modell")
    assert [name for name, _ in english["pairs"]] == [name for name, _ in swedish["pairs"]]
    for field in ("canon", "commands", "sub"):
        assert swedish[field] == english[field]
    assert [c["name"] for c in swedish["categories"]] == [c["name"] for c in english["categories"]]
