"""Platform command catalogs survive a broken skill-bundle file and only advertise dispatchable bundles."""

import pytest

from agent.skill_bundles import resolve_bundle_command_key
from hermes_cli.commands_platforms import discord_skill_commands_by_category, telegram_menu_commands


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "skill-bundles"
    bundle_dir.mkdir()
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundle_dir))
    monkeypatch.setattr("hermes_cli.commands_platforms._gateway_available_commands", list)
    monkeypatch.setattr("hermes_cli.commands_platforms._iter_plugin_command_entries", lambda: iter([]))
    monkeypatch.setattr("hermes_cli.commands_platforms._iter_gateway_skills",
                        lambda _platform: iter([("/valid-skill", {"description": "A valid skill"}, ())]))
    return bundle_dir


def test_a_non_utf8_bundle_never_hides_the_other_commands(catalog):
    (catalog / "broken.yaml").write_bytes(b"name: \xff\xfe broken\nskills: [a]\n")

    menu, _hidden = telegram_menu_commands()
    assert "valid_skill" in [name for name, _ in menu]
    _categories, uncategorized, _hidden = discord_skill_commands_by_category(set())
    assert ("valid-skill", "A valid skill", "/valid-skill") in uncategorized


def test_every_advertised_bundle_name_dispatches_to_its_bundle(catalog):
    long_name = "research-" + "x" * 40  # past Telegram's 32-char limit
    for name in ("short-bundle", long_name):
        (catalog / f"{name}.yaml").write_text(f"name: {name}\nskills: [one]\n", encoding="utf-8")

    menu, _hidden = telegram_menu_commands()
    names = [name for name, _ in menu]
    assert "short_bundle" in names
    for name in names:
        if name.startswith(("short", "research")):
            assert resolve_bundle_command_key(name) is not None, name
