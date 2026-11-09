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
