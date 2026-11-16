"""Gateway skill menus must follow the owning profile under multiplexing (#150628)."""

from pathlib import Path

from agent.runtime_scope import bind_profile_scope
from hermes_cli.commands_platforms import discord_skill_commands_by_category, telegram_menu_commands


def _skill(root, name):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill.\n---\n\nTest instructions.\n",
        encoding="utf-8",
    )


def test_gateway_skill_menus_follow_profile_scope(tmp_path, monkeypatch):
    """Real discovery includes local and external skills, never another profile's skills."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(launch))
    external = tmp_path / "shared-skills"
    _skill(external, "shared-skill")
    homes = [launch, launch / "profiles" / "content"]
    for home, name in zip(homes, ("launch-only", "content-only")):
        _skill(home / "skills", name)
        _skill(home / "skills" / ".hub", "hidden-hub")
        (home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external.as_posix()}\n", encoding="utf-8",
        )

    for home, local, foreign in ((homes[0], "launch-only", "content-only"),
                                 (homes[1], "content-only", "launch-only"),
                                 (homes[0], "launch-only", "content-only")):
        with bind_profile_scope(home, hydrate_secrets=False):
            categories, uncategorized, hidden = discord_skill_commands_by_category(set())
            entries = [*uncategorized, *(entry for group in categories.values() for entry in group)]
            discord_keys = {entry[2] for entry in entries}
            assert f"/{local}" in discord_keys
            assert "/shared-skill" in discord_keys
            assert f"/{foreign}" not in discord_keys
            assert "/hidden-hub" not in discord_keys
            assert hidden == 0
            menu, _hidden = telegram_menu_commands()
            telegram_keys = {name for name, _description in menu}
            assert local.replace("-", "_") in telegram_keys
            assert foreign.replace("-", "_") not in telegram_keys
