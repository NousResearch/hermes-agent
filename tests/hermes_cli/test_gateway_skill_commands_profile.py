"""Gateway skill command lists follow the active profile.

One gateway process serves several profiles and binds the active one per activity. The skill
collector behind the Telegram menu and the Discord ``/skill`` list must scan that profile's
skills folder, not the folder the process started with.
"""

from __future__ import annotations

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _profile_with_skill(home, skill_name):
    skill = home / "skills" / skill_name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: Only in this profile\n---\n\nBody.\n", encoding="utf-8")
    return home


def test_each_served_profile_lists_its_own_skill_commands(tmp_path):
    from hermes_cli.commands_platforms import discord_skill_commands_by_category

    homes = {name: _profile_with_skill(tmp_path / name, f"only-in-{name}") for name in ("alpha", "beta")}
    listed = {}
    for name, home in homes.items():
        token = set_hermes_home_override(home)
        try:
            _, uncategorized, _ = discord_skill_commands_by_category(reserved_names=set())
        finally:
            reset_hermes_home_override(token)
        listed[name] = {cmd_key for _, _, cmd_key in uncategorized}

    assert "/only-in-alpha" in listed["alpha"] and "/only-in-beta" not in listed["alpha"]
    assert "/only-in-beta" in listed["beta"] and "/only-in-alpha" not in listed["beta"]
