"""Regression for #147165: gateway-only names must not hide local skills."""

import pytest

from hermes_constants import get_hermes_home


def _write_skill(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Surface routing fixture.\n---\n"
        "# Fixture\nKeep the synthetic project notes in order.\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("platform", [None, "local", "cli", "tui", "desktop", "acp"])
def test_gateway_only_names_load_locally_but_stay_reserved_in_messaging(monkeypatch, platform):
    from agent import skill_commands as sc
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    for name in ("start", "set-home", "notes-fixture"):
        _write_skill(get_hermes_home() / "skills" / name, name)

    def check_local():
        tokens = set_session_vars(platform=platform)
        try:
            commands = sc.get_interactive_skill_commands()
            assert "/notes-fixture" in commands
            assert "Keep the synthetic project notes in order." in sc.build_skill_invocation_message("/notes-fixture")
            for name in ("start", "set-home"):
                assert f"/{name}" in commands
                assert sc.skill_command_collision_note(name) is None
                message = sc.build_skill_invocation_message(f"/{name}", "Sort the fixture notes.")
                assert "Keep the synthetic project notes in order." in message
                assert "Sort the fixture notes." in message
        finally:
            clear_session_vars(tokens)

    check_local()
    tokens = set_session_vars(platform="telegram")
    try:
        commands = sc.get_interactive_skill_commands()
        assert "/notes-fixture" in commands
        for name in ("start", "set-home"):
            assert f"/{name}" not in commands
            assert sc.build_skill_invocation_message(f"/{name}") is None
            assert sc.skill_command_collision_note(name) is not None
    finally:
        clear_session_vars(tokens)
    check_local()  # The messaging view must not poison the local cache slot.


@pytest.mark.parametrize("name", ["handoff", "exit"])
def test_core_collisions_remain_reserved_and_frontmatter_rename_unblocks_them(monkeypatch, name):
    from agent import skill_commands as sc

    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    folder = get_hermes_home() / "skills" / name
    _write_skill(folder, name)
    assert f"/{name}" not in sc.scan_skill_commands()
    note = sc.skill_command_collision_note(name)
    assert "rename" in note and "frontmatter" in note
    assert "use /skill " not in note

    renamed = f"{name}-notes"
    _write_skill(folder, renamed)
    assert f"/{renamed}" in sc.scan_skill_commands()
    message = sc.build_skill_invocation_message(f"/{renamed}")
    assert "Keep the synthetic project notes in order." in message
