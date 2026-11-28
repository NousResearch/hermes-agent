"""CFG-8: a skill installed from another terminal reaches the long-lived gateway's slash map.

The one gateway owns every local session for its whole lifetime, so a process-lifetime memo of the
skill command map left ``hermes skills install`` (or a dropped SKILL.md) invisible to Ink's
``/command`` lookup and completion until ``/reload-skills`` or a gateway restart.
"""
from unittest.mock import patch


def _make_skill(skills_dir, name):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\nDo the thing.\n", encoding="utf-8")
    return skill_dir


def test_skill_added_on_disk_after_the_first_scan_is_served_without_reload(tmp_path):
    import shutil

    import agent.skill_commands as sc_mod
    import tools.skills_tool as st
    from agent.skill_commands import get_skill_commands

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), patch.object(sc_mod, "_skill_commands_by_key", {}):
        st.clear_skills_cache()
        _make_skill(tmp_path, "first")
        assert set(get_skill_commands()) == {"/first"}

        _make_skill(tmp_path, "installed-elsewhere")  # another terminal: no reload_skills() here
        assert set(get_skill_commands()) == {"/first", "/installed-elsewhere"}

        shutil.rmtree(tmp_path / "first")  # an uninstall elsewhere drops its command too
        assert set(get_skill_commands()) == {"/installed-elsewhere"}


def test_unchanged_skills_dirs_reuse_the_published_map(tmp_path):
    import agent.skill_commands as sc_mod
    import tools.skills_tool as st
    from agent.skill_commands import get_skill_commands

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), patch.object(sc_mod, "_skill_commands_by_key", {}):
        st.clear_skills_cache()
        _make_skill(tmp_path, "steady")
        first = get_skill_commands()
        with patch.object(sc_mod, "scan_skill_commands", side_effect=AssertionError("rescanned")):
            assert get_skill_commands() is first


def test_a_skill_installed_under_an_existing_category_subdir_is_served(tmp_path):
    """``skills/mlops/training/<new>/SKILL.md``: the install moves only ``training``'s mtime, two
    levels below the root, and the published map must still be rescanned (the catalog walks it)."""
    import agent.skill_commands as sc_mod
    import tools.skills_tool as st
    from agent.skill_commands import get_skill_commands

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), patch.object(sc_mod, "_skill_commands_by_key", {}):
        st.clear_skills_cache()
        training = tmp_path / "mlops" / "training"
        _make_skill(training, "existing")
        assert set(get_skill_commands()) == {"/existing"}
        _make_skill(training, "new-nested")
        assert set(get_skill_commands()) == {"/existing", "/new-nested"}
        # A skill's own support dirs are not walked: editing references/ is not an install.
        before = sc_mod._skill_dirs_signature()
        (training / "existing" / "references").mkdir()
        (training / "existing" / "references" / "deep").mkdir()
        after = sc_mod._skill_dirs_signature()
        assert [row for row in after[0] if "references" in row[0]] == []
        assert before != after  # the skill dir itself gained a child name


def test_only_an_unreadable_skill_root_keeps_the_published_map():
    """A stat failure (OSError) gives no signature and keeps the published map; a bug in the signature
    code raises instead of pinning a stale map for the life of the gateway without a trace."""
    import pytest

    import agent.skill_commands as sc_mod

    with patch("tools.skills_tool._skills_scan_signature", side_effect=PermissionError("denied")):
        assert sc_mod._skill_dirs_signature() is None
    with patch("tools.skills_tool._skills_scan_signature", side_effect=TypeError("bug")):
        with pytest.raises(TypeError):
            sc_mod._skill_dirs_signature()
