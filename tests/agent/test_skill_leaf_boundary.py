"""A skill's own tree is support content, not more skills.

The agentskills.io spec: "A skill directory may contain any files and
directories beyond the required ``SKILL.md``." Everything beneath a directory
that owns a SKILL.md is that skill's support material (references/, scripts/,
evals/, fixtures/, vendored copies). Indexing nested SKILL.md files as
top-level commands publishes a skill's internal test material as user-facing
commands and lets it collide with real skills by name — the scan treats a
directory that owns a SKILL.md as a leaf.
"""

import json


def _make_parent_skill(root, rel="cat/parent"):
    d = root / rel
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: parent\ndescription: Real skill.\n---\n\nBody.\n", encoding="utf-8")
    return d


def test_nested_skill_md_is_not_a_catalog_command(tmp_path, monkeypatch):
    import tools.skills_tool as st
    from agent.skill_commands import scan_skill_commands

    parent = _make_parent_skill(tmp_path)
    nested = parent / "fixtures" / "nested-demo"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: nested-demo\ndescription: Internal fixture, never a command.\n---\n\nx\n",
        encoding="utf-8")
    flat = tmp_path / "cat" / "real"
    flat.mkdir(parents=True)
    (flat / "SKILL.md").write_text(
        "---\nname: real\ndescription: Real.\n---\n\nx\n", encoding="utf-8")

    monkeypatch.setattr(st, "SKILLS_DIR", tmp_path)
    st.clear_skills_cache()  # fresh signature forces a rescan of the patched root
    cmds = scan_skill_commands()  # dict keyed by slash-prefixed command name
    names = {k.lstrip("/") for k in cmds}

    assert "parent" in names and "real" in names
    assert "nested-demo" not in names, (
        "a SKILL.md nested under another skill's directory is support content, "
        "not a user-facing command")


def test_nested_skill_md_is_not_locatable_via_skill_view(tmp_path, monkeypatch):
    import tools.skills_tool as st

    parent = _make_parent_skill(tmp_path)
    nested = parent / "references" / "nested-demo"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: nested-demo\ndescription: Internal fixture.\n---\n\nx\n", encoding="utf-8")

    monkeypatch.setattr(st, "SKILLS_DIR", tmp_path)
    st.clear_skills_cache()
    result = st.skill_view("nested-demo")
    assert "not found" in json.dumps(result).lower(), (
        "locating must not surface a skill nested inside another skill's support tree")
