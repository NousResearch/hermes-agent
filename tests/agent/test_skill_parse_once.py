"""Parse-once: a cold catalog RPC parses each SKILL.md exactly one time.

``_skill_catalog`` read and parsed every SKILL.md to build its rows, then
``scan_skill_commands`` re-read and re-parsed the same files (``_scan_skill_md``)
to build the slash-command map — two full I/O + YAML passes over the whole tree
per cold scan. The catalog row now carries what the command map needs
(``_cmd_name`` / ``_cmd_desc``), so the map registers from rows with no second read.
"""

from pathlib import Path
from unittest.mock import patch


def _make_skill(skills_dir: Path, name: str) -> None:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Skill {name}.\n---\n\nBody.\n", encoding="utf-8")


def test_cold_catalog_parses_each_skill_md_once(tmp_path):
    import agent.skill_utils as su
    from tui_gateway.command_discovery import command_catalog

    for i in range(40):
        _make_skill(tmp_path, f"skill{i:02d}")

    parses = []
    real_parse = su.parse_frontmatter

    def counting_parse(content):
        parses.append(1)
        return real_parse(content)

    import tools.skills_tool as st

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        st.clear_skills_cache()
        import agent.skill_commands as sc_mod

        with patch.object(sc_mod, "_skill_commands_by_key", {}):
            with patch.object(su, "parse_frontmatter", side_effect=counting_parse):
                command_catalog()  # cold: catalog scan AND command-map build

    assert len(parses) <= 40, (
        f"cold command_catalog parsed {len(parses)} frontmatters for 40 skills; "
        "each SKILL.md must be parsed exactly once (the command map reuses catalog rows)"
    )


def test_command_map_matches_row_registration(tmp_path):
    """Row-driven registration keeps the map byte-equal: same names, descriptions, paths."""
    import tools.skills_tool as st
    import agent.skill_commands as sc_mod

    for i in range(6):
        _make_skill(tmp_path, f"s{i}")
    # one skill with no frontmatter description -> body first line (<=80 chars) is used
    body_only = tmp_path / "bodyonly"
    body_only.mkdir()
    (body_only / "SKILL.md").write_text("---\nname: bodyonly\n---\nFirst body line as the description.\n", encoding="utf-8")

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        st.clear_skills_cache()
        with patch.object(sc_mod, "_skill_commands_by_key", {}):
            commands = sc_mod.get_skill_commands()

    assert set(commands) == {f"/s{i}" for i in range(6)} | {"/bodyonly"}
    assert commands["/bodyonly"]["description"] == "First body line as the description."
    assert commands["/s0"]["description"] == "Skill s0."
    assert commands["/s0"]["skill_md_path"].endswith("SKILL.md")
