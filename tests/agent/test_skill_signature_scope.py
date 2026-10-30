"""One signature walk per completion call.

``command_catalog`` validated the published skill-command map (one full directory
signature walk) and then ``find_all_skills`` re-validated the shared catalog with a
second identical walk inside the same RPC — 2x the stat/scandir traffic per palette
open for no freshness benefit (both walks observe the same disk state microseconds
apart). Measured on an 852-skill install: 181 ms warm catalog, 162 ms of it the two
walks.
"""
from unittest.mock import patch


def _make_skill(skills_dir, name):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\nDo the thing.\n", encoding="utf-8"
    )
    return skill_dir


def test_warm_command_catalog_walks_the_tree_once(tmp_path):
    import tools.skills_tool as st
    from tui_gateway.command_discovery import command_catalog

    for i in range(30):
        _make_skill(tmp_path, f"skill{i:02d}")

    walks = []
    real_compute = st._compute_scan_signature

    def counting_sig(dirs, disabled):
        walks.append(1)
        return real_compute(dirs, disabled)

    command_catalog()  # prime every cache with the real tree

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        st.clear_skills_cache()
        import agent.skill_commands as sc_mod

        with patch.object(sc_mod, "_skill_commands_by_key", {}):
            command_catalog()  # scan once under the patched dir
        # warm now: every cache valid, signature unchanged. Count DISK WALKS: the scope-aware
        # dispatcher may be called by several validators, but only one may reach the tree.
        walks.clear()
        with patch.object(st, "_compute_scan_signature", side_effect=counting_sig):
            command_catalog()

    assert len(walks) <= 1, (
        f"warm command_catalog walked the skill tree {len(walks)} times; "
        "one RPC must pay at most one walk (the validators share it)"
    )
