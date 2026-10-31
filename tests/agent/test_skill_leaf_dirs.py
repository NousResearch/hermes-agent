"""Skill directories are leaves: scanners never descend below a dir holding SKILL.md.

Walkers full-descended the tree (pruning only the four support-dir names), so a
skill carrying unrelated bulk — docs/, fixtures/, node_modules — taxed every scan:
~4,700 needless directories per cold catalog, every per-turn prompt-manifest walk,
and every warm completion signature walk. A SKILL.md nested in a non-support
subdir of another skill was never a real skill in practice (zero occurrences
across stock, hub, and 850+ skill corpora).
"""

from pathlib import Path
from unittest.mock import patch


def _skill(path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Skill {name}.\n---\n\nBody.\n", encoding="utf-8")


def test_nested_skill_md_is_not_a_skill(tmp_path):
    import tools.skills_tool as st
    from agent.skill_utils import iter_skill_index_files

    _skill(tmp_path / "outer", "outer")
    _skill(tmp_path / "outer" / "docs" / "inner", "inner")  # non-support subdir

    found = {p.name: p for p in iter_skill_index_files(tmp_path, "SKILL.md")}
    assert str(found.get("outer", tmp_path / "outer" / "SKILL.md")).endswith("outer/SKILL.md")
    assert all("inner" not in str(p) for p in found), (
        "a SKILL.md below another skill's directory must not register as a skill"
    )

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        st.clear_skills_cache()
        names = {s["name"] for s in st.find_all_skills()}
    assert names == {"outer"}


def test_manifest_does_not_descend_past_skill_dirs(tmp_path):
    import agent.prompt_builder as pb

    _skill(tmp_path / "s", "s")
    bulk = tmp_path / "s" / "docs"
    for i in range(50):
        p = bulk / f"d{i}" / "e"
        p.mkdir(parents=True, exist_ok=True)
        (p / "bigfile.md").write_text("x" * 1000, encoding="utf-8")

    visited = []
    real_walk = __import__("os").walk

    def counting_walk(top, **kw):
        for root, dirs, files in real_walk(top, **kw):
            visited.append(root)
            yield root, dirs, files

    with patch("os.walk", counting_walk):
        manifest = pb._build_skills_manifest(tmp_path)

    assert set(manifest) == {"s/SKILL.md"}, "manifest covers exactly the top-level SKILL.md"
    assert not any("docs" in v for v in visited), (
        "the manifest walk must not descend into a skill's subdirectories"
    )
