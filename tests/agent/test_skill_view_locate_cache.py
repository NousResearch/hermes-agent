"""skill_view must not re-parse the whole library to locate one skill.

``_collect_skill_candidates`` walked every SKILL.md and re-parsed its frontmatter
(``_safe_frontmatter``) on EVERY skill_view call — ~850 YAML parses and a full
tree walk per invocation, ~250-440ms warm at 850 skills, for a 3KB result. The
cached catalog rows already carry each copy's name and path.
"""

from pathlib import Path
from unittest.mock import patch


def _make_skill(root: Path, rel: str, name: str, desc: str = "d.") -> Path:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    md = d / "SKILL.md"
    md.write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\nBody.\n", encoding="utf-8")
    return md


def test_warm_skill_view_parses_no_frontmatter(tmp_path):
    import agent.skill_utils as su
    import tools.skills_tool as st

    skills_root = tmp_path / "tree"  # isolated: home (hermes_test) stays outside
    for i in range(30):
        _make_skill(skills_root, f"cat/skill{i:02d}", f"skill{i:02d}")

    parses = []
    real = su.parse_frontmatter

    def counting(content):
        parses.append(1)
        return real(content)

    with patch("tools.skills_tool.SKILLS_DIR", skills_root):
        st.clear_skills_cache()
        with patch.object(su, "parse_frontmatter", side_effect=counting):
            st.skill_view("skill07")  # cold: catalog scan parses once per skill
        cold = len(parses)
        parses.clear()
        with patch.object(su, "parse_frontmatter", side_effect=counting):
            st.skill_view("skill08")  # warm: catalog cached

    assert cold >= 30, "baseline sanity: a cold view builds the catalog"
    assert len(parses) <= 2, (
        f"a warm skill_view re-parsed {len(parses)} frontmatters; locating one skill "
        "must reuse the cached catalog rows (only the viewed file itself may parse)"
    )


def test_locate_finds_by_dir_name_and_frontmatter_name(tmp_path):
    import tools.skills_tool as st

    # dir name differs from frontmatter name; both lookups must work
    skills_root = tmp_path / "tree"
    _make_skill(skills_root, "cat/dir-name", "frontmatter-name")

    with patch("tools.skills_tool.SKILLS_DIR", skills_root):
        st.clear_skills_cache()
        by_dir = st.skill_view("dir-name")
        by_fm = st.skill_view("frontmatter-name")

    assert "frontmatter-name" in by_dir and "frontmatter-name" in by_fm
