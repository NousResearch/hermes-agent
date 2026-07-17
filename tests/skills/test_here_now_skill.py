"""Regression coverage for the bundled here-now skill metadata."""

from pathlib import Path

from agent.skill_utils import parse_frontmatter
from tools.skills_hub import UrlSource


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "here-now"
    / "SKILL.md"
)


def test_here_now_frontmatter_name_is_installable() -> None:
    frontmatter, _ = parse_frontmatter(SKILL_PATH.read_text(encoding="utf-8"))

    name = frontmatter.get("name")
    assert name == SKILL_PATH.parent.name == "here-now"
    assert UrlSource._is_valid_skill_name(name)
