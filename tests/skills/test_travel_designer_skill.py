"""Tests for the travel-designer optional skill."""
from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "travel-designer"
    / "SKILL.md"
)


def _frontmatter_and_body() -> tuple[dict[str, str], str]:
    text = SKILL_PATH.read_text(encoding="utf-8")
    _, raw_frontmatter, body = text.split("---", 2)
    frontmatter = {}
    for line in raw_frontmatter.splitlines():
        if ": " in line and not line.startswith(" "):
            key, value = line.split(": ", 1)
            frontmatter[key] = value.strip().strip('"')
    return frontmatter, body


def test_travel_designer_frontmatter_is_valid():
    frontmatter, body = _frontmatter_and_body()

    assert frontmatter["name"] == "travel-designer"
    assert frontmatter["description"] == "Plan trips around traveler style and current facts."
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert frontmatter["platforms"] == "[linux, macos, windows]"
    assert "Nico Allen" in frontmatter["author"]
    assert body.strip()


def test_travel_designer_encodes_core_modes_and_safety_rules():
    _, body = _frontmatter_and_body()

    for mode in [
        "Pick a Destination",
        "Plan the Trip",
        "Budget",
        "Pack",
        "On the Ground",
    ]:
        assert mode in body

    assert "Verify current facts" in body
    assert "recommend against" in body
    assert "TRAVELER PROFILE" in body
    assert "TRAVEL LESSONS" in body
