"""HARDLINE checks for the socialrobot-campaign-report optional skill."""

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "social-media" / "socialrobot-campaign-report"
SKILL_MD = SKILL_DIR / "SKILL.md"

REPORT_TOOLS = {
    "list_connected_accounts",
    "get_posts_with_analytics",
    "get_account_analytics",
    "get_follower_demographics",
}

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

BANNED_MARKETING_WORDS = ["powerful", "comprehensive", "seamless", "advanced"]


def _frontmatter(text):
    match = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
    assert match, "no YAML frontmatter block found"
    fields = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


@pytest.fixture(scope="module")
def skill_text():
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    return SKILL_MD.read_text()


def test_description_harline(skill_text):
    fields = _frontmatter(skill_text)
    desc = fields["description"]
    assert len(desc) <= 60, f"description too long: {len(desc)} chars"
    assert desc.endswith("."), "description must end with a period"
    assert desc.count(". ") == 0, "description must be one sentence"
    lower = desc.lower()
    for word in BANNED_MARKETING_WORDS:
        assert word not in lower, f"banned marketing word: {word}"
    assert "socialrobot-campaign-report" not in lower, "description repeats the skill name"


def test_frontmatter_required_fields(skill_text):
    fields = _frontmatter(skill_text)
    assert fields["name"] == "socialrobot-campaign-report"
    assert fields.get("version") == "0.1.0", "new skills start at 0.1.0"
    assert fields.get("author", "").startswith("Nicolas Torres"), "author must credit the human first"
    assert fields.get("author") != "Hermes Agent", "author must not be Hermes Agent alone"
    assert fields.get("license") == "MIT"
    assert fields.get("platforms") == "[linux, macos, windows]", "platforms must be declared"


def test_modern_section_order(skill_text):
    positions = [skill_text.find(section) for section in REQUIRED_SECTIONS]
    assert all(p >= 0 for p in positions), "a required section is missing"
    assert positions == sorted(positions), "sections must appear in canonical order"


def test_report_tools_are_referenced(skill_text):
    for tool in REPORT_TOOLS:
        assert tool in skill_text, f"missing reference to {tool}"


def test_explicit_window_required(skill_text):
    assert "window" in skill_text
    assert "start" in skill_text.lower() and "end" in skill_text.lower()


def test_no_secrets_in_skill(skill_text):
    assert "sk-" not in skill_text
    assert "ghp_" not in skill_text
    assert re.search(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}", skill_text) is None
