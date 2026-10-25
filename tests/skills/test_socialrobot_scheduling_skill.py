"""HARDLINE checks for the socialrobot-scheduling optional skill."""

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "social-media" / "socialrobot-scheduling"
SKILL_MD = SKILL_DIR / "SKILL.md"

KNOWN_MCP_TOOLS = {
    "list_connected_accounts",
    "get_media_upload_url",
    "create_post",
    "list_posts",
    "delete_post",
    "update_post",
    "reschedule_post",
    "get_account_analytics",
    "get_post_analytics",
    "get_posts_with_analytics",
    "get_follower_demographics",
    "instagram_best_post_times",
    "pinterest_list_boards",
    "pinterest_create_board",
    "tiktok_get_creator_info",
    "linkedin_search_geo_locations",
    "linkedin_search_people_mentions",
    "linkedin_search_organizations",
}

ALLOWED_PROTOCOL_METHODS = {"skills/get", "skills/list", "resources/read"}

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
    assert "socialrobot-scheduling" not in lower, "description repeats the skill name"


def test_frontmatter_required_fields(skill_text):
    fields = _frontmatter(skill_text)
    assert fields["name"] == "socialrobot-scheduling"
    assert fields.get("version") == "0.1.0", "new skills start at 0.1.0"
    assert fields.get("author", "").startswith("Nicolas Torres"), "author must credit the human first"
    assert fields.get("author") != "Hermes Agent", "author must not be Hermes Agent alone"
    assert fields.get("license") == "MIT"
    assert fields.get("platforms") == "[linux, macos, windows]", "platforms must be declared"


def test_modern_section_order(skill_text):
    positions = [skill_text.find(section) for section in REQUIRED_SECTIONS]
    assert all(p >= 0 for p in positions), "a required section is missing"
    assert positions == sorted(positions), "sections must appear in canonical order"


def test_referenced_tools_exist(skill_text):
    referenced = set()
    for token in re.findall(r"`([^`]+)`", skill_text):
        if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)+", token):
            referenced.add(token)
    unknown = referenced - KNOWN_MCP_TOOLS - ALLOWED_PROTOCOL_METHODS
    assert not unknown, f"referenced tools not in the SocialRobot MCP surface: {sorted(unknown)}"


def test_no_secrets_in_skill(skill_text):
    assert "sk-" not in skill_text
    assert "ghp_" not in skill_text
    assert re.search(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}", skill_text) is None


def test_verification_section_mentions_list_posts(skill_text):
    assert "list_posts" in skill_text.split("## Verification", 1)[1]
