"""Contract tests for the bundled Hermes skill-authoring guidance."""

from __future__ import annotations

import re
from pathlib import Path

from tools.skill_manager_tool import _validate_frontmatter


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "hermes-agent-skill-authoring"
)
SKILL_MD = SKILL_DIR / "SKILL.md"


def _source() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _scalar(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}:\s*[\"']?(.*?)[\"']?\s*$",
        _source(),
        re.MULTILINE,
    )
    assert match, f"missing frontmatter field: {name}"
    return match.group(1)


def test_frontmatter_passes_runtime_validator() -> None:
    assert _validate_frontmatter(_source()) is None


def test_name_matches_skill_directory() -> None:
    assert _scalar("name") == SKILL_DIR.name


def test_description_meets_repository_hardline() -> None:
    description = _scalar("description")
    assert len(description) <= 60
    assert "\n" not in description
    assert description.endswith(".")
    assert not any(
        word in description.lower()
        for word in ("powerful", "comprehensive", "seamless", "advanced")
    )


def test_modern_sections_are_present_in_order() -> None:
    source = _source()
    headings = (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    )
    positions = [source.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_linked_local_resources_exist() -> None:
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", _source())
    local_links = [link for link in links if "://" not in link and not link.startswith("#")]
    assert local_links, "authoring skill should progressively disclose its detailed contract"
    for link in local_links:
        assert (SKILL_DIR / link).is_file(), f"broken skill resource link: {link}"


def test_guidance_has_no_developer_specific_home_path() -> None:
    for path in SKILL_DIR.rglob("*.md"):
        source = path.read_text(encoding="utf-8")
        assert "/home/" not in source
        assert "C:\\Users\\" not in source
