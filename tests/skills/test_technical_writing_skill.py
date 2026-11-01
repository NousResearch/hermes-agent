"""Tests for the technical-writing bundled skill.

Asserts the hardline SKILL.md contract (frontmatter, description length,
human-first author, modern section order) and that prose-grounded repo
paths cited by the skill still exist on the branch.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT / "skills" / "software-development" / "technical-writing" / "SKILL.md"
)

MODERN_SECTIONS = (
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)

GROUNDED_PATHS = (
    REPO_ROOT / "tools" / "skills_guard.py",
    REPO_ROOT / "hermes_cli" / "skills_hub.py",
    REPO_ROOT / "scripts" / "run_tests.sh",
    REPO_ROOT / "CONTRIBUTING.md",
)


def _frontmatter_and_body() -> tuple[dict, str]:
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must start with ---"
    match = re.search(r"\n---\s*\n", content[3:])
    assert match, "frontmatter must close with ---"
    frontmatter = yaml.safe_load(content[3 : match.start() + 3])
    assert isinstance(frontmatter, dict), "frontmatter must be a YAML mapping"
    body = content[match.end() + 3 :]
    return frontmatter, body


def test_skill_file_exists() -> None:
    assert SKILL_PATH.is_file()


def test_required_frontmatter_fields() -> None:
    frontmatter, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in frontmatter, f"missing frontmatter field: {field}"
    assert frontmatter["name"] == "technical-writing"
    hermes = (frontmatter.get("metadata") or {}).get("hermes") or {}
    assert hermes.get("tags"), "metadata.hermes.tags must be present"


def test_description_hardline() -> None:
    frontmatter, _ = _frontmatter_and_body()
    description = str(frontmatter["description"])
    assert len(description) <= 60, (
        f"description is {len(description)} chars; hardline is 60: {description!r}"
    )
    assert description.rstrip().endswith("."), "description must end with a period"
    assert description.count(".") == 1, "description must be one sentence"


def test_author_credits_human_first() -> None:
    frontmatter, _ = _frontmatter_and_body()
    author = str(frontmatter["author"])
    assert not author.startswith("Hermes Agent"), (
        "author must credit the human contributor first"
    )
    assert "mcpeezy" in author
    assert "Hermes Agent" in author


def test_modern_section_order() -> None:
    _, body = _frontmatter_and_body()
    positions = []
    for heading in MODERN_SECTIONS:
        index = body.find(heading)
        assert index != -1, f"SKILL.md missing section: {heading}"
        positions.append(index)
    assert positions == sorted(positions), "modern sections must appear in standard order"


def test_related_skills_resolve_in_repo() -> None:
    frontmatter, _ = _frontmatter_and_body()
    hermes = (frontmatter.get("metadata") or {}).get("hermes") or {}
    for name in hermes.get("related_skills") or []:
        hits = (
            list(REPO_ROOT.glob(f"skills/*/{name}/SKILL.md"))
            + list(REPO_ROOT.glob(f"skills/*/*/{name}/SKILL.md"))
            + list(REPO_ROOT.glob(f"optional-skills/*/{name}/SKILL.md"))
            + list(REPO_ROOT.glob(f"optional-skills/*/*/{name}/SKILL.md"))
        )
        assert hits, f"related_skills entry does not resolve in-repo: {name}"


def test_grounded_repo_paths_exist() -> None:
    for path in GROUNDED_PATHS:
        assert path.is_file(), f"skill-grounded path missing: {path.relative_to(REPO_ROOT)}"


def test_body_names_guard_and_hub_symbols() -> None:
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "tools/skills_guard.py" in content
    assert "hermes_cli/skills_hub.py" in content
    assert "format_scan_report()" in content
    assert "should_allow_install" in content


def test_contributing_points_at_skill() -> None:
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "skills/software-development/technical-writing/SKILL.md" in contributing
    assert "not for product UI copy" in contributing


def test_no_machine_local_paths() -> None:
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert not re.search(r"/home/(?!runner\b)[a-z0-9_-]+/", content)
    assert "C:\\Users\\" not in content
