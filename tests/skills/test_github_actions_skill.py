"""Contract tests for the bundled GitHub Actions skill."""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "github" / "github-actions" / "SKILL.md"
OLD_SKILL_MD = (
    REPO_ROOT
    / "skills"
    / "software-development"
    / "github-actions"
    / "SKILL.md"
)
REQUIRED_SECTIONS = [
    "# GitHub Actions Skill",
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]
USES_PATTERN = re.compile(r"uses:\s+([^\s#]+)(?:\s+#\s*(\S+))?")
PINNED_ACTION_PATTERN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _content() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter_value(name: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(name)}:\s*(.+)$", _content(), re.MULTILINE
    )
    assert match, f"missing {name} frontmatter"
    return match.group(1).strip()


def test_skill_is_in_github_category_only():
    assert SKILL_MD.is_file()
    assert not OLD_SKILL_MD.exists()
    assert _frontmatter_value("name") == "github-actions"


def test_description_matches_hardline_limit():
    description = _frontmatter_value("description")
    assert len(description) <= 60
    assert description.endswith(".")


def test_modern_sections_exist_in_required_order():
    content = _content()
    positions = [content.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)
    intro = content.split("# GitHub Actions Skill\n", 1)[1].split(
        "## When to Use", 1
    )[0]
    assert "It does not cover" in intro
    assert intro.count(".") >= 3


def test_every_action_reference_is_immutable_and_versioned():
    references = USES_PATTERN.findall(_content())
    assert references
    for action, version_comment in references:
        assert PINNED_ACTION_PATTERN.fullmatch(action), action
        assert version_comment.startswith("v"), action


def test_skill_defers_overlapping_operations_to_existing_skills():
    content = _content()
    assert "run diagnosis" in content
    assert "secret management" in content
    related = _frontmatter_value("related_skills")
    assert "github-pr-workflow" in related
    assert "github-repo-management" in related


def test_security_guidance_covers_high_risk_patterns():
    content = _content()
    assert "pull_request_target" in content
    assert "40-character commit SHA" in content
    assert "attacker-controlled expressions directly into shell source" in content
    assert "id-token: write" in content
    assert "timeout-minutes" in content
    assert "actions/download-artifact@" in content
    assert "name: deploy-package" in content
    assert "./scripts/deploy.sh dist/" in content
