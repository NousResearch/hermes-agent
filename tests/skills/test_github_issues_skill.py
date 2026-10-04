"""Contract and workflow tests for the github-issues skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "github" / "github-issues" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## Prerequisites",
    "## 1. Viewing Issues",
    "## 2. Creating Issues",
    "## 3. Managing Issues",
    "## 4. Issue Triage Workflow",
    "## 5. Bulk Operations",
    "## Quick Reference Table",
]

LINKING_KEYWORDS = ["Closes #", "Fixes #", "Resolves #"]


class TestGithubIssuesContract:
    """Test that github-issues adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "github-issues"
        hermes = fm["metadata"]["hermes"]
        assert hermes["tags"]
        assert "related_skills" in hermes

    def test_description_hardline(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        desc = fm["description"]
        assert len(desc) <= 60, f"description is {len(desc)} chars; max allowed is 60"
        assert desc.endswith("."), "description must end with a period"
        assert not re.search(
            r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
            desc,
            re.I,
        )

    def test_related_skills_resolve_in_repo(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        missing = resolve_related_skills_in_repo(fm["metadata"]["hermes"]["related_skills"])
        assert not missing, f"related_skills entries do not resolve in repo: {missing}"

    def test_no_machine_local_paths(self):
        content = SKILL_MD.read_text(encoding="utf-8")
        assert "/home/" not in content
        assert not re.search(r"[A-Z]:\\\\Users", content)


class TestGithubIssuesContentAndStructure:
    """Test issue management workflows and templates in github-issues."""

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow structured workflow order"

    def test_dual_mode_commands_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "gh issue list" in body
        assert "gh issue create" in body
        assert "gh issue edit" in body
        assert "gh issue comment" in body
        assert "gh issue close" in body
        assert "api.github.com/repos" in body

    def test_issue_templates_included(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        # Bug report template
        assert "## Bug Description" in body
        assert "## Steps to Reproduce" in body
        assert "## Expected Behavior" in body
        # Feature request template
        assert "## Feature Description" in body
        assert "## Motivation" in body
        assert "## Proposed Solution" in body

    def test_pr_linking_keywords_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for kw in LINKING_KEYWORDS:
            assert kw in body, f"keyword {kw} must be documented for issue linking"

    def test_triage_workflow_steps(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Issue Triage Workflow" in body
        assert "needs-triage" in body
