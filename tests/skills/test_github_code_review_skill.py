"""Contract and methodology tests for the github-code-review skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "github" / "github-code-review" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## Prerequisites",
    "## 1. Reviewing Local Changes (Pre-Push)",
    "## 2. Reviewing a Pull Request on GitHub",
    "## 3. Review Checklist",
    "## 4. Pre-Push Review Workflow",
    "## 5. PR Review Workflow (End-to-End)",
]

CHECKLIST_CATEGORIES = [
    "### Correctness",
    "### Security",
    "### Code Quality",
    "### Testing",
    "### Performance",
    "### Documentation",
]

REVIEW_OUTPUT_SECTIONS = [
    "### Critical",
    "### Warnings",
    "### Suggestions",
    "### Looks Good",
]


class TestGithubCodeReviewContract:
    """Test that github-code-review adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "github-code-review"
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


class TestGithubCodeReviewContentAndStructure:
    """Test review workflows, checklist categories, and output standards."""

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow structured workflow order"

    def test_review_checklist_comprehensive(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for category in CHECKLIST_CATEGORIES:
            assert category in body, f"checklist category {category} must be present"

    def test_structured_output_format(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for section in REVIEW_OUTPUT_SECTIONS:
            assert section in body, f"review output section {section} must be defined"

    def test_formal_review_events_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "APPROVE" in body
        assert "REQUEST_CHANGES" in body
        assert "COMMENT" in body

    def test_local_git_diff_inspection_commands(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "git diff --staged" in body
        assert "git diff main...HEAD" in body
        assert "git diff main...HEAD --stat" in body
