"""Contract and workflow discipline tests for the github-pr-workflow skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "github" / "github-pr-workflow" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## Prerequisites",
    "## 1. Branch Creation",
    "## 2. Making Commits",
    "## 3. Pushing and Creating a PR",
    "## 4. Monitoring CI Status",
    "## 5. Auto-Fixing CI Failures",
    "## 6. Merging",
    "## 7. Complete Workflow Example",
    "## Useful PR Commands Reference",
]

CONVENTIONAL_COMMIT_TYPES = [
    "feat",
    "fix",
    "refactor",
    "docs",
    "test",
    "ci",
    "chore",
    "perf",
]


class TestGithubPrWorkflowContract:
    """Test that github-pr-workflow adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "github-pr-workflow"
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


class TestGithubPrWorkflowContentAndStructure:
    """Test workflow instructions and dual-mode tooling in github-pr-workflow."""

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow chronological PR lifecycle order"

    def test_dual_mode_tooling_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        # CLI tool gh
        assert "gh pr create" in body
        assert "gh pr checks" in body
        assert "gh pr merge" in body
        # REST API / curl fallback
        assert "api.github.com/repos" in body
        assert "GITHUB_TOKEN" in body

    def test_conventional_commits_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Conventional Commits" in body
        for commit_type in CONVENTIONAL_COMMIT_TYPES:
            assert f"`{commit_type}`" in body, f"commit type {commit_type} must be listed"

    def test_ci_auto_fix_loop_discipline(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Auto-Fix Loop Pattern" in body
        assert "3 attempts" in body or "up to 3" in body.lower()

    def test_merge_methods_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "--squash" in body or '"squash"' in body
        assert "--delete-branch" in body or "git push origin --delete" in body
