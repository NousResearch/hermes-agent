"""Contract and CLI tool tests for the docx productivity skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "productivity" / "docx" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
]

DOCX_SCRIPTS = [
    "docx_create.py",
    "docx_read.py",
    "docx_edit.py",
    "docx_template.py",
    "docx_revisions.py",
    "docx_comments.py",
    "docx_validate.py",
]


class TestDocxContract:
    """Test that docx skill adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "docx"
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


class TestDocxContentAndStructure:
    """Test docx document operations, prerequisites, and CLI helper scripts."""

    def test_required_sections_present(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for section in REQUIRED_SECTIONS:
            assert section in body, f"section {section} must be present"

    def test_python_docx_dependency_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "python-docx" in body, "must document python-docx requirement"

    def test_helper_scripts_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for script in DOCX_SCRIPTS:
            assert script in body, f"helper script {script} must be documented"
