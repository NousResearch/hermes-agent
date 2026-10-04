"""Contract and workflow discipline tests for the session-librarian skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "productivity" / "session-librarian" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## When to Use",
    "## The Two Surfaces",
    "## Procedure",
    "## Parallel Workstreams",
    "## Pitfalls",
    "## Verification",
]

PROCEDURE_STEPS = [
    "① **Discover.**",
    "② **Summarize per session.**",
    "③ **Plan before acting",
    "④ **Act with the safest primitive.**",
    "⑤ **Report.**",
]


class TestSessionLibrarianContract:
    """Test that session-librarian adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "session-librarian"
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


class TestSessionLibrarianContentAndStructure:
    """Test session librarian procedures, safety gates, and dry-run rules."""

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow structured order"

    def test_procedure_steps_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(step) for step in PROCEDURE_STEPS]
        assert positions == sorted(positions), "procedure steps must follow 1-5 sequence"

    def test_safety_and_dry_run_disciplines(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "--dry-run" in body, "must require --dry-run before destructive mutation"
        assert "--yes" in body, "must document --yes for confirmed deletion"
        assert "export" in body, "must offer session export backup before pruning"
        assert "archive" in body, "must prefer reversible archive over hard delete"

    def test_parallel_workstreams_uses_delegation(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "delegate_task" in body, "must use delegate_task for parallel workstreams"
