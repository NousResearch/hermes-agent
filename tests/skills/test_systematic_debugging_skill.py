"""Contract tests for the bundled systematic-debugging skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = (
    REPO_ROOT
    / "skills"
    / "software-development"
    / "systematic-debugging"
    / "SKILL.md"
)

REQUIRED_PHASES = [
    "## Phase 1: Root Cause Investigation",
    "## Phase 2: Pattern Analysis",
    "## Phase 3: Hypothesis and Testing",
    "## Phase 4: Implementation",
]

NATIVE_TOOLS = [
    "search_files",
    "read_file",
    "terminal",
    "web_search",
    "web_extract",
]


class TestSystematicDebuggingSkillContract:
    """Test that SKILL.md adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "systematic-debugging"
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

    def test_four_phases_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(phase) for phase in REQUIRED_PHASES]
        assert positions == sorted(positions), "phases must follow 1 -> 2 -> 3 -> 4 sequence"

    def test_iron_law_and_disciplines_present(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST" in body
        assert "Feedback Loop" in body
        assert "Rule of Three" in body or "Rule of three" in body.lower()
        assert "Question Architecture" in body or "question fundamentals" in body.lower()
        assert "delegate_task" in body

    @pytest.mark.parametrize("tool_name", NATIVE_TOOLS)
    def test_documents_native_hermes_tools(self, tool_name):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert f"`{tool_name}`" in body, f"native tool {tool_name} must be documented"
