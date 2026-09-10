"""Contract and methodology discipline tests for the plan skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = (
    REPO_ROOT
    / "skills"
    / "software-development"
    / "plan"
    / "SKILL.md"
)

WRITING_PROCESS_STEPS = [
    "### Step 1: Understand Requirements",
    "### Step 2: Explore the Codebase",
    "### Step 3: Design Approach",
    "### Step 4: Write Tasks",
    "### Step 5: Add Complete Details",
    "### Step 6: Review the Plan",
]

CORE_PRINCIPLES = [
    "DRY (Don't Repeat Yourself)",
    "YAGNI (You Aren't Gonna Need It)",
    "Frequent Commits",
]


class TestPlanSkillContract:
    """Test that SKILL.md adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "plan"
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

    def test_writing_process_steps_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(step) for step in WRITING_PROCESS_STEPS]
        assert positions == sorted(positions), "writing process steps must be sequential 1-6"

    def test_plan_location_and_format_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert ".hermes/plans/" in body, "plan file must be stored in .hermes/plans/"
        assert ".hermes/plans/YYYY-MM-DD_HHMMSS-<slug>.md" in body

    def test_task_granularity_and_estimates(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "2-5 minutes" in body or "2–5 minutes" in body
        assert "Bite-Sized" in body or "bite-sized" in body

    def test_engineering_principles_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for principle in CORE_PRINCIPLES:
            assert principle in body, f"engineering principle {principle} must be present"
