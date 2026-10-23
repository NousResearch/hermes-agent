"""Contract and methodology tests for the research-paper-writing skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "research" / "research-paper-writing" / "SKILL.md"

PIPELINE_PHASES = [
    "## Phase 0: Project Setup",
    "## Phase 1: Literature Review",
    "## Phase 2: Experiment Design",
    "## Phase 3: Experiment Execution & Monitoring",
    "## Phase 4: Result Analysis",
    "## Phase 5: Paper Drafting",
    "## Phase 6: Self-Review & Revision",
    "## Phase 7: Submission Preparation",
    "## Phase 8: Post-Acceptance Deliverables",
]

TARGET_VENUES = ["NeurIPS", "ICML", "ICLR", "ACL", "AAAI", "COLM"]


class TestResearchPaperWritingContract:
    """Test that research-paper-writing adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "research-paper-writing"
        assert fm.get("dependencies")
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


class TestResearchPaperWritingContentAndStructure:
    """Test research paper writing pipeline phases, core philosophy, and conference venues."""

    def test_pipeline_phases_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(phase) for phase in PIPELINE_PHASES]
        assert positions == sorted(positions), "phases must follow 0-8 research lifecycle order"

    def test_target_venues_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for venue in TARGET_VENUES:
            assert venue in body, f"target venue {venue} must be covered in the paper pipeline"

    def test_core_philosophy_rules_present(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Never hallucinate citations" in body
        assert "Paper is a story" in body
        assert "Experiments serve claims" in body
        assert "Commit early, commit often" in body

    def test_proactivity_and_collaboration_matrix(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Proactivity and Collaboration" in body
        assert "Draft first, ask with the draft" in body
