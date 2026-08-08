"""Regression coverage for the bundled Parallel Orchestrator skill."""

import json
import re
from pathlib import Path

from tools import skills_tool
from tools.skill_manager_tool import _validate_frontmatter


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "autonomous-ai-agents" / "parallel-orchestrator"
SKILL_MD = SKILL_DIR / "SKILL.md"


def _skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_parallel_orchestrator_skill_frontmatter_is_valid():
    content = _skill_text()

    assert _validate_frontmatter(content) is None
    assert "name: parallel-orchestrator" in content
    description = re.search(r'^description: "(.+)"$', content, re.MULTILINE)
    assert description is not None
    assert len(description.group(1)) < 60
    assert "author: web3blind + Hermes Agent" in content
    assert "aliases:" in content
    assert "  - параллельно" in content
    assert "  - распараллелить" in content
    assert "  - быстрее" not in content


def test_parallel_orchestrator_skill_is_discoverable(monkeypatch, tmp_path):
    local_skills = tmp_path / "skills"
    local_skills.mkdir()
    bundled_skills = REPO_ROOT / "skills"

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", local_skills)
    monkeypatch.setattr(
        "agent.skill_utils.get_external_skills_dirs",
        lambda: [bundled_skills],
    )

    listed = json.loads(skills_tool.skills_list("autonomous-ai-agents"))
    assert listed["success"] is True
    assert any(skill["name"] == "parallel-orchestrator" for skill in listed["skills"])

    viewed = json.loads(skills_tool.skill_view("parallel-orchestrator"))
    assert viewed["success"] is True
    assert viewed["path"].endswith("parallel-orchestrator/SKILL.md")
    assert "delegate_task(tasks=[...])" in viewed["content"]


def test_parallel_orchestrator_documents_safety_and_batch_contracts():
    content = _skill_text()

    required_phrases = [
        "Safe by Default: Read-Only Fan-Out",
        "Hard Stop: External Side Effects",
        "Do not parallelize automatically if children may write to the same repo",
        "Never auto-parallelize external side effects",
        "Use one tool call with `tasks=[...]` instead of serial child calls.",
        "No child was asked to perform external side effects.",
        "Results were synthesized, not pasted.",
        "If there are more objects than the limit",
        "Parent Tool Availability",
        "Children inherit the parent run's enabled tool surface",
        "Do not document or pass model-facing per-task tool restrictions",
        "When children can use `terminal`, constrain commands to read-only inspection",
        "Do not let children install packages, run formatters, update lockfiles",
    ]
    for phrase in required_phrases:
        assert phrase in content

    forbidden_vendor_references = ["OpenCode", "opencode", "OpenClaw", "openclaw"]
    for phrase in forbidden_vendor_references:
        assert phrase not in content

    assert '"toolsets"' not in content


def test_parallel_orchestrator_documents_delegate_task_limitations():
    content = _skill_text()

    required_phrases = [
        "cancelled if the parent turn is interrupted",
        "Children cannot clarify with the user",
        "Subagent results are leads, not verified facts",
        "Use cron jobs or background terminal processes for durable monitoring",
    ]
    for phrase in required_phrases:
        assert phrase in content


def test_parallel_orchestrator_has_output_and_done_sections():
    content = _skill_text()

    required_sections = [
        "## Output Contract",
        "## Quick Test Checklist",
        "## Done Criteria",
    ]
    for section in required_sections:
        assert section in content
