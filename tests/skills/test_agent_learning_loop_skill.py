"""Contract tests for the optional agent-learning-loop skill."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills/mlops/agent-learning-loop/SKILL.md"
GUIDE = ROOT / "website/docs/developer-guide/agent-learning-loop.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_description_matches_authoring_standard():
    text = _text(SKILL)
    match = re.search(r'^description: ["\']?(.*?)["\']?$', text, re.MULTILINE)
    assert match is not None
    description = match.group(1)
    assert len(description) <= 60
    assert description.endswith(".")


def test_required_sections_are_present_in_order():
    text = _text(SKILL)
    headings = [
        "# Agent Learning Loop Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_docs_use_profile_safe_state_paths():
    combined = _text(SKILL) + _text(GUIDE)
    assert "$HERMES_HOME" in combined
    assert "`~/.hermes" not in combined


def test_architecture_example_is_not_presented_as_executable():
    text = _text(SKILL)
    assert "not a bundled executable" in text
    assert "python scripts/" not in text
