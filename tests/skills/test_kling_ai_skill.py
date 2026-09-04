from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "creative" / "kling-ai"
SKILL = SKILL_DIR / "SKILL.md"


def _frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    assert match is not None, f"missing {key}"
    return match.group(1).strip().strip('"')


def test_kling_ai_skill_metadata() -> None:
    text = SKILL.read_text(encoding="utf-8")
    description = _frontmatter_value(text, "description")

    assert _frontmatter_value(text, "name") == "kling-ai"
    assert len(description) <= 60
    assert description.endswith(".")
    assert _frontmatter_value(text, "author").startswith("William (@Wlain)")
    assert "category: creative" in text


def test_kling_ai_skill_contract_and_structure() -> None:
    text = SKILL.read_text(encoding="utf-8")
    headings = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    assert "# Kling AI Skill" in text
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    for phrase in (
        "Plugin-Hermes-kling-ai",
        "https://kling.ai/mcp",
        "credit-consuming",
        "at most once",
        "generationId",
        "taskTraceId",
        "text fallback",
    ):
        assert phrase in text

    assert "https://klingai.com/mcp" not in text

    assert (SKILL_DIR / "references" / "tool-workflows.md").is_file()
    assert (SKILL_DIR / "references" / "troubleshooting.md").is_file()
