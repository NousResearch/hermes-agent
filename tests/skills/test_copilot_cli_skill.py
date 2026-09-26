"""Contract tests for the Copilot CLI bundled skill."""

import re
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "copilot-cli"
    / "SKILL.md"
)


def _source() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_frontmatter_meets_skill_standard():
    source = _source()
    description = re.search(r"^description: (.+)$", source, re.MULTILINE)
    author = re.search(r"^author: (.+)$", source, re.MULTILINE)

    assert description
    assert len(description.group(1)) <= 60
    assert description.group(1).endswith(".")
    assert author and author.group(1).startswith("Ken Kuang (@sykuang)")
    assert "platforms: [linux, macos, windows]" in source


def test_uses_trusted_acp_configuration():
    source = _source()

    assert "provider: copilot-acp" in source
    assert "delegate_task(" in source
    assert "acp_command=" not in source
    assert "acp_args=" not in source
    assert "toolsets=" not in source


def test_has_modern_section_order():
    source = _source()
    sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    positions = [source.index(section) for section in sections]
    assert positions == sorted(positions)
