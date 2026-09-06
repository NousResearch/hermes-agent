"""Contract tests for the optional YouTube topic scouting skill."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "research" / "youtube-topic-scouting" / "SKILL.md"


def test_skill_frontmatter_and_description_contract():
    content = SKILL.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    match = re.search(r"^description: (.+)$", content, re.MULTILINE)
    assert match
    description = match.group(1)
    assert len(description) <= 60
    assert description.endswith(".")
    assert "author: Hugo Gomes (Ukrawave), Hermes Agent" in content
    assert "platforms: [linux, macos, windows]" in content


def test_skill_enforces_evidence_and_quality_gates():
    content = SKILL.read_text(encoding="utf-8")
    for required in (
        "SOURCE CHECK:",
        "DEDUP CHECK:",
        "Claim-level evidence",
        "Source quality",
        "Content-level dedup",
        "No quota padding",
        "[SILENT]",
    ):
        assert required in content
