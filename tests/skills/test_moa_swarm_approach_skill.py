"""Contract tests for the bundled MoA swarm approach skill."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "skills" / "software-development" / "moa-swarm-approach"
SKILL_PATH = SKILL_DIR / "SKILL.md"
SVG_PATH = SKILL_DIR / "assets" / "moa-swarm-workflow.svg"


def skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def frontmatter_value(name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", skill_text(), re.MULTILINE)
    assert match, f"missing frontmatter field: {name}"
    return match.group(1).strip().strip('"\'')


def test_required_files_exist() -> None:
    assert SKILL_PATH.is_file()
    assert SVG_PATH.is_file()


def test_frontmatter_meets_hardline_metadata_rules() -> None:
    description = frontmatter_value("description")
    author = frontmatter_value("author")

    assert frontmatter_value("name") == "moa-swarm-approach"
    assert len(description) <= 60
    assert description.endswith(".")
    assert "\n" not in description
    assert author.startswith("misery-hl")
    assert "Hermes Agent" in author
    assert frontmatter_value("version") == "1.0.0"
    assert frontmatter_value("license") == "MIT"


def test_body_uses_required_title_and_section_order() -> None:
    text = skill_text()
    assert "# MoA Swarm Approach Skill" in text

    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [text.index(section) for section in required]
    assert positions == sorted(positions)
    assert re.findall(r"^## .+$", text, re.MULTILINE) == required


def test_skill_is_condensed() -> None:
    line_count = len(skill_text().splitlines())
    assert 150 <= line_count <= 240, f"expected a complex skill near 200 lines, got {line_count}"


def test_related_skills_are_bundled() -> None:
    text = skill_text()
    match = re.search(r"related_skills:\s*\[([^]]+)]", text)
    assert match
    related = {item.strip() for item in match.group(1).split(",")}
    assert related == {"plan", "requesting-code-review"}

    skill_names = {
        frontmatter.group(1)
        for path in (ROOT / "skills").rglob("SKILL.md")
        if (frontmatter := re.search(r"^name:\s*([^\n]+)", path.read_text(encoding="utf-8"), re.MULTILINE))
    }
    assert related <= skill_names


def test_only_standard_build_surfaces_are_required() -> None:
    text = skill_text()
    toolsets = (ROOT / "toolsets.py").read_text(encoding="utf-8")
    core_tools = re.search(r"_HERMES_CORE_TOOLS\s*=\s*\[(.*?)]", toolsets, re.DOTALL)

    assert core_tools
    assert '"delegate_task"' in core_tools.group(1)
    assert "`delegation.provider`" in text
    assert "`delegation.model`" in text
    assert "No optional skill, plugin, or extra toolset is required" in text


def test_skill_uses_current_hermes_surfaces() -> None:
    text = skill_text()
    assert "`/moa" in text
    assert "`delegate_task`" in text
    assert "heavy-thinking-review" not in text
    assert "kanban-orchestrator" not in text


def test_no_machine_or_profile_specific_paths_remain() -> None:
    content = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in SKILL_DIR.rglob("*")
        if path.is_file()
    )
    private_path_patterns = (
        r"/Users/[^/\s]+/",
        r"/home/[^/\s]+/\.hermes/",
        r"[A-Za-z]:\\Users\\[^\\\s]+\\",
    )
    assert not any(re.search(pattern, content) for pattern in private_path_patterns)


def test_workflow_svg_is_valid_xml() -> None:
    root = ET.parse(SVG_PATH).getroot()
    assert root.tag.endswith("svg")
