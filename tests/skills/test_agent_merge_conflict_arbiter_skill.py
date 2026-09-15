"""Contract checks for the optional agent-merge-conflict-arbiter skill asset.

Reads only the SKILL.md markdown asset — no .py source reads, no network.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "autonomous-ai-agents"
    / "agent-merge-conflict-arbiter"
    / "SKILL.md"
)

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must open with frontmatter"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    fm_text = content[3 : m.start() + 3]
    body = content[m.end() + 3 :]
    fm = {}
    for line in fm_text.splitlines():
        km = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if km:
            fm[km.group(1)] = km.group(2).strip().strip('"')
    return fm, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "agent-merge-conflict-arbiter"


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith("."), "description must end with a period"
    assert desc.count(".") == 1, "description must be one sentence"


def test_required_sections_present_in_order():
    _, body = _frontmatter_and_body()
    positions = []
    for section in REQUIRED_SECTIONS:
        idx = body.find(section)
        assert idx != -1, f"missing section: {section}"
        positions.append(idx)
    assert positions == sorted(positions), "sections out of required order"


def test_body_size_within_bundled_norms():
    content = SKILL_PATH.read_text(encoding="utf-8")
    lines = content.count("\n") + 1
    assert 80 <= lines <= 260, f"SKILL.md is {lines} lines; expected ~100-200"


def test_references_native_hermes_tools():
    _, body = _frontmatter_and_body()
    for tool in ("`terminal`", "`read_file`", "`patch`", "`delegate_task`"):
        assert tool in body, f"body must reference native tool {tool}"


def test_classification_taxonomy_present():
    _, body = _frontmatter_and_body()
    for cls in (
        "disjoint-intent",
        "same-question-different-answer",
        "superseded",
    ):
        assert cls in body, f"missing hunk class: {cls}"


def test_impartiality_contract_stated():
    _, body = _frontmatter_and_body()
    assert "never favor" in body.lower()
    assert "drive-by" in body.lower()


def test_no_machine_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "/home/" not in content
