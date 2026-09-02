"""Contract checks for the optional multi-agent PR convergence skill."""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = (
    REPO_ROOT
    / "optional-skills"
    / "autonomous-ai-agents"
    / "multi-agent-pr-convergence"
)
SKILL_PATH = SKILL_DIR / "SKILL.md"
REFERENCE_PATH = SKILL_DIR / "references" / "herdr-lessons.md"


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must open with frontmatter"
    match = re.search(r"\n---\s*\n", content[3:])
    assert match, "frontmatter must close with ---"
    frontmatter = content[3 : match.start() + 3]
    body = content[match.end() + 3 :]
    fields = {}
    for line in frontmatter.splitlines():
        field = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if field:
            fields[field.group(1)] = field.group(2).strip().strip('"')
    return fields, body


def test_skill_assets_exist():
    assert SKILL_PATH.is_file()
    assert REFERENCE_PATH.is_file()


def test_frontmatter_and_description_contract():
    fields, _ = _frontmatter_and_body()
    for name in ("name", "description", "version", "author", "license", "platforms"):
        assert name in fields, f"missing frontmatter field: {name}"
    assert fields["name"] == "multi-agent-pr-convergence"
    assert len(fields["description"]) <= 60
    assert fields["description"].endswith(".")
    assert "hrnbld" in fields["author"]


def test_process_sections_are_present_in_order():
    _, body = _frontmatter_and_body()
    sections = [
        "## When to Use",
        "## Prerequisites",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [body.find(section) for section in sections]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


def test_convergence_invariants_are_explicit():
    _, body = _frontmatter_and_body()
    required = (
        "exactly one writer",
        "already_upstream",
        "Every five minutes",
        "author evidence",
        "mutation or revert",
        "Fetch the remote again",
        "merge_authorized",
        "every candidate freeze",
        "writer lease",
        "lease epoch",
        "explicit owner decision",
        "identity rename/re-key",
        "same-name recreation",
        "overlapping sends",
    )
    for phrase in required:
        assert phrase in body, f"missing convergence invariant: {phrase}"


def test_herdr_reference_separates_runtime_state_from_evidence():
    reference = REFERENCE_PATH.read_text(encoding="utf-8")
    for phrase in (
        "Persistent server ownership",
        "Event-driven status",
        "idle` or `done` is not a tested commit",
        "exact SHA",
    ):
        assert phrase in reference


def test_no_machine_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8") + REFERENCE_PATH.read_text(encoding="utf-8")
    assert "/Users/" not in content
    assert "/home/" not in content
