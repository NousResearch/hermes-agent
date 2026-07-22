"""Contract tests for skills/autonomous-ai-agents/hermes-claude-codex-workstream."""

import re
from pathlib import Path

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "hermes-claude-codex-workstream"
)
SKILL_MD = SKILL_DIR / "SKILL.md"


def _frontmatter() -> dict:
    text = SKILL_MD.read_text(encoding="utf-8")
    body = text.split("---", 2)[1]
    fields = {}
    for line in body.splitlines():
        m = re.match(r"^(\w+): (.+)$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def test_description_is_within_authoring_limit():
    desc = _frontmatter()["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars, limit is 60"
    assert desc.endswith("."), "description must be one sentence ending with a period"


def test_author_credits_the_human_contributor_first():
    author = _frontmatter()["author"]
    assert not author.startswith("Hermes Agent"), (
        "external contributions must credit the human contributor first"
    )
    assert "@joelbrilliant" in author


def test_declared_platforms_match_path_strategy():
    fields = _frontmatter()
    platforms = fields["platforms"]
    text = SKILL_MD.read_text(encoding="utf-8")
    # The workflow is bash-based, so Windows must not be declared, and no
    # path may be hard-coded under /tmp (worktrees are repo siblings, run
    # artefacts live under the repo).
    assert "windows" not in platforms
    assert "/tmp/" not in text, "no hard-coded /tmp paths; use RUN_DIR/WORKTREE"


def test_run_dir_is_defined_absolute_before_use():
    text = SKILL_MD.read_text(encoding="utf-8")
    definition = text.find('RUN_DIR="$REPO_ROOT/')
    first_use = text.find('"$RUN_DIR/')
    assert definition != -1, "RUN_DIR must be defined from REPO_ROOT (absolute)"
    assert first_use == -1 or definition < first_use, (
        "RUN_DIR must be defined before any use"
    )
    # No stale repo-relative artefact references (the old bug: a worktree
    # command referring to a path only valid from the repo root).
    assert not re.search(r"(?<![$\w/])\.hermes/runs/\$RUN_ID", text), (
        "artefact references must go through the absolute $RUN_DIR"
    )


def test_required_sections_present_in_order():
    text = SKILL_MD.read_text(encoding="utf-8")
    sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [text.find(s) for s in sections]
    assert all(p != -1 for p in positions), (
        f"missing sections: {[s for s, p in zip(sections, positions) if p == -1]}"
    )
    assert positions == sorted(positions), "sections must follow the modern order"


def test_referenced_files_exist():
    text = SKILL_MD.read_text(encoding="utf-8")
    for rel in re.findall(r"\]\((references/[^)]+)\)", text):
        assert (SKILL_DIR / rel).is_file(), f"broken reference link: {rel}"
