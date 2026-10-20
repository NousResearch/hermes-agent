"""Contract: every skill this skill points at actually ships.

SKILL.md and its templates are instructions the agent executes verbatim. A
reference to a skill that is not in the tree dead-ends every run that follows
it — the `generative-widgets` tunnel instruction did exactly that until it was
replaced with `browser_navigate` + `browser_vision`. This catches the same
class of dangling reference in either direction (SKILL.md or any template).
"""
from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "creative" / "popular-web-designs"

SHIPPED_SKILLS = {
    p.parent.name
    for root in ("skills", "optional-skills")
    for p in (REPO / root).rglob("SKILL.md")
}

DOC_FILES = [SKILL_DIR / "SKILL.md", *sorted((SKILL_DIR / "templates").glob("*.md"))]

# "`name` skill" (prose) and skill_view(name="name") (the loader invocation).
_SKILL_REF = re.compile(
    r'`([a-z0-9][a-z0-9._-]*)`\s+skill|skill_view\(name=["\']([a-z0-9._-]+)["\']'
)


def _references(text: str) -> set[str]:
    return {a or b for a, b in _SKILL_REF.findall(text)}


def test_doc_files_are_present():
    """The skill ships SKILL.md plus its template directory (guards the glob)."""
    assert (SKILL_DIR / "SKILL.md").exists()
    assert len(DOC_FILES) > 1, "templates/ glob matched nothing"


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_skill_references_resolve(path: Path):
    dangling = sorted(_references(path.read_text(encoding="utf-8")) - SHIPPED_SKILLS)
    assert not dangling, (
        f"{path.relative_to(REPO)} points at skills that do not ship: {dangling}"
    )


# The judgment half of the authoring standards (skills/AGENTS.md #5) is not
# machine-enforced repo-wide, but once a skill has been modernised to the
# section order the sections have to stay — dropping them silently reverts it.
MODERN_SECTIONS = ["## When to Use", "## Prerequisites", "## How to Run", "## Pitfalls", "## Verification"]


def test_modern_section_order():
    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    positions = [body.find(h) for h in MODERN_SECTIONS]
    missing = [h for h, pos in zip(MODERN_SECTIONS, positions) if pos < 0]
    assert not missing, f"SKILL.md is missing standard sections: {missing}"
    assert positions == sorted(positions), (
        f"sections out of standard order: {list(zip(MODERN_SECTIONS, positions))}"
    )


def test_block_title_matches_templates():
    """SKILL.md must call the template notes block what the templates call it.

    SKILL.md used to say "Hermes Implementation Notes" while all 54 templates
    title theirs "Hermes Agent — Implementation Notes" — a name the agent
    cannot grep for in either direction.
    """
    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    named = set(re.findall(r"\*\*([^*\n]*Implementation Notes)\*\*", body))
    titles = set()
    for path in (SKILL_DIR / "templates").glob("*.md"):
        m = re.search(r"^> \*\*([^*\n]*Implementation Notes)\*\*", path.read_text(encoding="utf-8"), re.M)
        if m:
            titles.add(m.group(1))
    assert len(named) == 1, f"SKILL.md names the block inconsistently: {sorted(named)}"
    assert titles == named, (
        f"SKILL.md says {sorted(named)} but templates say {sorted(titles)}"
    )
