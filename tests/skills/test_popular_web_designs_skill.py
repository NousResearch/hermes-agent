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
