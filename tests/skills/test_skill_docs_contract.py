"""Contract: the committed skill docs stay in sync with the shipped skill tree.

``website/scripts/generate-skill-docs.py`` is the documented source of both catalogs and of the
per-skill pages under ``website/docs/user-guide/skills/``. The *Docs Site Checks* workflow
regenerates them before building the site, but nothing compares that output with what is
committed, so the committed copies — which GitHub renders and which contributors read and copy
from — drifted:

- ``optional-skills-catalog.md`` was missing ``agent-merge-conflict-arbiter`` and listed
  ``pr-lens`` under ``blockchain`` instead of ``software-development``;
- both catalogs and 196 per-skill pages carried Windows path separators in the ``Path`` column
  (``skills/apple\\apple-notes``) and inside GitHub blob links, where a backslash makes the URL
  resolve to nothing.

These tests assert the relationships the generator guarantees, so the next skill that ships
without a catalog entry (or a page regenerated on a Windows host) fails here instead of on the
published page. They read the shipped ``.md`` artifacts and the skill tree — never source text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOGS = {
    "skills": REPO_ROOT / "website" / "docs" / "reference" / "skills-catalog.md",
    "optional-skills": REPO_ROOT / "website" / "docs" / "reference" / "optional-skills-catalog.md",
}
SKILL_PAGES = REPO_ROOT / "website" / "docs" / "user-guide" / "skills"

_ROW_RE = re.compile(r"^\|\s*\[(?:\*\*)?`?([^`*\]]+)`?(?:\*\*)?\]\(([^)]+)\)", re.MULTILINE)
_PATH_ROW_RE = re.compile(r"^\|\s*Path\s*\|\s*`([^`]+)`", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\]\((\.[^)]+\.md)(?:#[^)]*)?\)")


def _shipped_skills(root: str) -> set[str]:
    return {p.parent.name for p in (REPO_ROOT / root).rglob("SKILL.md")}


def _catalog_rows(path: Path) -> dict[str, str]:
    return {m.group(1).strip(): m.group(2).strip() for m in _ROW_RE.finditer(path.read_text(encoding="utf-8"))}


@pytest.mark.parametrize("root", sorted(CATALOGS))
def test_catalog_lists_every_shipped_skill(root):
    """Every skill shipped in the tree has a row in its catalog."""
    listed = set(_catalog_rows(CATALOGS[root]))
    missing = sorted(_shipped_skills(root) - listed)
    assert not missing, f"{CATALOGS[root].name} does not list shipped skills: {missing}"


@pytest.mark.parametrize("root", sorted(CATALOGS))
def test_catalog_rows_resolve_to_a_shipped_skill(root):
    """No catalog row outlives the skill it documents."""
    shipped = _shipped_skills(root)
    dangling = sorted(name for name in _catalog_rows(CATALOGS[root]) if name not in shipped)
    assert not dangling, f"{CATALOGS[root].name} lists skills that no longer ship: {dangling}"


@pytest.mark.parametrize("root", sorted(CATALOGS))
def test_catalog_links_resolve(root):
    """Every relative link in a catalog points at a file that exists."""
    catalog = CATALOGS[root]
    broken = sorted(
        target for target in set(_MD_LINK_RE.findall(catalog.read_text(encoding="utf-8")))
        if not (catalog.parent / target).resolve().exists()
    )
    assert not broken, f"{catalog.name} has dangling links: {broken}"


def test_generated_pages_use_posix_separators():
    """A ``Path`` row or relative link carrying a backslash means the page was not regenerated.

    The generator emits POSIX separators; a backslash survives only in a stale copy, and inside a
    GitHub blob link it is a silently broken URL.
    """
    offenders: list[str] = []
    for page in sorted(SKILL_PAGES.rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        for bad in _PATH_ROW_RE.findall(text):
            if "\\" in bad:
                offenders.append(f"{page.relative_to(REPO_ROOT)}: Path `{bad}`")
        for target in _MD_LINK_RE.findall(text):
            if "\\" in target:
                offenders.append(f"{page.relative_to(REPO_ROOT)}: link ({target})")
    for catalog in CATALOGS.values():
        for bad in _PATH_ROW_RE.findall(catalog.read_text(encoding="utf-8")):
            if "\\" in bad:
                offenders.append(f"{catalog.name}: Path `{bad}`")
    assert not offenders, "Windows separators in generated docs:\n" + "\n".join(offenders[:20])
