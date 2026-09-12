"""CI enforcement of the skill authoring standards (AGENTS.md hardline).

Every bundled (skills/) and optional (optional-skills/) SKILL.md must satisfy
the programmatically-checkable subset of the authoring standards. Judgment
calls (tier placement, router-skill smell, prose quality) stay with review;
everything here is mechanical.

Pre-existing violations that need non-trivial content work are grandfathered
in the GRANDFATHER dict below. Do NOT add new entries for new skills — fix
the skill instead. Remove entries as the debt is paid down.
"""
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MARKETING = re.compile(
    r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
    re.I,
)
MACHINE_LOCAL = re.compile(r"/home/(?!runner\b)[a-z0-9_-]+/|[A-Z]:\\+Users\\+(?!<)")

# ---------------------------------------------------------------------------
# Grandfathered pre-existing debt. Shrink this list; never grow it.
# ---------------------------------------------------------------------------
GRANDFATHER: dict[str, set[str]] = {
    # (empty — the Aug 2026 sweep cleared all mechanical violations)
}


def _skill_paths():
    return sorted(
        list(REPO.glob("skills/**/SKILL.md"))
        + list(REPO.glob("optional-skills/**/SKILL.md"))
    )


def _rel(p: Path) -> str:
    return str(p.parent.relative_to(REPO))


def _params():
    return [pytest.param(p, id=_rel(p)) for p in _skill_paths()]


def _grandfathered(p: Path, rule: str) -> bool:
    return rule in GRANDFATHER.get(_rel(p), set())


def _frontmatter(p: Path):
    content = p.read_text(encoding="utf-8")
    assert content.startswith("---"), f"{_rel(p)}: SKILL.md must start with ---"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, f"{_rel(p)}: unclosed frontmatter"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    assert isinstance(fm, dict), f"{_rel(p)}: frontmatter must be a YAML mapping"
    return fm, content


ALL_SKILL_NAMES = None


def _all_names():
    global ALL_SKILL_NAMES
    if ALL_SKILL_NAMES is None:
        names = set()
        for p in _skill_paths():
            names.add(p.parent.name)
        ALL_SKILL_NAMES = names
    return ALL_SKILL_NAMES


def test_at_least_the_expected_population():
    # sanity: the globs actually find the trees (not a count snapshot)
    paths = _skill_paths()
    assert any("optional-skills" in str(p) for p in paths)
    assert any(str(p.parent).startswith(str(REPO / "skills")) for p in paths)


@pytest.mark.parametrize("p", _params())
def test_required_frontmatter_fields(p):
    fm, _ = _frontmatter(p)
    missing = [
        f
        for f in ("name", "description", "version", "author", "license", "platforms")
        if f not in fm
    ]
    if missing and not _grandfathered(p, "fields"):
        pytest.fail(f"{_rel(p)}: missing frontmatter fields: {missing}")
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    if not (hermes.get("tags") or fm.get("tags")) and not _grandfathered(p, "tags"):
        pytest.fail(f"{_rel(p)}: no tags (metadata.hermes.tags or top-level tags)")


@pytest.mark.parametrize("p", _params())
def test_name_matches_directory(p):
    fm, _ = _frontmatter(p)
    if fm.get("name") != p.parent.name and not _grandfathered(p, "name"):
        pytest.fail(
            f"{_rel(p)}: frontmatter name {fm.get('name')!r} != dir {p.parent.name!r}"
        )


@pytest.mark.parametrize("p", _params())
def test_description_hardline(p):
    fm, _ = _frontmatter(p)
    desc = str(fm.get("description") or "")
    if _grandfathered(p, "description"):
        return
    assert len(desc) <= 60, f"{_rel(p)}: description {len(desc)} chars (hardline 60)"
    assert desc.rstrip().endswith("."), f"{_rel(p)}: description must end with a period"
    m = MARKETING.search(desc)
    assert not m, f"{_rel(p)}: marketing word in description: {m.group(0)!r}"


@pytest.mark.parametrize("p", _params())
def test_related_skills_resolve(p):
    fm, _ = _frontmatter(p)
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    dangling = [
        rs for rs in (hermes.get("related_skills") or []) if rs not in _all_names()
    ]
    if dangling and not _grandfathered(p, "related"):
        pytest.fail(f"{_rel(p)}: dangling related_skills: {dangling}")


@pytest.mark.parametrize("p", _params())
def test_no_machine_local_paths(p):
    _, content = _frontmatter(p)
    m = MACHINE_LOCAL.search(content)
    if m and not _grandfathered(p, "paths"):
        pytest.fail(f"{_rel(p)}: machine-local path {m.group(0)!r}")


@pytest.mark.parametrize("p", _params())
def test_size_limit(p):
    _, content = _frontmatter(p)
    if len(content) > 100_000 and not _grandfathered(p, "size"):
        pytest.fail(
            f"{_rel(p)}: {len(content)} chars > 100k — split into references/"
        )


def test_grandfather_entries_still_needed():
    """A grandfather entry whose violation is fixed must be removed."""
    for rel in GRANDFATHER:
        assert (REPO / rel / "SKILL.md").exists(), f"stale grandfather entry: {rel}"


QR_PLACEHOLDER = "Quick reference patterns will be added as you use the skill."
SCRAPER_MARK = "automatically generated from official documentation"


@pytest.mark.parametrize("p", _params())
def test_no_scraper_quick_reference_placeholder(p):
    """Scraper leftover QR placeholder must not remain in any SKILL.md."""
    content = p.read_text(encoding="utf-8")
    if QR_PLACEHOLDER in content:
        pytest.fail(
            f"{_rel(p)}: leftover scraper placeholder {QR_PLACEHOLDER!r} — "
            "fill Quick Reference with **Pattern N:** entries"
        )


def _scraper_generated_with_index():
    for p in _skill_paths():
        content = p.read_text(encoding="utf-8")
        if SCRAPER_MARK not in content:
            continue
        index = p.parent / "references" / "index.md"
        if index.is_file():
            yield pytest.param(p, id=_rel(p))


@pytest.mark.parametrize("p", list(_scraper_generated_with_index()))
def test_scraper_reference_files_listed(p):
    """Scraper skills with references/index.md must list every references/*.md.

    Fail-open: handwritten skills and skills without index.md are skipped so a
    repo-wide "every references/*.md appears as a SKILL.md bullet" rule cannot
    fire the 77-FP sweep. Non-.md files are ignored; index.md is not required
    to list itself.
    """
    skill = p.read_text(encoding="utf-8")
    index_path = p.parent / "references" / "index.md"
    index = index_path.read_text(encoding="utf-8")
    missing_skill = []
    missing_index = []
    for ref in sorted((p.parent / "references").glob("*.md")):
        if ref.name == "index.md":
            continue
        if ref.name not in skill:
            missing_skill.append(ref.name)
        if ref.name not in index:
            missing_index.append(ref.name)
    problems = []
    if missing_skill:
        problems.append(f"unlisted in SKILL.md: {missing_skill}")
    if missing_index:
        problems.append(f"unlisted in references/index.md: {missing_index}")
    if problems:
        pytest.fail(f"{_rel(p)}: {' | '.join(problems)}")
