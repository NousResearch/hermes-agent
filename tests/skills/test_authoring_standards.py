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
SHELL_FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})\s*(bash|sh|shell|zsh|console)\b", re.I)
# A line-continuation backslash must be the last character on its line: a
# comment after it voids the continuation and the flags below run as a
# separate command when the snippet is pasted (#113448).
CONTINUATION_THEN_COMMENT = re.compile(r"\\\s+#")

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


def _shell_fence_lines(content: str):
    """Yield (lineno, line) for every line inside a fenced bash/sh block."""
    fence = None
    for i, line in enumerate(content.splitlines(), 1):
        if fence is None:
            m = SHELL_FENCE_OPEN.match(line)
            if m:
                fence = m.group(1)
            continue
        if line.strip().startswith(fence):
            fence = None
            continue
        yield i, line


@pytest.mark.parametrize("p", _params())
def test_shell_snippets_paste_safe(p):
    _, content = _frontmatter(p)
    bad = [
        f"{_rel(p)}/SKILL.md:{i}: {line.strip()}"
        for i, line in _shell_fence_lines(content)
        if CONTINUATION_THEN_COMMENT.search(line)
    ]
    assert not bad, (
        "comment after a line-continuation backslash breaks the pasted command; "
        "move the note above the whole command:\n" + "\n".join(bad)
    )


def test_grandfather_entries_still_needed():
    """A grandfather entry whose violation is fixed must be removed."""
    for rel in GRANDFATHER:
        assert (REPO / rel / "SKILL.md").exists(), f"stale grandfather entry: {rel}"
