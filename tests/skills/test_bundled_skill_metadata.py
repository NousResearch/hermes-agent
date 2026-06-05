"""Validate metadata and linked-file references for every bundled Hermes skill.

Bundled skills live under ``skills/`` and ``optional-skills/``. Each ships a
``SKILL.md`` whose YAML frontmatter is consumed by the skill loader. Two classes
of mistake silently break a skill at load (or surprise the user mid-task) and are
easy to introduce by hand-editing markdown:

  1. **Frontmatter validity.** A skill with malformed/unclosed frontmatter, a
     non-mapping header, a missing ``name``/``description``, an over-long
     description, or an illegally-shaped name. We reuse the *exact* validators the
     live skill manager runs when an agent creates or edits a skill
     (``tools/skill_manager_tool.py``) so this test can never drift from the
     runtime contract — if the rules tighten there, the whole bundled tree is
     re-checked here for free.

  2. **Dangling linked files.** A markdown link in SKILL.md that points into the
     skill's own subdirectories (``scripts/``, ``references/``, ``templates/``,
     ``assets/``, ``examples/``) but no such file ships. These are the references
     the agent is told to open; a typo means a dead end at runtime.

Both checks are parametrized per skill so a failure names the offending skill
directory. Dependencies are stdlib + PyYAML (already a hard dependency, and the
same parser the runtime validator uses).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# conftest.py puts the project root on sys.path, so the runtime validators are
# importable. Reusing them is deliberate: the bundled tree must satisfy the same
# contract the live editor enforces, with zero duplicated rules to drift.
from tools.skill_manager_tool import _validate_frontmatter, _validate_name

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOTS = (REPO_ROOT / "skills", REPO_ROOT / "optional-skills")

# Subdirectories a skill ships alongside SKILL.md. A markdown link whose target
# begins with one of these (and resolves under the skill dir) is an in-skill file
# reference we can verify. Anything else — URLs, anchors, absolute paths, upstream
# repo paths, generated-output examples — is intentionally left alone to keep the
# check free of false positives.
LINKED_SUBDIRS = (
    "scripts/",
    "references/",
    "reference/",
    "templates/",
    "template/",
    "assets/",
    "examples/",
    "example/",
)

_FENCE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _discover_skills() -> list[Path]:
    """Every bundled SKILL.md, sorted for deterministic parametrization order."""
    found: list[Path] = []
    for root in SKILL_ROOTS:
        if root.is_dir():
            found.extend(root.rglob("SKILL.md"))
    return sorted(found)


SKILL_FILES = _discover_skills()
# Stable, readable test ids: "skills/research/arxiv", "optional-skills/finance/...".
SKILL_IDS = [str(p.parent.relative_to(REPO_ROOT)) for p in SKILL_FILES]


def _frontmatter_yaml(content: str) -> str:
    """Slice the raw YAML between the opening and closing ``---`` fences.

    Mirrors the slicing in ``_validate_frontmatter`` so the parsed mapping we
    inspect is exactly the bytes the runtime validator parsed.
    """
    end = re.search(r"\n---\s*\n", content[3:])
    assert end, "frontmatter must be closed before YAML can be sliced"
    return content[3 : end.start() + 3]


def test_skills_are_discovered() -> None:
    """Guard against a silent zero-collection (e.g. a moved skills tree)."""
    assert SKILL_FILES, "no bundled SKILL.md files discovered under skills/ or optional-skills/"


@pytest.mark.parametrize("skill_md", SKILL_FILES, ids=SKILL_IDS)
def test_frontmatter_valid(skill_md: Path) -> None:
    """SKILL.md satisfies the same frontmatter contract the live editor enforces."""
    content = skill_md.read_text(encoding="utf-8")

    err = _validate_frontmatter(content)
    assert err is None, f"{skill_md.relative_to(REPO_ROOT)}: {err}"

    # _validate_frontmatter guarantees a parseable mapping with name+description;
    # additionally hold `name` to the runtime naming rules (regex + length), which
    # _validate_frontmatter checks on create but not on the stored content.
    import yaml  # local import: stdlib-style lazy load, parser already a dep

    parsed = yaml.safe_load(_frontmatter_yaml(content))
    name_err = _validate_name(str(parsed["name"]))
    assert name_err is None, f"{skill_md.relative_to(REPO_ROOT)}: {name_err}"

    assert str(parsed["description"]).strip(), (
        f"{skill_md.relative_to(REPO_ROOT)}: description must not be blank"
    )


@pytest.mark.parametrize("skill_md", SKILL_FILES, ids=SKILL_IDS)
def test_linked_files_exist(skill_md: Path) -> None:
    """Markdown links into the skill's own subdirectories resolve to shipped files."""
    skill_dir = skill_md.parent
    # Strip fenced code blocks first: example/snippet links inside ``` fences are
    # illustrative, not real references to shipped files.
    prose = _FENCE_RE.sub("", skill_md.read_text(encoding="utf-8"))

    missing: list[str] = []
    for match in _MD_LINK_RE.finditer(prose):
        target = match.group(1).strip()
        if not target:
            continue
        # Drop an optional link title: [text](path "title").
        target = target.split()[0]
        # Skip anything that isn't a plain in-skill relative path.
        if re.match(r"^[a-z][a-z0-9+.-]*://", target):  # URL scheme
            continue
        if target.startswith(("#", "mailto:", "/")):  # anchor / mail / absolute
            continue
        if any(ch in target for ch in "<>{}*"):  # placeholder/glob template
            continue
        target = target.split("#")[0].split("?")[0]  # drop anchor/query
        if not target.startswith(LINKED_SUBDIRS):
            continue
        if not (skill_dir / target).exists():
            missing.append(target)

    assert not missing, (
        f"{skill_md.relative_to(REPO_ROOT)}: linked files referenced in SKILL.md "
        f"do not exist: {sorted(set(missing))}"
    )
