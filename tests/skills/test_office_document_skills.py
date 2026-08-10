"""Invariant tests for the bundled office/document skills.

Covers skills/productivity/{docx,xlsx,pdf,powerpoint} — the clean-room
MIT office document suite. Tests assert contracts (frontmatter shape,
referenced scripts exist, script CLI conventions, UTF-8-explicit I/O),
not snapshots of skill content.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
SKILLS = REPO / "skills"

OFFICE_SKILLS = ["docx", "xlsx", "pdf", "powerpoint"]


def _skill_dir(name: str) -> Path:
    return SKILLS / "productivity" / name


def _frontmatter(skill_md: Path) -> dict:
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{skill_md} has no YAML frontmatter"
    return yaml.safe_load(match.group(1))


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_skill_exists_with_frontmatter(name):
    skill_md = _skill_dir(name) / "SKILL.md"
    assert skill_md.exists(), f"missing {skill_md}"
    fm = _frontmatter(skill_md)
    assert fm["name"] == name
    assert fm["description"].strip()
    assert len(fm["description"]) <= 60, (
        f"{name}: description is {len(fm['description'])} chars (max 60)"
    )
    assert fm["description"].rstrip('"').endswith(".")
    platforms = fm.get("platforms")
    assert platforms, f"{name}: missing platforms gating"
    assert set(platforms) <= {"linux", "macos", "windows"}


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_mit_licensed_clean_room(name):
    """The office suite is the clean-room rewrite: MIT, no Anthropic
    license text, no proprietary license markers anywhere in the dir."""
    skill_dir = _skill_dir(name)
    fm = _frontmatter(skill_dir / "SKILL.md")
    assert str(fm.get("license", "")).strip() == "MIT", (
        f"{name}: license must be MIT, got {fm.get('license')!r}"
    )
    assert not (skill_dir / "LICENSE.txt").exists(), (
        f"{name}: legacy LICENSE.txt present — clean-room dirs ship LICENSE (MIT)"
    )
    license_file = skill_dir / "LICENSE"
    assert license_file.exists(), f"{name}: missing MIT LICENSE file"
    text = license_file.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Anthropic" not in text
    for path in skill_dir.rglob("*"):
        if path.is_file() and path.suffix in (".md", ".py"):
            content = path.read_text(encoding="utf-8", errors="replace")
            assert "Anthropic" not in content, (
                f"{name}: {path.relative_to(skill_dir)} references Anthropic — "
                "clean-room provenance violation"
            )


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_referenced_scripts_exist(name):
    """Every scripts/... path mentioned in SKILL.md must exist on disk."""
    skill_dir = _skill_dir(name)
    body = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"scripts/[\w./-]+\.py", body))
    assert refs, f"{name}: SKILL.md references no helper scripts"
    for ref in refs:
        assert (skill_dir / ref).exists(), f"{name}: SKILL.md references missing {ref}"


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_all_shipped_scripts_are_documented(name):
    """Every shipped scripts/*.py is mentioned in SKILL.md (no dead cargo).
    Shared/internal modules (underscore-prefixed or *_common.py) are exempt."""
    skill_dir = _skill_dir(name)
    body = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    for script in (skill_dir / "scripts").glob("*.py"):
        if script.name.startswith("_") or script.stem.endswith("_common"):
            continue
        assert script.name in body, (
            f"{name}: scripts/{script.name} is shipped but never mentioned in SKILL.md"
        )


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_scripts_use_explicit_utf8_text_io(name):
    """No locale-default text-mode I/O in helper scripts: every text-mode
    open() must pass encoding=. Binary-mode opens are exempt. This is the
    class of bug that mojibake'd form fills on cp1251/GBK/cp932 hosts."""
    skill_dir = _skill_dir(name)
    offenders = []
    for script in (skill_dir / "scripts").rglob("*.py"):
        content = script.read_text(encoding="utf-8")
        for m in re.finditer(r"(?<![\w.])open\(([^)]*)\)", content):
            args = m.group(1)
            if re.search(r"['\"][rwaxt+]*b[rwaxt+]*['\"]", args):
                continue  # binary mode
            if "encoding" not in args:
                line = content[: m.start()].count("\n") + 1
                offenders.append(f"{script.relative_to(skill_dir)}:{line}: open({args})")
    assert not offenders, (
        f"{name}: text-mode open() without explicit encoding:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_scripts_are_argparse_clis(name):
    """Helper scripts are argparse CLIs: importable arg parsing + a main
    guard, so `python scripts/x.py --help` works everywhere."""
    skill_dir = _skill_dir(name)
    for script in (skill_dir / "scripts").glob("*.py"):
        if script.name.startswith("_") or script.stem.endswith("_common"):
            continue
        content = script.read_text(encoding="utf-8")
        assert "argparse" in content, f"{name}: scripts/{script.name} is not an argparse CLI"
        assert '__name__' in content, f"{name}: scripts/{script.name} lacks a __main__ guard"


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_skill_has_tests(name):
    """Each office skill ships its own e2e pytest suite."""
    tests_dir = _skill_dir(name) / "tests"
    assert tests_dir.is_dir(), f"{name}: missing tests/ directory"
    assert list(tests_dir.glob("test_*.py")), f"{name}: no test files in tests/"


def test_docs_pages_generated():
    """Each bundled office skill has a generated docs-site page."""
    docs_dir = REPO / "website" / "docs" / "user-guide" / "skills" / "bundled" / "productivity"
    for name in OFFICE_SKILLS:
        assert (docs_dir / f"productivity-{name}.md").exists(), (
            f"missing generated docs page for {name}; run website/scripts/generate-skill-docs.py"
        )
