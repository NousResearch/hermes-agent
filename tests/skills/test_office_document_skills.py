"""Invariant tests for the bundled office/document skills.

Covers skills/productivity/{docx,xlsx,pdf,powerpoint} — the office
document creation/editing suite. Tests assert contracts (frontmatter
shape, referenced scripts exist, cross-links resolve), not snapshots
of skill content.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
SKILLS = REPO / "skills"
OPTIONAL_SKILLS = REPO / "optional-skills"

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
def test_referenced_scripts_exist(name):
    """Every scripts/... path mentioned in SKILL.md must exist on disk."""
    skill_dir = _skill_dir(name)
    body = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"scripts/[\w./-]+\.py", body))
    assert refs, f"{name}: SKILL.md references no helper scripts"
    for ref in refs:
        assert (skill_dir / ref).exists(), f"{name}: SKILL.md references missing {ref}"


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_related_skills_resolve(name):
    """related_skills entries must name skills that exist in skills/ or optional-skills/."""
    fm = _frontmatter(_skill_dir(name) / "SKILL.md")
    related = fm.get("metadata", {}).get("hermes", {}).get("related_skills", [])
    assert related, f"{name}: office skills must cross-link related_skills"
    all_skill_names = {
        p.parent.name
        for root in (SKILLS, OPTIONAL_SKILLS)
        for p in root.rglob("SKILL.md")
    }
    for rel in related:
        assert rel in all_skill_names, f"{name}: related skill {rel!r} does not exist"


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_license_file_present(name):
    """Adapted Anthropic skills must carry their LICENSE.txt."""
    fm = _frontmatter(_skill_dir(name) / "SKILL.md")
    if "LICENSE.txt" in str(fm.get("license", "")):
        assert (_skill_dir(name) / "LICENSE.txt").exists(), (
            f"{name}: license points to LICENSE.txt but the file is missing"
        )


@pytest.mark.parametrize("name", OFFICE_SKILLS)
def test_scripts_compile(name):
    """All shipped helper scripts must be valid Python."""
    import py_compile

    skill_dir = _skill_dir(name)
    scripts = list((skill_dir / "scripts").rglob("*.py")) if (skill_dir / "scripts").exists() else []
    assert scripts, f"{name}: expected helper scripts under scripts/"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)


def test_docx_validator_schema_paths_exist():
    """base.py maps XML parts to XSD files — every mapped schema must ship."""
    for skill in ("docx", "powerpoint"):
        base = _skill_dir(skill) / "scripts" / "office" / "validators" / "base.py"
        schemas = _skill_dir(skill) / "scripts" / "office" / "schemas"
        text = base.read_text(encoding="utf-8")
        refs = set(re.findall(r'"((?:ecma|ISO|mce|microsoft)[\w./-]+\.xsd)"', text))
        assert refs, f"{skill}: no schema references found in validators/base.py"
        for ref in refs:
            assert (schemas / ref).exists(), f"{skill}: validator references missing schema {ref}"


def test_pdf_reference_docs_exist():
    """pdf SKILL.md links forms.md and reference.md — both must ship."""
    pdf_dir = _skill_dir("pdf")
    body = (pdf_dir / "SKILL.md").read_text(encoding="utf-8")
    for doc in ("forms.md", "reference.md"):
        assert doc in body
        assert (pdf_dir / doc).exists(), f"pdf: missing linked doc {doc}"


def test_docs_pages_generated():
    """Each bundled office skill has a generated docs-site page."""
    docs_dir = REPO / "website" / "docs" / "user-guide" / "skills" / "bundled" / "productivity"
    for name in OFFICE_SKILLS:
        assert (docs_dir / f"productivity-{name}.md").exists(), (
            f"missing generated docs page for {name}; run website/scripts/generate-skill-docs.py"
        )
