"""Tests for optional-skills/hermes/memory-hygiene skill."""

import re
from pathlib import Path

import pytest
import hermes_yaml as yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "hermes" / "memory-hygiene"
SKILL_FILE = SKILL_DIR / "SKILL.md"


def _frontmatter():
    content = SKILL_FILE.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must start with ---"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "unclosed frontmatter"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    assert isinstance(fm, dict), "frontmatter must be a YAML mapping"
    return fm, content


def test_required_frontmatter_fields():
    fm, _ = _frontmatter()
    required = ("name", "description", "version", "author", "license", "platforms")
    missing = [f for f in required if f not in fm]
    assert not missing, f"missing frontmatter fields: {missing}"
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    assert hermes.get("tags"), "no tags in metadata.hermes.tags"


def test_name_matches_directory():
    fm, _ = _frontmatter()
    assert fm.get("name") == SKILL_DIR.name


def test_description_hardline():
    fm, _ = _frontmatter()
    desc = str(fm.get("description") or "")
    assert len(desc) <= 60, f"description {len(desc)} chars (hardline 60)"
    assert desc.rstrip().endswith("."), "description must end with a period"
    marketing = re.compile(r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b", re.I)
    assert not marketing.search(desc), "marketing word in description"


def test_related_skills_resolve():
    fm, _ = _frontmatter()
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    # Collect all in-repo skill names
    all_names = set()
    for p in REPO.glob("skills/**/SKILL.md"):
        all_names.add(p.parent.name)
    for p in REPO.glob("optional-skills/**/SKILL.md"):
        all_names.add(p.parent.name)
    dangling = [rs for rs in (hermes.get("related_skills") or []) if rs not in all_names]
    assert not dangling, f"dangling related_skills: {dangling}"


def test_no_machine_local_paths():
    _, content = _frontmatter()
    machine_local = re.compile(r"/home/(?!runner\b)[a-z0-9_-]+/|[A-Z]:\\+Users\\+(?!<)")
    m = machine_local.search(content)
    assert not m, f"machine-local path {m.group(0)!r}"


def test_size_limit():
    _, content = _frontmatter()
    assert len(content) <= 100_000, f"{len(content)} chars > 100k"


def test_frontmatter_yaml_valid():
    """Verify frontmatter parses as valid YAML with expected types."""
    fm, _ = _frontmatter()
    assert isinstance(fm["name"], str)
    assert isinstance(fm["description"], str)
    assert isinstance(fm["version"], str)
    assert isinstance(fm["author"], str)
    assert isinstance(fm["license"], str)
    assert isinstance(fm["platforms"], list)
    assert all(isinstance(p, str) for p in fm["platforms"])
    assert "metadata" in fm
    assert "hermes" in fm["metadata"]
    assert "tags" in fm["metadata"]["hermes"]
    assert "related_skills" in fm["metadata"]["hermes"]
    assert isinstance(fm["metadata"]["hermes"]["tags"], list)
    assert isinstance(fm["metadata"]["hermes"]["related_skills"], list)


def test_platforms_audited():
    """Platforms should match actual skill usage (Hermes tools only = all three)."""
    fm, _ = _frontmatter()
    # This skill uses Hermes tools and cross-platform CLIs only
    assert fm["platforms"] == ["linux", "macos", "windows"]


def test_author_credits_human_first():
    fm, _ = _frontmatter()
    author = fm["author"]
    # Must contain a human name + github handle before ", Hermes Agent"
    assert ", Hermes Agent" in author
    human_part = author.split(", Hermes Agent")[0].strip()
    assert human_part, "human name missing before 'Hermes Agent'"


def test_version_semver():
    fm, _ = _frontmatter()
    # Basic semver check: major.minor.patch
    import re as _re
    assert _re.match(r"^\d+\.\d+\.\d+", fm["version"]), f"version {fm['version']!r} not semver"


def test_license_mit():
    fm, _ = _frontmatter()
    assert fm["license"] == "MIT"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))