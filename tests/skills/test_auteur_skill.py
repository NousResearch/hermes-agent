"""Tests for the auteur optional skill (ported from agiwhitelist/auteur, MIT)."""
import re
from pathlib import Path

import yaml

def _find_skill_dir() -> Path:
    base = Path(__file__).resolve().parents[2]
    candidate = base / "optional-skills" / "creative" / "auteur"
    if candidate.is_dir():
        return candidate
    # fallback: glob from repo root for any auteur skill dir
    for p in base.glob("**/optional-skills/**/auteur"):
        if (p / "SKILL.md").exists():
            return p
    raise AssertionError("auteur skill directory not found relative to test file")

SKILL_DIR = _find_skill_dir()
SKILL_MD = SKILL_DIR / "SKILL.md"


def _frontmatter() -> dict:
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    fm = text.split("---\n", 2)[1]
    data = yaml.safe_load(fm)
    assert isinstance(data, dict)
    return data


def test_frontmatter_parses():
    fm = _frontmatter()
    assert fm["name"] == "auteur"


def test_description_length():
    fm = _frontmatter()
    desc = fm["description"]
    assert isinstance(desc, str) and desc.strip()
    assert len(desc) <= 60, f"description is {len(desc)} chars, must be <= 60"


def test_required_fields():
    fm = _frontmatter()
    assert fm.get("license") == "MIT"
    assert fm.get("author"), "author missing"
    platforms = fm.get("platforms")
    assert isinstance(platforms, list) and platforms, "platforms missing"
    assert set(platforms) <= {"linux", "macos", "windows"}
    assert fm.get("version")


def test_mentioned_paths_exist_or_annotated():
    """Every references/scripts/templates path mentioned in SKILL.md exists on
    disk, or its line carries an 'upstream' / 'not vendored' annotation."""
    pattern = re.compile(r"(references|scripts|templates)/[A-Za-z0-9._-]+")
    missing = []
    for line in SKILL_MD.read_text(encoding="utf-8").splitlines():
        for m in pattern.finditer(line):
            rel = m.group(0).rstrip(".")
            if (SKILL_DIR / rel).exists():
                continue
            low = line.lower()
            if "upstream" in low or "not vendored" in low:
                continue
            missing.append(rel)
    assert not missing, f"paths mentioned but absent and unannotated: {missing}"


def test_no_claude_residue():
    text = SKILL_MD.read_text(encoding="utf-8").lower()
    for token in ("claude", "allowed-tools", "argument-hint"):
        assert token not in text, f"residual '{token}' in SKILL.md"


def test_related_skills_resolve():
    fm = _frontmatter()
    related = (fm.get("metadata") or {}).get("hermes", {}).get("related_skills", [])
    if not related:
        return  # empty list is allowed
    staging_root = Path(__file__).resolve().parents[2]
    repo_root = Path.home() / ".hermes" / "hermes-agent"
    roots = [
        staging_root / "skills",
        staging_root / "optional-skills",
        repo_root / "skills",
        repo_root / "optional-skills",
    ]
    for name in related:
        found = any(r.is_dir() and list(r.glob(f"**/{name}")) for r in roots)
        assert found, f"related skill '{name}' not found in skills/optional-skills trees"


def test_vendored_tree_shape():
    assert (SKILL_DIR / "LICENSE").exists()
    assert len(list((SKILL_DIR / "references").glob("*.md"))) == 11
    assert len(list((SKILL_DIR / "scripts").glob("*.mjs"))) == 8
    assert len(list((SKILL_DIR / "templates").iterdir())) == 6
