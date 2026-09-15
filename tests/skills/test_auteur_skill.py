"""Tests for the auteur optional skill (ported from agiwhitelist/auteur, MIT).

Frontmatter shape, description length and related_skills resolution are
covered repo-wide by tests/skills/test_authoring_standards.py; this file only
holds the two invariants specific to the port.
"""
import re
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "auteur"
SKILL_MD = SKILL_DIR / "SKILL.md"


def test_mentioned_paths_exist_or_annotated():
    """Every references/scripts/templates path mentioned in SKILL.md exists on
    disk, or its line carries an 'upstream' / 'not vendored' annotation (the
    port deliberately drops upstream's README gallery and CLI assets)."""
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


def test_no_upstream_harness_residue():
    """Upstream ships as a plugin for another agent harness; its plugin-only
    frontmatter keys and harness name must not leak into the Hermes port."""
    text = SKILL_MD.read_text(encoding="utf-8").lower()
    for token in ("claude", "allowed-tools", "argument-hint"):
        assert token not in text, f"residual '{token}' in SKILL.md"
