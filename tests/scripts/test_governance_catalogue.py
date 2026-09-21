"""Tests for scripts/governance_catalogue.py — bounded design gap.

P05 Batch 1: proves the governance catalogue audit runs and confirms the
bounded design gap: no owner or tier field exists in the current skill
SKILL.md frontmatter, so a Tier 1/2/3 governance framework CANNOT be
safely derived.  The test asserts the gap is real (not invented code)
and that the audit function extracts the existing metadata faithfully.

Uses a fixture skills tree — never touches live ``~/.hermes``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "governance_catalogue_under_test",
        str(SCRIPTS / "governance_catalogue.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_skill(skills_root: Path, name: str, frontmatter: str = "") -> Path:
    d = skills_root / name
    d.mkdir(parents=True, exist_ok=True)
    content = "---\n"
    content += frontmatter
    content += "\n---\n\n# " + name + "\n"
    (d / "SKILL.md").write_text(content)
    return d


@pytest.fixture
def fake_skills(tmp_path):
    s = tmp_path / "skills"
    s.mkdir()
    return s


class TestAuditSkillMetadata:
    def test_audit_empty_dir(self, fake_skills):
        mod = _load_script()
        result = mod.audit_skill_metadata(fake_skills)
        assert result == []

    def test_audit_one_skill(self, fake_skills):
        _make_skill(
            fake_skills,
            "test-skill",
            "name: test-skill\nadoption_status: permanent",
        )
        mod = _load_script()
        result = mod.audit_skill_metadata(fake_skills)
        assert len(result) == 1
        assert result[0]["name"] == "test-skill"
        assert result[0]["adoption_status"] == "permanent"
        assert result[0]["has_owner"] is False
        assert result[0]["has_tier"] is False

    def test_audit_multiple_skills(self, fake_skills):
        _make_skill(fake_skills, "skill-a", "name: skill-a")
        _make_skill(fake_skills, "skill-b", "name: skill-b\nadoption_status: permanent")
        _make_skill(fake_skills, "skill-c", "name: skill-c")
        mod = _load_script()
        result = mod.audit_skill_metadata(fake_skills)
        assert len(result) == 3

    def test_audit_nonexistent_dir(self, tmp_path):
        mod = _load_script()
        result = mod.audit_skill_metadata(tmp_path / "nonexistent")
        assert result == []

    def test_audit_skips_dirs_without_skill_md(self, fake_skills):
        _make_skill(fake_skills, "real-skill", "name: real-skill")
        (fake_skills / "no-skill-md").mkdir()
        mod = _load_script()
        result = mod.audit_skill_metadata(fake_skills)
        assert len(result) == 1
        assert result[0]["name"] == "real-skill"


class TestBoundedDesignGap:
    """The gap report must confirm no owner/tier exists."""

    def test_gap_report_shows_no_owner(self, fake_skills):
        _make_skill(fake_skills, "s1", "name: s1\nadoption_status: permanent")
        mod = _load_script()
        report = mod.catalogue_gap_report(fake_skills)
        assert report["any_owner"] is False
        assert report["any_tier"] is False

    def test_gap_report_shows_gap_string(self, fake_skills):
        _make_skill(fake_skills, "s1", "name: s1")
        mod = _load_script()
        report = mod.catalogue_gap_report(fake_skills)
        assert "design gap" in report["gap"].lower() or "cannot build" in report["gap"].lower()

    def test_gap_report_tiers_undefined(self, fake_skills):
        mod = _load_script()
        report = mod.catalogue_gap_report(fake_skills)
        assert report["tiers_defined"] is False

    def test_gap_report_adoption_statuses(self, fake_skills):
        _make_skill(fake_skills, "s1", "name: s1\nadoption_status: permanent")
        _make_skill(fake_skills, "s2", "name: s2")
        mod = _load_script()
        report = mod.catalogue_gap_report(fake_skills)
        assert "permanent" in report["adoption_statuses"]

    def test_governance_tiers_empty(self):
        """GOVERNANCE_TIERS must be empty (no schema decision yet)."""
        mod = _load_script()
        for tier, members in mod.GOVERNANCE_TIERS.items():
            assert members == set(), f"Tier {tier} should be empty"

    def test_design_gap_note_present(self):
        mod = _load_script()
        assert "cannot build" in mod.DESIGN_GAP_NOTE.lower() or "design gap" in mod.DESIGN_GAP_NOTE.lower()


class TestRealSkillsAudit:
    """Audit the real repo skills tree to confirm the gap is factual."""

    def test_real_skills_have_no_owner(self):
        mod = _load_script()
        skills = REPO / "skills"
        if not skills.is_dir():
            pytest.skip("no skills/ dir in repo")
        report = mod.catalogue_gap_report(skills)
        assert report["any_owner"] is False, (
            "Found owner field — schema may have changed; revisit the gap"
        )
        assert report["any_tier"] is False, (
            "Found tier field — schema may have changed; revisit the gap"
        )
        assert report["audited_skills"] > 0
