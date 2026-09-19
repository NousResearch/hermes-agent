"""Tests for the NIS2 GRC skill (skills/grc/nis2-grc).

The skill is documentation-only (no executable scripts), so these tests pin the
contracts that matter for a content skill:

1. It passes the repo's real skill validator (tools/skill_manager_tool).
2. It passes the security skill scanner (tools/skills_guard) with no findings.
3. Its content invariants hold: the 10 Art. 21(2) measures, the 5-phase GRC
   workflow, and the NIS2 incident-reporting windows (24h / 72h / 1 month).

Run with: pytest tests/skills/test_nis2_grc_skill.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "grc" / "nis2-grc"
SKILL_MD = SKILL_DIR / "SKILL.md"

# Load the repo's real validators by path (they are not package-importable by name).
def _load(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load("skill_manager_tool_grc_test", REPO_ROOT / "tools" / "skill_manager_tool.py")
GUARD = _load("skills_guard_grc_test", REPO_ROOT / "tools" / "skills_guard.py")


def _read_skill() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_skill_exists():
    assert SKILL_MD.exists()
    assert SKILL_DIR.is_dir()


def test_frontmatter_validation_passes():
    VALIDATOR._validate_frontmatter(_read_skill())


def test_content_size_within_limit():
    VALIDATOR._validate_content_size(_read_skill())


def test_name_validation_passes():
    VALIDATOR._validate_name("nis2-grc")


def test_category_validation_passes():
    VALIDATOR._validate_category("grc")


@pytest.mark.parametrize(
    "rel_path",
    [
        "references/scoping.md",
        "references/measures.md",
        "references/gap-assessment.md",
        "references/incident-reporting.md",
        "templates/gap-assessment.md",
        "templates/incident-report.md",
        "templates/scoping-questionnaire.md",
    ],
)
def test_supporting_files_exist_and_validate_path(rel_path: str):
    assert (SKILL_DIR / rel_path).exists()
    # Should not raise for any allowed references/ / templates/ subpath.
    VALIDATOR._validate_file_path(rel_path)


def test_skill_scans_safe():
    """A content skill must not trip the security scanner."""
    result = GUARD.scan_skill(SKILL_DIR, source="official")
    assert result.verdict == "safe"
    assert result.findings == []


def test_lists_all_ten_minimum_measures():
    """Art. 21(2) enumerates 10 minimum cybersecurity measures (a)-(j)."""
    body = _read_skill()
    for letter in "abcdefghij":
        assert f"({letter})" in body, f"measure ({letter}) missing from SKILL.md"


def test_defines_five_phase_grc_workflow():
    for phase in ["Fase 1", "Fase 2", "Fase 3", "Fase 4", "Fase 5"]:
        assert phase in _read_skill(), f"{phase} missing"


def test_incident_reporting_windows_present():
    """NIS2 mandatory reporting windows: early warning 24h, intermediate 72h, final 1 mês.

    The skill is authored in Portuguese, so the final-window marker is the
    Portuguese "1 mês" (not "1 month").
    """
    body = _read_skill()
    for marker in ["24h", "72h", "1 mês"]:
        assert marker in body, f"incident window marker {marker!r} missing"
