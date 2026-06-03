"""Tests for scripts/code-scan/domain_surfaces.py.

Strict TDD: tests written first (or extended from fixture RED).
Covers inventory-style detection only — no semantic claims.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/code-scan is on sys.path for sibling imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts" / "code-scan"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from domain_surfaces import (
    scan_domain_surfaces,
    build_domain_surfaces_summary,
    DOMAIN_SURFACE_PATTERNS,
)

FIXTURES = _PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "domain_surfaces"


def _load_scan(name: str) -> dict:
    """Load scan.json for a domain_surface fixture."""
    return json.loads((FIXTURES / name / "scan.json").read_text(encoding="utf-8"))


def _load_expected(name: str) -> dict:
    """Load expected domain-surfaces.json for comparison."""
    return json.loads((FIXTURES / name / "expected.json").read_text(encoding="utf-8"))


def _normalize_surfaces(s: dict) -> dict:
    """Strip volatile fields for deterministic comparison (scanned_at etc)."""
    out = {
        "surfaces": sorted(s.get("surfaces", []), key=lambda x: (x.get("surface", ""), x.get("path", ""))),
        "summary": s.get("summary", {}),
        "claim_type": s.get("claim_type"),
        "semantic_status": s.get("semantic_status"),
    }
    return out


# ── Schema & contract tests ─────────────────────────────────────────────


class TestOutputSchema:
    """Domain surfaces output is a pure inventory with required labels."""

    def test_output_has_required_top_level(self):
        scan = _load_scan("full_project")
        result = scan_domain_surfaces(scan)
        assert "surfaces" in result
        assert "summary" in result
        assert result.get("claim_type") == "deterministic_inventory"
        assert result.get("semantic_status") == "not_validated"

    def test_each_surface_has_required_fields(self):
        scan = _load_scan("full_project")
        result = scan_domain_surfaces(scan)
        for surf in result.get("surfaces", []):
            for key in ("surface", "path", "claim_type", "semantic_status"):
                assert key in surf
            assert surf["claim_type"] == "deterministic_inventory"
            assert surf["semantic_status"] == "not_validated"

    def test_summary_shape(self):
        scan = _load_scan("full_project")
        result = scan_domain_surfaces(scan)
        summary = result.get("summary", {})
        assert "total_surfaces" in summary
        assert "surface_types" in summary
        assert isinstance(summary["surface_types"], dict)


# ── Fixture-driven regression tests (RED → GREEN) ───────────────────────


class TestEmptyProject:
    """Empty / minimal project has zero domain surfaces."""

    def test_empty_has_zero_surfaces(self):
        scan = _load_scan("empty_project")
        result = scan_domain_surfaces(scan)
        expected = _load_expected("empty_project")
        assert _normalize_surfaces(result) == _normalize_surfaces(expected)
        assert result["summary"]["total_surfaces"] == 0


class TestFullProject:
    """Full project exercises all surface categories from fixture."""

    def test_full_detects_all_expected_surfaces(self):
        scan = _load_scan("full_project")
        result = scan_domain_surfaces(scan)
        expected = _load_expected("full_project")
        norm_result = _normalize_surfaces(result)
        norm_expected = _normalize_surfaces(expected)
        assert norm_result == norm_expected
        # Sanity: supabase surfaces present
        surfaces = [s["surface"] for s in result["surfaces"]]
        assert "supabase_migration" in surfaces
        assert "supabase_edge_function" in surfaces
        # vite + sw + ci + deployment
        assert "vite_config" in surfaces
        assert "service_worker" in surfaces
        assert "ci_workflow" in surfaces
        assert "vercel_config" in surfaces

    def test_full_summary_counts_match(self):
        scan = _load_scan("full_project")
        result = scan_domain_surfaces(scan)
        assert result["summary"]["total_surfaces"] == 7
        st = result["summary"]["surface_types"]
        assert st["supabase_migration"] == 2
        assert st.get("supabase_edge_function") == 1


class TestPackageScriptsProject:
    """package.json with scripts section is treated as package_scripts surface."""

    def test_package_scripts_detected_when_scripts_present(self):
        scan = _load_scan("package_scripts_project")
        result = scan_domain_surfaces(scan)
        expected = _load_expected("package_scripts_project")
        assert _normalize_surfaces(result) == _normalize_surfaces(expected)
        surfaces = [s["surface"] for s in result["surfaces"]]
        assert "package_scripts" in surfaces


class TestPWAProject:
    """PWA indicators: manifest + service worker."""

    def test_pwa_manifest_and_service_worker(self):
        scan = _load_scan("pwa_project")
        result = scan_domain_surfaces(scan)
        expected = _load_expected("pwa_project")
        assert _normalize_surfaces(result) == _normalize_surfaces(expected)
        surfaces = [s["surface"] for s in result["surfaces"]]
        assert "pwa_manifest" in surfaces
        assert "service_worker" in surfaces
        assert result["summary"]["total_surfaces"] == 3


# ── Pattern / determinism tests ─────────────────────────────────────────


class TestSurfacePatterns:
    """Patterns are deterministic and don't rely on content parsing."""

    def test_known_patterns_covered(self):
        # Ensure we declare the expected pattern families
        patterns = DOMAIN_SURFACE_PATTERNS
        assert any("supabase/migrations" in p for p in patterns)
        assert any("supabase/functions" in p for p in patterns)
        assert any("vite.config" in p for p in patterns)
        assert any("sw.js" in p or "service-worker" in p for p in patterns)

    def test_deterministic_order(self):
        """Same scan input → same ordered surfaces output."""
        scan = _load_scan("full_project")
        r1 = scan_domain_surfaces(scan)
        r2 = scan_domain_surfaces(scan)
        assert r1 == r2
        # surfaces list should be sorted stably
        paths1 = [s["path"] for s in r1["surfaces"]]
        paths2 = [s["path"] for s in r2["surfaces"]]
        assert paths1 == paths2


# ── CLI smoke (optional but good) ───────────────────────────────────────


def test_cli_help_runs():
    """domain_surfaces.py should support --help without crashing."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS_DIR / "domain_surfaces.py"), "--help"],
        capture_output=True,
        text=True,
    )
    # May be 0 or 2 depending on argparse; main point: no crash / traceback
    assert result.returncode in (0, 2)
    assert "domain" in (result.stdout + result.stderr).lower() or result.returncode == 2
