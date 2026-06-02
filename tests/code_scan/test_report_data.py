"""Tests for scripts/code-scan/report_data.py.

Covers: scan-only input, full-artifact input, missing optional artifacts,
truncation warnings, deterministic reading-plan candidates, malformed JSON,
schema-version handling, and CLI behavior.

Strict TDD: these tests MUST fail before report_data.py is implemented.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure scripts/code-scan is importable
_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "code-scan"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import report_data

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "report_data"


def _fixture(name: str) -> str:
    return str(_FIXTURE_DIR / name)


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text())


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run report_data.py as a subprocess and return the result."""
    return subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "report_data.py")] + args,
        capture_output=True,
        text=True,
    )


# ── Core unit tests ──────────────────────────────────────────────────────


class TestSchemaVersion:
    """Top-level schema_version must be present and correct."""

    def test_schema_version_present(self):
        scan = _load_fixture("scan.json")
        result = report_data.build_report_data(scan=scan)
        assert result["schema_version"] == "1.0.0"

    def test_schema_version_is_string(self):
        scan = _load_fixture("scan.json")
        result = report_data.build_report_data(scan=scan)
        assert isinstance(result["schema_version"], str)


class TestSourcesTracking:
    """sources dict must track which artifacts were loaded or missing."""

    def test_sources_with_scan_only(self):
        scan = _load_fixture("scan.json")
        result = report_data.build_report_data(scan=scan)
        src = result["sources"]
        assert src["scan"] == "loaded"
        assert src["classified_imports"] == "not_provided"
        assert src["entrypoints"] == "not_provided"
        assert src["graph"] == "not_provided"
        assert src["orphan_triage"] == "not_provided"
        assert src["hub_rankings"] == "not_provided"
        assert src["semantic_signals"] == "not_provided"
        assert src["delta"] == "not_provided"
        assert src["readiness"] == "not_provided"

    def test_sources_with_all_artifacts(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            classified_imports=_load_fixture("classified_imports.json"),
            entrypoints=_load_fixture("entrypoints.json"),
            graph=_load_fixture("graph.json"),
            orphan_triage=_load_fixture("orphan_triage.json"),
            hub_rankings=_load_fixture("hubs.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
            delta=_load_fixture("delta.json"),
            readiness=_load_fixture("readiness.json"),
        )
        src = result["sources"]
        for key in src:
            assert src[key] == "loaded", f"{key} should be loaded"


class TestSectionsPresence:
    """sections dict must contain scan section and optional sections."""

    def test_sections_with_scan_only(self):
        scan = _load_fixture("scan.json")
        result = report_data.build_report_data(scan=scan)
        assert "sections" in result
        sections = result["sections"]
        assert "scan" in sections
        assert "classification" not in sections or sections.get("classification") == "not_available"

    def test_scan_section_contains_summary(self):
        scan = _load_fixture("scan.json")
        result = report_data.build_report_data(scan=scan)
        sec = result["sections"]["scan"]
        assert sec["total_files"] == 10
        assert sec["total_lines"] == 500
        assert sec["project_root"] == "/tmp/test_project"

    def test_full_artifact_sections(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            classified_imports=_load_fixture("classified_imports.json"),
            entrypoints=_load_fixture("entrypoints.json"),
            graph=_load_fixture("graph.json"),
            orphan_triage=_load_fixture("orphan_triage.json"),
            hub_rankings=_load_fixture("hubs.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
            delta=_load_fixture("delta.json"),
            readiness=_load_fixture("readiness.json"),
        )
        sections = result["sections"]
        assert "scan" in sections
        assert "classification" in sections
        assert "entrypoints" in sections
        assert "graph_analysis" in sections
        assert "orphan_triage" in sections
        assert "hub_rankings" in sections
        assert "semantic_signals" in sections
        assert "delta" in sections
        assert "readiness" in sections


class TestClassificationSection:
    """Classification section must present totals and file counts."""

    def test_classification_totals(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            classified_imports=_load_fixture("classified_imports.json"),
        )
        cls = result["sections"]["classification"]
        assert cls["totals"]["stdlib"] == 5
        assert cls["totals"]["third_party"] == 1
        assert cls["totals"]["local"] == 1
        assert cls["totals"]["relative"] == 2

    def test_classification_not_available(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert result["sections"].get("classification") == "not_available"


class TestEntrypointsSection:
    """Entrypoints section must list entrypoint candidates."""

    def test_entrypoints_list(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            entrypoints=_load_fixture("entrypoints.json"),
        )
        eps = result["sections"]["entrypoints"]
        assert len(eps["entrypoints"]) == 2
        assert eps["totals"]["entrypoints_found"] == 2

    def test_entrypoints_not_available(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert result["sections"].get("entrypoints") == "not_available"


class TestOrphanTriageSection:
    """Orphan triage section must categorize orphans."""

    def test_orphan_counts(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            orphan_triage=_load_fixture("orphan_triage.json"),
        )
        triage = result["sections"]["orphan_triage"]
        assert triage["totals"]["suspicious"] == 2
        assert triage["totals"]["expected"] == 1

    def test_orphan_not_available(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert result["sections"].get("orphan_triage") == "not_available"


class TestHubRankingsSection:
    """Hub rankings section must list ranked hubs."""

    def test_hub_rankings(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            hub_rankings=_load_fixture("hubs.json"),
        )
        hubs = result["sections"]["hub_rankings"]
        assert len(hubs["hubs"]) == 5
        assert hubs["hubs"][0]["file_path"] == "src/app.py"
        assert hubs["hubs"][0]["hub_score"] == 3.0

    def test_hubs_not_available(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert result["sections"].get("hub_rankings") == "not_available"


class TestSemanticSignalsSection:
    """Semantic signals section must list symbol counts."""

    def test_semantic_totals(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
        )
        sem = result["sections"]["semantic_signals"]
        assert sem["totals"]["symbols"] == 6
        assert sem["totals"]["files_processed"] == 3

    def test_semantic_not_available(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert result["sections"].get("semantic_signals") == "not_available"


class TestDeltaSection:
    """Delta section must show added/removed files."""

    def test_delta_summary(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            delta=_load_fixture("delta.json"),
        )
        d = result["sections"]["delta"]
        assert "src/new_module.py" in d["files"]["added"]
        assert d["files"]["common_count"] == 10

    def test_delta_not_available(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert result["sections"].get("delta") == "not_available"


class TestReadinessSection:
    """Readiness section must show verification status."""

    def test_readiness_status(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            readiness=_load_fixture("readiness.json"),
        )
        r = result["sections"]["readiness"]
        assert r["verification_status"] == "verification_ready"
        assert r["detected_stacks"] == ["python"]

    def test_readiness_not_available(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert result["sections"].get("readiness") == "not_available"


# ── Warnings ─────────────────────────────────────────────────────────────


class TestWarnings:
    """Warnings must reflect missing artifacts and truncation."""

    def test_missing_artifact_warnings(self):
        scan = _load_fixture("scan.json")
        result = report_data.build_report_data(scan=scan)
        warnings = result["warnings"]
        # At minimum, warn about missing optional artifacts
        assert any("classified_imports" in w for w in warnings)
        assert any("entrypoints" in w for w in warnings)
        assert any("graph" in w for w in warnings)

    def test_no_warning_for_loaded_artifacts(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            classified_imports=_load_fixture("classified_imports.json"),
        )
        warnings = result["warnings"]
        assert not any("classified_imports" in w for w in warnings)


class TestTruncationWarnings:
    """Bounded lists must emit truncation warnings when exceeded."""

    def test_hubs_truncation_warning(self):
        """When hub_rankings exceed limit, a truncation warning must be emitted."""
        hubs_data = _load_fixture("hubs.json")
        # Create more hubs than the limit (MAX_HUB_RANKINGS = 20)
        extra_hubs = []
        for i in range(30):
            extra_hubs.append({
                "node_id": f"n_extra_{i}",
                "file_path": f"src/extra_{i}.py",
                "hub_score": float(30 - i),
                "in_degree": 1,
                "out_degree": 1,
                "confidence": "high",
            })
        hubs_data["hub_rankings"] = extra_hubs
        hubs_data["totals"]["files_ranked"] = len(extra_hubs)

        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            hub_rankings=hubs_data,
        )
        # Should have truncation warning about hubs
        assert any("hub_rankings" in w and "truncated" in w for w in result["warnings"])
        # The reading_plan should be bounded
        assert len(result["reading_plan"]) <= report_data.MAX_READING_PLAN

    def test_suspicious_orphans_truncation(self):
        """When suspicious orphans exceed limit, a truncation warning must be emitted."""
        orphans_data = _load_fixture("orphan_triage.json")
        extra_suspicious = []
        for i in range(200):
            extra_suspicious.append({
                "node_id": f"n_sus_{i}",
                "reason": "unreferenced source",
            })
        orphans_data["orphans"]["suspicious"] = extra_suspicious
        orphans_data["totals"]["suspicious"] = len(extra_suspicious)
        orphans_data["totals"]["total_orphans"] = len(extra_suspicious) + 1

        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            orphan_triage=orphans_data,
        )
        assert any("suspicious" in w and "truncated" in w for w in result["warnings"])


# ── Reading Plan ─────────────────────────────────────────────────────────


class TestReadingPlan:
    """Reading plan must combine candidates from multiple sources."""

    def test_reading_plan_exists(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert "reading_plan" in result
        assert isinstance(result["reading_plan"], list)

    def test_reading_plan_from_entrypoints(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            entrypoints=_load_fixture("entrypoints.json"),
        )
        plan = result["reading_plan"]
        entrypoint_items = [p for p in plan if p["type"] == "entrypoint"]
        assert len(entrypoint_items) >= 1
        # Check that entrypoint file paths are present
        ep_files = {p["file"] for p in entrypoint_items}
        assert "src/main.py" in ep_files

    def test_reading_plan_from_hubs(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            hub_rankings=_load_fixture("hubs.json"),
        )
        plan = result["reading_plan"]
        hub_items = [p for p in plan if p["type"] == "hub"]
        assert len(hub_items) >= 1
        hub_files = {p["file"] for p in hub_items}
        assert "src/app.py" in hub_files

    def test_reading_plan_from_suspicious_orphans(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            orphan_triage=_load_fixture("orphan_triage.json"),
        )
        plan = result["reading_plan"]
        sus_items = [p for p in plan if p["type"] == "suspicious_orphan"]
        assert len(sus_items) >= 1
        # n6 and n7 are suspicious; check that at least one file path is mapped
        sus_files = {p["file"] for p in sus_items}
        assert len(sus_files) >= 1

    def test_reading_plan_from_semantic_hotspots(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
        )
        plan = result["reading_plan"]
        hot_items = [p for p in plan if p["type"] == "semantic_hotspot"]
        assert len(hot_items) >= 1
        hot_files = {p["file"] for p in hot_items}
        assert "src/app.py" in hot_files

    def test_reading_plan_sorted_deterministically(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            entrypoints=_load_fixture("entrypoints.json"),
            hub_rankings=_load_fixture("hubs.json"),
            orphan_triage=_load_fixture("orphan_triage.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
        )
        plan = result["reading_plan"]
        # Verify sorted by confidence descending, then file ascending
        for i in range(len(plan) - 1):
            c1, c2 = plan[i].get("confidence", 0), plan[i + 1].get("confidence", 0)
            f1, f2 = plan[i].get("file", ""), plan[i + 1].get("file", "")
            assert (c1 > c2) or (c1 == c2 and f1 <= f2), \
                f"Plan not sorted at index {i}: {plan[i]} vs {plan[i+1]}"

    def test_reading_plan_bounded(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            entrypoints=_load_fixture("entrypoints.json"),
            hub_rankings=_load_fixture("hubs.json"),
            orphan_triage=_load_fixture("orphan_triage.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
        )
        assert len(result["reading_plan"]) <= report_data.MAX_READING_PLAN


# ── Totals ───────────────────────────────────────────────────────────────


class TestTotals:
    """totals must aggregate across all loaded sections."""

    def test_totals_scan_only(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        t = result["totals"]
        assert t["total_files"] == 10
        assert t["total_lines"] == 500
        assert t["languages"] == {"markdown": 1, "json": 2, "python": 7}

    def test_totals_with_classification(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            classified_imports=_load_fixture("classified_imports.json"),
        )
        t = result["totals"]
        assert t["classified_imports"]["stdlib"] == 5
        assert t["classified_imports"]["third_party"] == 1

    def test_totals_with_entrypoints(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            entrypoints=_load_fixture("entrypoints.json"),
        )
        t = result["totals"]
        assert t["entrypoints_count"] == 2

    def test_totals_with_hubs(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            hub_rankings=_load_fixture("hubs.json"),
        )
        t = result["totals"]
        assert t["hubs_count"] == 5

    def test_totals_with_semantic(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
        )
        t = result["totals"]
        assert t["symbol_count"] == 6


# ── Top-level structure ──────────────────────────────────────────────────


class TestTopLevelStructure:
    """Report data must have required top-level keys."""

    def test_required_keys(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        for key in ("schema_version", "sources", "sections", "warnings", "totals"):
            assert key in result, f"Missing required key: {key}"

    def test_warnings_is_list(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert isinstance(result["warnings"], list)

    def test_sources_is_dict(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert isinstance(result["sources"], dict)

    def test_sections_is_dict(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert isinstance(result["sections"], dict)

    def test_reading_plan_is_list(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        assert isinstance(result["reading_plan"], list)

    def test_reading_plan_item_structure(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            entrypoints=_load_fixture("entrypoints.json"),
        )
        for item in result["reading_plan"]:
            assert "file" in item
            assert "type" in item
            assert "priority" in item
            assert "confidence" in item
            assert "reason" in item


# ── Malformed JSON ───────────────────────────────────────────────────────


class TestMalformedInput:
    """Malformed JSON input should be handled gracefully."""

    def test_malformed_classified_imports(self):
        """If classified_imports JSON is malformed, it should be skipped with a warning."""
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            classified_imports={"not": "valid_data"},
        )
        warnings = result["warnings"]
        # Should warn about invalid data
        assert any("classified_imports" in w for w in warnings)
        # Classification section should be not_available
        assert result["sections"].get("classification") == "not_available"

    def test_malformed_entrypoints(self):
        result = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            entrypoints={"garbage": True},
        )
        assert result["sections"].get("entrypoints") == "not_available"
        assert any("entrypoints" in w for w in result["warnings"])


# ── CLI tests ────────────────────────────────────────────────────────────


class TestCLI:
    """CLI must accept artifacts via --scan and optional flags."""

    def test_cli_scan_only(self, tmp_path):
        out = tmp_path / "out.json"
        result = _run_cli([
            "--scan", _fixture("scan.json"),
            "--output", str(out),
        ])
        assert result.returncode == 0
        data = json.loads(out.read_text())
        assert data["schema_version"] == "1.0.0"
        assert data["sources"]["scan"] == "loaded"

    def test_cli_full_artifacts(self, tmp_path):
        out = tmp_path / "out.json"
        result = _run_cli([
            "--scan", _fixture("scan.json"),
            "--classified-imports", _fixture("classified_imports.json"),
            "--entrypoints", _fixture("entrypoints.json"),
            "--graph", _fixture("graph.json"),
            "--orphan-triage", _fixture("orphan_triage.json"),
            "--hubs", _fixture("hubs.json"),
            "--semantic", _fixture("semantic_signals.json"),
            "--delta", _fixture("delta.json"),
            "--readiness", _fixture("readiness.json"),
            "--output", str(out),
        ])
        assert result.returncode == 0
        data = json.loads(out.read_text())
        for key in ("classified_imports", "entrypoints", "graph", "orphan_triage",
                     "hub_rankings", "semantic_signals", "delta", "readiness"):
            assert data["sources"][key] == "loaded", f"{key} should be loaded"

    def test_cli_missing_scan_fails(self):
        result = _run_cli([])
        assert result.returncode != 0

    def test_cli_missing_file_gives_warning(self, tmp_path):
        out = tmp_path / "out.json"
        result = _run_cli([
            "--scan", _fixture("scan.json"),
            "--classified-imports", "/nonexistent/file.json",
            "--output", str(out),
        ])
        assert result.returncode == 0
        data = json.loads(out.read_text())
        assert data["sources"]["classified_imports"] == "not_provided"

    def test_cli_malformed_scan_fails(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        out = tmp_path / "out.json"
        result = _run_cli([
            "--scan", str(bad),
            "--output", str(out),
        ])
        assert result.returncode != 0

    def test_cli_json_stdout_only(self, tmp_path):
        """When no --output is given, JSON goes to stdout only."""
        result = _run_cli([
            "--scan", _fixture("scan.json"),
        ])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["schema_version"] == "1.0.0"
        # stderr should be empty or only contain warnings
        # (no JSON data leaked to stderr)
        if result.stderr.strip():
            # stderr may contain warnings, but not JSON
            try:
                json.loads(result.stderr)
                pytest.fail("JSON leaked to stderr")
            except json.JSONDecodeError:
                pass  # OK - stderr has non-JSON content


# ── Determinism ──────────────────────────────────────────────────────────


class TestDeterminism:
    """Output must be deterministic (repeated runs produce identical JSON)."""

    def test_deterministic_output(self):
        r1 = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            classified_imports=_load_fixture("classified_imports.json"),
            entrypoints=_load_fixture("entrypoints.json"),
            hub_rankings=_load_fixture("hubs.json"),
            orphan_triage=_load_fixture("orphan_triage.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
            delta=_load_fixture("delta.json"),
            readiness=_load_fixture("readiness.json"),
        )
        r2 = report_data.build_report_data(
            scan=_load_fixture("scan.json"),
            classified_imports=_load_fixture("classified_imports.json"),
            entrypoints=_load_fixture("entrypoints.json"),
            hub_rankings=_load_fixture("hubs.json"),
            orphan_triage=_load_fixture("orphan_triage.json"),
            semantic_signals=_load_fixture("semantic_signals.json"),
            delta=_load_fixture("delta.json"),
            readiness=_load_fixture("readiness.json"),
        )
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2, "Output is not deterministic"


# ── No-fabrication guard ─────────────────────────────────────────────────


class TestNoFabrication:
    """Missing data must never be fabricated."""

    def test_no_fabricated_classification(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        cls = result["sections"].get("classification")
        # With no classified_imports data, this must be "not_available"
        assert cls == "not_available"

    def test_no_fabricated_entrypoints(self):
        result = report_data.build_report_data(scan=_load_fixture("scan.json"))
        eps = result["sections"].get("entrypoints")
        # With no entrypoints data, this must be "not_available"
        assert eps == "not_available"
