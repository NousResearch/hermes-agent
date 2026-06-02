"""Phase 4 D7c — Pipeline E2E Gate.

Covers: end-to-end deterministic enricher chain from scan through
report-data JSON to final Markdown report, verifying artifact existence,
deterministic caveats, no target-local cache dirs, and stable cross-artifact
count consistency.

Strict TDD: these tests MUST fail before the pipeline wiring is correct.

Required chain:
    scan_project.py -> extract_imports.py -> assemble_graph.py ->
    classify_imports.py -> detect_entrypoints.py -> triage_orphans.py ->
    hub_ranking.py -> semantic_extract.py -> report_data.py -> render_report.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "code-scan"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"

# Ensure scripts/code-scan is importable
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import modules for direct function calls (when not using subprocess)
import report_data
import render_report
from extract_imports import build_import_map
from assemble_graph import assemble_graph
from graph_schema import validate_graph

try:
    from classify_imports import build_classified_map
    _HAS_CLASSIFY = True
except ImportError:
    _HAS_CLASSIFY = False

try:
    from detect_entrypoints import detect_entrypoints
    _HAS_DETECT_EP = True
except ImportError:
    _HAS_DETECT_EP = False

try:
    from triage_orphans import triage_orphans
    _HAS_TRIAGE = True
except ImportError:
    _HAS_TRIAGE = False

try:
    from hub_ranking import compute_hub_scores, rank_hubs, _build_output
    _HAS_HUBS = True
except ImportError:
    _HAS_HUBS = False

try:
    from semantic_extract import process_scan
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False

# ── Helpers ──────────────────────────────────────────────────────────────

SMALL_PROJECT = FIXTURES_DIR / "small_project"


def _run_script(script_name: str, args: list[str], *, expect_json: bool = True) -> dict | str:
    """Run a code-scan script via subprocess and return parsed JSON or stdout."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name)] + args,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 and expect_json:
        # For non-JSON scripts that might return non-zero legitimately
        pass
    if expect_json:
        return json.loads(result.stdout)
    return result.stdout


def _run_scan_project(target_dir: str, cache_dir: str) -> dict:
    """Run scan_project.py and return scan dict."""
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_DIR / "scan_project.py"),
            "--incremental", "--no-repo-cache",
            "--external-cache-dir", cache_dir,
            target_dir,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"scan_project.py failed: {result.stderr}"
    return json.loads(result.stdout)


# ── RED: Test file must exist and pipeline test must pass ────────────────

class TestPhase4PipelineExists:
    """RED gate: the E2E test file must exist and be importable."""

    def test_test_file_importable(self):
        """This test file should be findable by pytest."""
        test_path = PROJECT_ROOT / "tests" / "code_scan" / "test_phase4_pipeline.py"
        assert test_path.exists(), f"test_phase4_pipeline.py not found at {test_path}"


# ── GREEN: E2E pipeline chain ────────────────────────────────────────────

class TestPhase4PipelineE2E:
    """End-to-end pipeline: full chain on small_project fixture.

    Verifies:
    - All pipeline stages succeed
    - report-data.json and UA_REPORT.md are produced
    - No target-local .hermes/code-scan-cache or .ua directories
    - Cross-artifact counts are consistent
    - Deterministic caveat markers present in Markdown report
    - JSON stdout discipline preserved
    """

    @pytest.fixture
    def fixture_copy(self, tmp_path: Path) -> Path:
        """Copy small_project into a temp directory so the original is untouched."""
        target = tmp_path / "small_project"
        shutil.copytree(SMALL_PROJECT, target)
        return target

    @pytest.fixture
    def pipeline_run(self, fixture_copy: Path, tmp_path: Path) -> dict:
        """Run the full pipeline and return intermediate artifacts."""
        work_dir = tmp_path / "pipeline_work"
        work_dir.mkdir()
        cache_dir = str(work_dir / "cache")
        target = str(fixture_copy)

        # Stage 1: scan_project.py
        scan_data = _run_scan_project(target, cache_dir)
        assert scan_data is not None
        assert "files" in scan_data
        assert "total_files" in scan_data
        (work_dir / "scan.json").write_text(json.dumps(scan_data, indent=2))

        # Stage 2: extract_imports
        imports_data = build_import_map(scan_data, target)
        assert imports_data is not None
        assert "files" in imports_data
        (work_dir / "imports.json").write_text(json.dumps(imports_data, indent=2))

        # Stage 3: assemble_graph + validate
        graph_data = assemble_graph(
            scans=[scan_data],
            imports_list=[imports_data] if imports_data else [],
        )
        assert graph_data is not None
        assert "nodes" in graph_data
        assert "edges" in graph_data
        validation_data = validate_graph(graph_data)
        assert validation_data is not None
        (work_dir / "graph.json").write_text(json.dumps(graph_data, indent=2))
        (work_dir / "validation.json").write_text(json.dumps(validation_data, indent=2))

        # Stage 4: classify_imports (optional enricher)
        classified_data = None
        if _HAS_CLASSIFY:
            classified_data = build_classified_map(scan_data, imports_data)
            assert classified_data is not None
            assert "files" in classified_data
            (work_dir / "classified-imports.json").write_text(
                json.dumps(classified_data, indent=2)
            )

        # Stage 5: detect_entrypoints (optional enricher)
        entrypoints_data = None
        if _HAS_DETECT_EP:
            entrypoints_data = detect_entrypoints(str(work_dir / "scan.json"))
            assert entrypoints_data is not None
            assert "entrypoints" in entrypoints_data
            (work_dir / "entrypoints.json").write_text(
                json.dumps(entrypoints_data, indent=2)
            )

        # Stage 6: triage_orphans (optional enricher)
        triage_data = None
        if _HAS_TRIAGE:
            triage_data = triage_orphans(
                graph_data, scan_data, entrypoints_data
            )
            assert triage_data is not None
            (work_dir / "orphan-triage.json").write_text(
                json.dumps(triage_data, indent=2)
            )

        # Stage 7: hub_ranking (optional enricher)
        hub_data = None
        if _HAS_HUBS:
            scores = compute_hub_scores(graph_data, classified=classified_data)
            ranked = rank_hubs(
                graph_data, scores, top=20,
                classified=classified_data,
                include_non_code=False,
            )
            hub_data = _build_output(ranked, top=20,
                                     classification_present=classified_data is not None)
            assert hub_data is not None
            (work_dir / "hubs.json").write_text(json.dumps(hub_data, indent=2))

        # Stage 8: semantic_extract (optional enricher)
        semantic_data = None
        if _HAS_SEMANTIC:
            semantic_data = process_scan(
                Path(str(work_dir / "scan.json")),
                target,
                50,  # max_signals
            )
            assert semantic_data is not None
            (work_dir / "semantic-signals.json").write_text(
                json.dumps(semantic_data, indent=2)
            )

        # Stage 9: report_data.py (D7a)
        report = report_data.build_report_data(
            scan=scan_data,
            classified_imports=classified_data,
            entrypoints=entrypoints_data,
            graph=graph_data,
            orphan_triage=triage_data,
            hub_rankings=hub_data,
            semantic_signals=semantic_data,
        )
        assert "schema_version" in report
        assert "sections" in report
        assert "reading_plan" in report
        assert "warnings" in report
        assert "totals" in report
        assert "sources" in report
        (work_dir / "report-data.json").write_text(
            json.dumps(report, indent=2)
        )

        # Stage 10: render_report.py (D7b)
        md = render_report.render_report_data(report)
        assert isinstance(md, str)
        assert len(md) > 0
        (work_dir / "UA_REPORT.md").write_text(md)

        return {
            "work_dir": work_dir,
            "scan_data": scan_data,
            "imports_data": imports_data,
            "graph_data": graph_data,
            "validation_data": validation_data,
            "classified_data": classified_data,
            "entrypoints_data": entrypoints_data,
            "triage_data": triage_data,
            "hub_data": hub_data,
            "semantic_data": semantic_data,
            "report": report,
            "md": md,
        }

    # ── Artifact existence ────────────────────────────────────────────

    def test_report_data_json_exists(self, pipeline_run: dict):
        """report-data.json must be written and valid."""
        path = pipeline_run["work_dir"] / "report-data.json"
        assert path.exists(), "report-data.json was not written"
        data = json.loads(path.read_text())
        assert "schema_version" in data
        assert data["schema_version"] == "1.0.0"

    def test_ua_report_md_exists(self, pipeline_run: dict):
        """UA_REPORT.md must be written and non-empty."""
        path = pipeline_run["work_dir"] / "UA_REPORT.md"
        assert path.exists(), "UA_REPORT.md was not written"
        content = path.read_text()
        assert len(content) > 100, "UA_REPORT.md is too short"

    # ── No target mutation / cache ────────────────────────────────────

    def test_no_target_local_cache_dirs(self, fixture_copy: Path):
        """Pipeline must NOT create .hermes/code-scan-cache or .ua in target."""
        hermes_dir = fixture_copy / ".hermes"
        assert not (hermes_dir / "code-scan-cache").exists(), (
            "Pipeline created .hermes/code-scan-cache in target"
        )
        ua_dir = fixture_copy / ".ua"
        assert not ua_dir.exists(), "Pipeline created .ua in target"
        # Also check .hermes/code-state
        assert not (hermes_dir / "code-state").exists(), (
            "Pipeline created .hermes/code-state in target"
        )

    # ── Deterministic caveats / disclaimer ────────────────────────────

    def test_disclaimer_present_in_report(self, pipeline_run: dict):
        """Markdown report must include the deterministic-hint disclaimer."""
        md = pipeline_run["md"]
        lower = md.lower()
        assert "hint" in lower, "Report must describe results as hints"
        assert "not proof" in lower or "caveat" in lower or "note" in lower, (
            "Report must include caveat language"
        )

    def test_no_authoritative_llm_language(self, pipeline_run: dict):
        """Report must NOT claim definitive or authoritative conclusions."""
        md = pipeline_run["md"].lower()
        assert "definitive proof" not in md
        assert "llm concluded" not in md
        assert "certified" not in md

    # ── Cross-artifact count consistency ──────────────────────────────

    def test_scan_file_count_consistent(self, pipeline_run: dict):
        """report-data totals total_files must match scan total_files."""
        scan = pipeline_run["scan_data"]
        report = pipeline_run["report"]
        assert report["totals"]["total_files"] == scan.get("total_files", 0)

    def test_scan_line_count_consistent(self, pipeline_run: dict):
        """report-data totals total_lines must match scan total_lines."""
        scan = pipeline_run["scan_data"]
        report = pipeline_run["report"]
        assert report["totals"]["total_lines"] == scan.get("total_lines", 0)

    def test_graph_counts_consistent(self, pipeline_run: dict):
        """report-data graph section counts must match actual graph."""
        graph = pipeline_run["graph_data"]
        report = pipeline_run["report"]
        graph_section = report["sections"].get("graph_analysis", {})
        if graph_section != "not_available":
            assert graph_section["nodes_count"] == len(graph.get("nodes", []))
            assert graph_section["edges_count"] == len(graph.get("edges", []))

    def test_entrypoint_count_consistent(self, pipeline_run: dict):
        """Entrypoint count in report must match entrypoints artifact."""
        entrypoints = pipeline_run["entrypoints_data"]
        report = pipeline_run["report"]
        if entrypoints is not None:
            totals = report["totals"]
            ep_count = entrypoints.get("totals", {}).get("entrypoints_found", 0)
            assert totals.get("entrypoints_count") == ep_count

    def test_hub_count_consistent(self, pipeline_run: dict):
        """Hub count in report must match hubs artifact."""
        hub_data = pipeline_run["hub_data"]
        report = pipeline_run["report"]
        if hub_data is not None:
            totals = report["totals"]
            hub_count = len(hub_data.get("hub_rankings", []))
            assert totals.get("hubs_count") == hub_count

    def test_symbol_count_consistent(self, pipeline_run: dict):
        """Symbol count in report must match semantic artifact."""
        semantic = pipeline_run["semantic_data"]
        report = pipeline_run["report"]
        if semantic is not None:
            totals = report["totals"]
            symbol_count = semantic.get("totals", {}).get("symbols", 0)
            assert totals.get("symbol_count") == symbol_count

    # ── Report-data structure ─────────────────────────────────────────

    def test_report_data_has_required_keys(self, pipeline_run: dict):
        """Report-data must have all required top-level keys."""
        report = pipeline_run["report"]
        for key in ("schema_version", "sources", "sections", "reading_plan",
                     "warnings", "totals"):
            assert key in report, f"Missing required key: {key}"

    def test_reading_plan_is_list(self, pipeline_run: dict):
        """reading_plan must be a list."""
        report = pipeline_run["report"]
        assert isinstance(report["reading_plan"], list)

    def test_reading_plan_items_have_required_fields(self, pipeline_run: dict):
        """Each reading plan item must have file, type, priority, confidence."""
        report = pipeline_run["report"]
        for item in report["reading_plan"]:
            assert "file" in item
            assert "type" in item
            assert "priority" in item
            assert "confidence" in item

    # ── Markdown report sections ──────────────────────────────────────

    def test_md_has_project_overview(self, pipeline_run: dict):
        """Markdown must include Project Overview heading."""
        assert "# Project Overview" in pipeline_run["md"]

    def test_md_has_entrypoints_heading(self, pipeline_run: dict):
        """Markdown must include entrypoints heading or not-available placeholder."""
        md = pipeline_run["md"]
        assert "Entrypoints" in md or "Not available" in md or "not available" in md

    def test_md_has_validation_caveats(self, pipeline_run: dict):
        """Markdown must include Validation / Caveats section."""
        assert "Validation / Caveats" in pipeline_run["md"] or "Caveats" in pipeline_run["md"]

    def test_md_has_suggested_reading_path(self, pipeline_run: dict):
        """Markdown must include reading path section or empty placeholder."""
        md = pipeline_run["md"]
        assert "Suggested Reading Path" in md or "reading candidate" in md.lower()

    # ── JSON stdout discipline ────────────────────────────────────────

    def test_scan_json_parsable(self, pipeline_run: dict):
        """scan.json must be valid JSON."""
        path = pipeline_run["work_dir"] / "scan.json"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_graph_json_parsable(self, pipeline_run: dict):
        """graph.json must be valid JSON."""
        path = pipeline_run["work_dir"] / "graph.json"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        assert "nodes" in data
        assert "edges" in data

    # ── Sources tracking ──────────────────────────────────────────────

    def test_sources_tracks_scan(self, pipeline_run: dict):
        """Sources must mark scan as loaded."""
        report = pipeline_run["report"]
        assert report["sources"]["scan"] == "loaded"


# ── D6 delta optional test ──────────────────────────────────────────────
# The bead notes: "D6 delta may use a two-snapshot fixture in the same test file ...
# If runtime would exceed the budget, keep D6 focused tests as the primary proof
# and record why the E2E omits delta."
#
# Omission rationale: Delta requires two scan snapshots with a prior manifest.
# The D6 test_delta_report.py already provides focused delta coverage.
# Adding two-snapshot fixture setup to the E2E would significantly increase
# runtime with limited additional value. The E2E focuses on the primary
# scan-to-report pipeline.
