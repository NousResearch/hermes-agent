"""Tests for scripts/code-scan/render_report.py.

Covers: full report rendering, scan-only report rendering, missing sections,
caveat placement, size cap, truncation warning, deterministic-hint wording,
and CLI behavior.

Strict TDD: these tests MUST fail before render_report.py is implemented.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure scripts/code-scan is importable
_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "code-scan"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import render_report

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "report_data"
_RENDER_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "render_report"


def _fixture(name: str) -> str:
    return str(_FIXTURE_DIR / name)


def _render_fixture(name: str) -> Path:
    return _RENDER_FIXTURE_DIR / name


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text())


def _make_report_data(**overrides) -> dict:
    """Build a minimal report-data dict suitable for rendering."""
    base = {
        "schema_version": "1.0.0",
        "sources": {"scan": "loaded"},
        "sections": {},
        "reading_plan": [],
        "warnings": [],
        "totals": {},
    }
    base.update(overrides)
    return base


def _make_full_report_data() -> dict:
    """Build a full report-data dict with all sections populated."""
    scan = _load_fixture("scan.json")
    from report_data import build_report_data

    return build_report_data(
        scan=scan,
        classified_imports=_load_fixture("classified_imports.json"),
        entrypoints=_load_fixture("entrypoints.json"),
        graph=_load_fixture("graph.json"),
        orphan_triage=_load_fixture("orphan_triage.json"),
        hub_rankings=_load_fixture("hubs.json"),
        semantic_signals=_load_fixture("semantic_signals.json"),
        delta=_load_fixture("delta.json"),
        readiness=_load_fixture("readiness.json"),
    )


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run render_report.py as a subprocess and return the result."""
    return subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "render_report.py")] + args,
        capture_output=True,
        text=True,
    )


# ── Core rendering functions ─────────────────────────────────────────────

class TestRenderReportData:
    """render_report_data() must produce deterministic Markdown from JSON dict."""

    def test_returns_string(self):
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "2026-01-01T00:00:00Z",
                    "total_files": 42,
                    "total_lines": 1000,
                    "languages": {"python": 42},
                    "categories": {"source": 42},
                    "frameworks": [],
                },
            },
        )
        md = render_report.render_report_data(report)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_project_overview_section(self):
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/myproject",
                    "scanned_at": "2026-01-01T00:00:00Z",
                    "total_files": 10,
                    "total_lines": 500,
                    "languages": {"python": 7, "json": 2, "markdown": 1},
                    "categories": {"source": 7, "config": 2, "docs": 1},
                    "frameworks": ["fastapi"],
                },
            },
            totals={"total_files": 10, "total_lines": 500},
        )
        md = render_report.render_report_data(report)
        assert "# Project Overview" in md
        assert "10 files" in md or "10" in md
        assert "500 lines" in md or "500" in md

    def test_deterministic_inventory_section(self):
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "2026-01-01T00:00:00Z",
                    "total_files": 10,
                    "total_lines": 500,
                    "languages": {"python": 7, "json": 2},
                    "categories": {"source": 7},
                    "frameworks": [],
                },
                "classification": {
                    "totals": {"stdlib": 5, "third_party": 1, "local": 1, "relative": 2, "unknown": 0},
                    "files_classified": 5,
                },
            },
        )
        md = render_report.render_report_data(report)
        assert "# Deterministic Inventory" in md or "## Deterministic Inventory" in md

    def test_entrypoints_section(self):
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 10,
                    "total_lines": 500,
                    "languages": {"python": 7},
                    "categories": {"source": 7},
                    "frameworks": [],
                },
                "entrypoints": {
                    "entrypoints": [
                        {
                            "file": "src/main.py",
                            "type": "python_main",
                            "confidence": 0.95,
                            "signals": ["if __name__"],
                        },
                    ],
                    "totals": {"entrypoints_found": 1},
                },
            },
        )
        md = render_report.render_report_data(report)
        assert "## Entrypoints / Where to Start" in md or "# Entrypoints" in md
        assert "src/main.py" in md

    def test_hub_rankings_section(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        assert "## Architectural Hubs" in md or "# Architectural Hubs" in md
        assert "src/app.py" in md

    def test_import_profile_section(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        assert "## Import Profile" in md
        assert "stdlib" in md

    def test_orphan_triage_section(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        assert "## Orphan Triage" in md or "# Orphan Triage" in md

    def test_semantic_signals_section(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        assert "## Semantic Signals" in md or "# Semantic Signals" in md

    def test_delta_section_when_present(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        assert "## Delta Summary" in md or "# Delta" in md

    def test_reading_path_section(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        assert "## Suggested Reading Path" in md or "# Suggested Reading Path" in md

    def test_caveats_section(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        assert "## Validation / Caveats" in md or "# Validation" in md or "# Caveats" in md


# ── Not-available sections ────────────────────────────────────────────────

class TestMissingSections:
    """Missing optional sections must render 'Not available' text."""

    def test_scan_only_has_not_available_for_optional(self):
        report_data = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 10,
                    "total_lines": 500,
                    "languages": {"python": 10},
                    "categories": {"source": 10},
                    "frameworks": [],
                },
                "classification": "not_available",
                "entrypoints": "not_available",
                "graph_analysis": "not_available",
                "orphan_triage": "not_available",
                "hub_rankings": "not_available",
                "semantic_signals": "not_available",
                "delta": "not_available",
                "readiness": "not_available",
            },
        )
        md = render_report.render_report_data(report_data)
        # For each missing section, we expect a "Not available" or "Not provided" indicator
        assert "Not available" in md or "Not provided" in md

    def test_graph_placeholder_when_not_available(self):
        report_data = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {},
                    "categories": {},
                    "frameworks": [],
                },
                "graph_analysis": "not_available",
            },
        )
        md = render_report.render_report_data(report_data)
        # Graph section should be present with placeholder text
        assert "Not available" in md


# ── Caveat / deterministic-hint wording ────────────────────────────────────

class TestCaveatWording:
    """Reports must label results as 'deterministic hints' and include caveats."""

    def test_hints_not_proof_disclaimer(self):
        """Report must say results are deterministic hints, not authoritative conclusions."""
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        lower = md.lower()
        assert "hint" in lower, "Report must describe results as hints"
        assert "caveat" in lower or "warn" in lower or "note" in lower or "not proof" in lower

    def test_no_llm_authoritative_language(self):
        """Report must NOT contain authoritative LLM-style language."""
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        lower = md.lower()
        # Must not claim definitive/authoritative conclusions
        assert "definitive proof" not in lower
        assert "llm concluded" not in lower
        assert "certified" not in lower

    def test_disclaimer_present_in_full_report(self):
        """Full report should include a disclaimer about deterministic interpretation."""
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        assert "deterministic" in md.lower()


# ── Size cap / truncation ─────────────────────────────────────────────────

class TestMaxBytesCap:
    """--max-bytes must enforce a size cap and emit a truncation warning."""

    def test_max_bytes_default(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        # Default cap should still accommodate a full report
        assert len(md.encode("utf-8")) <= render_report.DEFAULT_MAX_BYTES

    def test_max_bytes_truncation(self):
        """When output exceeds max_bytes, it must be truncated with a warning."""
        report = _make_full_report_data()
        # Set a very small limit
        md = render_report.render_report_data(report, max_bytes=100)
        marker_len = len(render_report._TRUNCATION_MARKER.encode("utf-8"))
        assert len(md.encode("utf-8")) <= 100 + marker_len
        assert "[TRUNCATED" in md

    def test_max_bytes_no_truncation_needed(self):
        """When under max_bytes, no truncation marker."""
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {"python": 1},
                    "categories": {},
                    "frameworks": [],
                },
            },
        )
        md = render_report.render_report_data(report, max_bytes=10000)
        assert "[TRUNCATED" not in md


# ── Safe markdown escaping ────────────────────────────────────────────────

class TestMarkdownEscaping:
    """Special characters in data paths must not break markdown."""

    def test_path_with_special_chars(self):
        """File paths with underscores or brackets should render safely."""
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {"python": 1},
                    "categories": {},
                    "frameworks": [],
                },
                "entrypoints": {
                    "entrypoints": [
                        {
                            "file": "src/test_module_v2_1.py",
                            "type": "python_main",
                            "confidence": 0.9,
                            "signals": ["__main__"],
                        },
                    ],
                    "totals": {"entrypoints_found": 1},
                },
            },
        )
        md = render_report.render_report_data(report)
        assert "src/test_module_v2_1.py" in md or "test_module" in md

    def test_no_raw_secrets(self):
        """If test fixture contains sensitive data, it must not leak as raw."""
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {},
                    "categories": {},
                    "frameworks": [],
                },
            },
        )
        md = render_report.render_report_data(report)
        # Should not contain obviously fake secret patterns
        assert "sk-" not in md.split("#")[0]  # Not in first heading


# ── Section ordering ──────────────────────────────────────────────────────

class TestSectionOrdering:
    """Sections must appear in a stable, deterministic order."""

    def test_section_order(self):
        """Project Overview must come before Entrypoints, which comes before Hubs."""
        report = _make_full_report_data()
        md = render_report.render_report_data(report)

        # Expected order of sections
        order = [
            "Project Overview",
            "Deterministic Inventory",
            "Entrypoints",
            "Architectural Hubs",
            "Import Profile",
            "Orphan Triage",
            "Semantic Signals",
        ]

        positions = []
        for section in order:
            pos = md.find(section)
            assert pos >= 0, f"Section '{section}' not found"
            positions.append(pos)

        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1], \
                f"Sections out of order: '{order[i]}' at {positions[i]} " \
                f"should be before '{order[i+1]}' at {positions[i+1]}"


# ── Warnings rendering ────────────────────────────────────────────────────

class TestWarningsRendering:
    """Warnings from report-data must be rendered in caveats section."""

    def test_warnings_rendered(self):
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {},
                    "categories": {},
                    "frameworks": [],
                },
            },
            warnings=["entrypoints: not provided", "graph: not found"],
        )
        md = render_report.render_report_data(report)
        assert "entrypoints" in md
        assert "not provided" in md or "not found" in md


# ── Reading plan rendering ────────────────────────────────────────────────

class TestReadingPlanRendering:
    """Reading plan candidates must be rendered as a ordered list."""

    def test_reading_plan_rendered(self):
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {},
                    "categories": {},
                    "frameworks": [],
                },
            },
            reading_plan=[
                {
                    "file": "src/main.py",
                    "type": "entrypoint",
                    "priority": "HIGH",
                    "confidence": 0.95,
                    "reason": "detected entrypoint",
                },
                {
                    "file": "src/app.py",
                    "type": "hub",
                    "priority": "MEDIUM",
                    "confidence": 0.3,
                    "reason": "hub_score=3",
                },
            ],
        )
        md = render_report.render_report_data(report)
        # Reading plan section heading
        assert "## Suggested Reading Path" in md or "# Suggested Reading Path" in md
        # Files listed
        assert "src/main.py" in md
        assert "src/app.py" in md
        # Priority labels
        assert "HIGH" in md
        assert "MEDIUM" in md


# ── Readiness section ─────────────────────────────────────────────────────

class TestReadinessRendering:
    """Readiness section must be rendered when present."""

    def test_readiness_rendered(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        # The readiness section should appear somewhere
        assert "## Readiness" in md or "# Readiness" in md or "readiness" in md.lower()


# ── Scan summary ──────────────────────────────────────────────────────────

class TestScanSummaryRendering:
    """Scan summary data must be rendered in the Project Overview."""

    def test_scan_summary_in_overview(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        scan = report["sections"]["scan"]
        # Check key values appear
        assert str(scan["total_files"]) in md
        assert str(scan["total_lines"]) in md


# ── CLI integration ───────────────────────────────────────────────────────

class TestCLI:
    """CLI must accept report-data JSON and produce Markdown."""

    def _make_temp_report(self, tmp_path, report_data: dict) -> Path:
        p = tmp_path / "report-data.json"
        p.write_text(json.dumps(report_data, indent=2))
        return p

    def test_cli_full_report(self, tmp_path):
        report_data = _make_full_report_data()
        inp = self._make_temp_report(tmp_path, report_data)
        out = tmp_path / "out.md"
        result = _run_cli([str(inp), "--output", str(out)])
        assert result.returncode == 0
        md = out.read_text()
        assert len(md) > 0
        assert "# Project Overview" in md

    def test_cli_scan_only(self, tmp_path):
        scan = _load_fixture("scan.json")
        from report_data import build_report_data
        report_data = build_report_data(scan=scan)
        inp = self._make_temp_report(tmp_path, report_data)
        out = tmp_path / "out.md"
        result = _run_cli([str(inp), "--output", str(out)])
        assert result.returncode == 0
        md = out.read_text()
        assert "# Project Overview" in md

    def test_cli_stdout(self, tmp_path):
        report_data = _make_full_report_data()
        inp = self._make_temp_report(tmp_path, report_data)
        result = _run_cli([str(inp)])
        assert result.returncode == 0
        assert "# Project Overview" in result.stdout

    def test_cli_max_bytes(self, tmp_path):
        report_data = _make_full_report_data()
        inp = self._make_temp_report(tmp_path, report_data)
        out = tmp_path / "out.md"
        result = _run_cli([str(inp), "--output", str(out), "--max-bytes", "500"])
        assert result.returncode == 0
        md = out.read_text()
        import render_report
        marker_len = len(render_report._TRUNCATION_MARKER.encode("utf-8"))
        assert len(md.encode("utf-8")) <= 500 + marker_len

    def test_cli_missing_file(self):
        result = _run_cli(["/nonexistent/report-data.json"])
        assert result.returncode != 0

    def test_cli_invalid_json(self, tmp_path):
        inp = tmp_path / "bad.json"
        inp.write_text("not json")
        result = _run_cli([str(inp)])
        assert result.returncode != 0

    def test_cli_deterministic_hint_wording(self, tmp_path):
        """CLI output must contain deterministic-hints wording."""
        report_data = _make_full_report_data()
        inp = self._make_temp_report(tmp_path, report_data)
        out = tmp_path / "out.md"
        result = _run_cli([str(inp), "--output", str(out)])
        assert result.returncode == 0
        md = out.read_text()
        assert "hint" in md.lower()


# ── Determinism ───────────────────────────────────────────────────────────

class TestDeterminism:
    """Report rendering must be deterministic."""

    def test_deterministic_rendering(self):
        report = _make_full_report_data()
        md1 = render_report.render_report_data(report)
        md2 = render_report.render_report_data(report)
        assert md1 == md2


# ── Sources tracking in report ────────────────────────────────────────────

class TestSourcesRendering:
    """Sources section must render which artifacts were loaded/missing."""

    def test_sources_rendered(self):
        report = _make_full_report_data()
        md = render_report.render_report_data(report)
        # Sources should be mentioned
        for key in report["sources"]:
            if report["sources"][key] == "loaded":
                assert key.replace("_", " ") in md.lower() or key in md

    def test_not_provided_sources(self):
        """Sources marked as not_provided should render as 'Not provided'."""
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {},
                    "categories": {},
                    "frameworks": [],
                },
            },
            sources={"scan": "loaded", "entrypoints": "not_provided"},
        )
        md = render_report.render_report_data(report)
        assert "Not provided" in md or "not provided" in md


# ── UA-P5-003: V2 Taxonomy rendering tests ────────────────────────────────


class TestV2TaxonomyRendering:
    """UA-P5-003: render_report must consume V2 triage shape or degrade cleanly."""

    def test_v2_triage_counts_render(self):
        """V2 enriched orphan entries should render correctly."""
        v2_triage = {
            "categories": {
                "expected": 2,
                "entrypoint_candidate": 1,
                "suspicious": 2,
                "unknown": 0,
            },
            "totals": {"total_orphans": 5},
        }
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {},
                    "categories": {},
                    "frameworks": [],
                },
                "orphan_triage": v2_triage,
            },
            sources={"scan": "loaded", "orphan_triage": "loaded"},
        )
        md = render_report.render_report_data(report)
        assert "Orphan Triage" in md
        # Category name from V2 section should appear
        assert "expected" in md.lower()

    def test_v2_triage_degrades_gracefully(self):
        """Old orphan_triage format should still render."""
        old_triage = {
            "categories": {"expected": 1, "entrypoint_candidate": 0, "suspicious": 1, "unknown": 0},
            "totals": {"total_orphans": 2},
        }
        report = _make_report_data(
            sections={
                "scan": {
                    "project_root": "/tmp/p",
                    "scanned_at": "",
                    "total_files": 1,
                    "total_lines": 10,
                    "languages": {},
                    "categories": {},
                    "frameworks": [],
                },
                "orphan_triage": old_triage,
            },
            sources={"scan": "loaded", "orphan_triage": "loaded"},
        )
        md = render_report.render_report_data(report)
        assert "Orphan Triage" in md
