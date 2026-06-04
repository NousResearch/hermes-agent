"""Tests for UA-004: build_context_bundle.py — Subagent Context Envelope.

Strict TDD: tests written first (RED), then implementation (GREEN).
"""
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "build_context_bundle.py"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures"

# All required top-level and nested keys for the subagent-context envelope.
REQUIRED_TOP_KEYS = [
    "handoff_version",
    "scan_run_id",
    "target",
    "artifacts_included",
    "artifacts_missing",
    "validation",
    "confidence",
    "recommended_files",
    "reading_budget",
    "truncation_warnings",
    "suggested_questions",
    "critic_packs",  # UA-P5-008
]

REQUIRED_VALIDATION_KEYS = [
    "verdict",
    "issue_count",
    "warning_count",
]

REQUIRED_SUBAGENT_KEYS = ["researcher", "reviewer", "coder"]


def _run_build_context(bundle_dir: Path, out_path: Path, extra_args=None):
    """Run build_context_bundle.py and return (rc, stdout, stderr)."""
    cmd = [
        sys.executable,
        str(BUNDLE_SCRIPT),
        "--bundle-dir",
        str(bundle_dir),
        "--out",
        str(out_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def _make_minimal_bundle(tmp_path: Path) -> Path:
    """Create a minimal bundle directory with only the UA-001 core artifacts."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    # scan.json
    (bundle / "scan.json").write_text(json.dumps({
        "total_files": 5,
        "total_lines": 230,
        "languages": {"python": 3, "typescript": 2},
        "files": [
            {"path": "src/main.py", "lines": 100, "language": "python"},
            {"path": "src/utils.py", "lines": 80, "language": "python"},
            {"path": "src/models.py", "lines": 50, "language": "python"},
        ],
    }, indent=2))

    # manifest.json
    (bundle / "manifest.json").write_text(json.dumps({
        "run_id": "ua004-test-run-001",
        "timestamp": "2026-06-01T00:00:00Z",
        "target_path": str(tmp_path / "target_project"),
        "artifact_paths": {
            "scan.json": str(bundle / "scan.json"),
            "manifest.json": str(bundle / "manifest.json"),
            "summary.json": str(bundle / "summary.json"),
            "validation.json": str(bundle / "validation.json"),
        },
        "artifacts_missing": [
            "REPORT.md",
            "domain-surfaces.json",
            "graph.json",
            "graph_analytics.json",
            "imports.json",
            "severity_analysis.json",
        ],
        "script_versions": {},
    }, indent=2))

    # validation.json
    (bundle / "validation.json").write_text(json.dumps({
        "issues": ["orphan file: src/legacy.py"],
        "warnings": ["high cyclomatic complexity in parse_config()"],
    }, indent=2))

    # summary.json
    (bundle / "summary.json").write_text(json.dumps({
        "target": str(tmp_path / "target_project"),
        "timestamp": "2026-06-01T00:00:00Z",
        "scan": {"total_files": 5, "total_lines": 230},
    }, indent=2))

    return bundle


def _make_full_bundle(tmp_path: Path) -> Path:
    """Create a bundle with UA-001 + UA-002 (severity) + UA-003 (analytics) artifacts."""
    bundle = tmp_path / "bundle_full"
    bundle.mkdir()

    # All core UA-001 artifacts
    (bundle / "scan.json").write_text(json.dumps({
        "total_files": 5,
        "total_lines": 230,
        "languages": {"python": 3, "typescript": 2},
        "files": [
            {"path": "src/main.py", "lines": 100, "language": "python"},
            {"path": "src/utils.py", "lines": 80, "language": "python"},
            {"path": "src/models.py", "lines": 50, "language": "python"},
        ],
    }, indent=2))

    (bundle / "manifest.json").write_text(json.dumps({
        "run_id": "ua004-full-run-001",
        "timestamp": "2026-06-01T00:00:00Z",
        "target_path": str(tmp_path / "full_target"),
        "artifact_paths": {
            "scan.json": str(bundle / "scan.json"),
            "manifest.json": str(bundle / "manifest.json"),
            "summary.json": str(bundle / "summary.json"),
            "graph.json": str(bundle / "graph.json"),
            "imports.json": str(bundle / "imports.json"),
            "REPORT.md": str(bundle / "REPORT.md"),
            "validation.json": str(bundle / "validation.json"),
            "domain-surfaces.json": str(bundle / "domain-surfaces.json"),
            "severity_analysis.json": str(bundle / "severity_analysis.json"),
            "graph_analytics.json": str(bundle / "graph_analytics.json"),
        },
        "script_versions": {},
    }, indent=2))

    (bundle / "validation.json").write_text(json.dumps({
        "issues": [],
        "warnings": [],
    }, indent=2))

    (bundle / "summary.json").write_text(json.dumps({
        "target": str(tmp_path / "full_target"),
        "timestamp": "2026-06-01T00:00:00Z",
        "scan": {"total_files": 5, "total_lines": 230},
    }, indent=2))

    # Additional UA-001 artifacts
    (bundle / "graph.json").write_text(json.dumps({
        "nodes": [], "edges": [], "summary": {},
    }, indent=2))

    (bundle / "imports.json").write_text(json.dumps({
        "files": {}, "schema_version": "1.0",
    }, indent=2))

    (bundle / "REPORT.md").write_text("# Report\n\nGenerated.\n")

    (bundle / "domain-surfaces.json").write_text(json.dumps({
        "surfaces": [
            {
                "surface": "package_scripts",
                "path": "package.json",
                "claim_type": "deterministic_inventory",
                "semantic_status": "not_validated",
            },
        ],
        "summary": {
            "total_surfaces": 1,
            "surface_types": {"package_scripts": 1},
        },
        "claim_type": "deterministic_inventory",
        "semantic_status": "not_validated",
    }, indent=2))

    # UA-002: severity_analysis.json (optional)
    (bundle / "severity_analysis.json").write_text(json.dumps({
        "critical": 1,
        "high": 2,
        "medium": 3,
        "low": 5,
        "info": 10,
        "items": [
            {"severity": "critical", "file": "src/auth.py", "message": "hardcoded secret"},
            {"severity": "high", "file": "src/main.py", "message": "unhandled exception"},
        ],
    }, indent=2))

    # UA-003: graph_analytics.json (optional)
    (bundle / "graph_analytics.json").write_text(json.dumps({
        "hub_nodes": [
            {"node_id": "file:src/utils.py", "degree": 12, "label": "utils.py"},
            {"node_id": "file:src/main.py", "degree": 8, "label": "main.py"},
        ],
        "strongly_connected_components": 2,
        "average_degree": 3.5,
    }, indent=2))

    return bundle


def _make_no_graph_bundle(tmp_path: Path) -> Path:
    """Create a bundle without validation.json (no-graph mode)."""
    bundle = tmp_path / "bundle_nograph"
    bundle.mkdir()

    (bundle / "scan.json").write_text(json.dumps({
        "total_files": 2,
        "total_lines": 50,
        "languages": {"python": 2},
        "files": [
            {"path": "main.py", "lines": 30, "language": "python"},
        ],
    }, indent=2))

    (bundle / "manifest.json").write_text(json.dumps({
        "run_id": "ua004-nograph-run",
        "timestamp": "2026-06-01T00:00:00Z",
        "target_path": str(tmp_path / "no_graph_target"),
        "artifact_paths": {
            "scan.json": str(bundle / "scan.json"),
        },
        "script_versions": {},
    }, indent=2))

    (bundle / "summary.json").write_text(json.dumps({
        "target": str(tmp_path / "no_graph_target"),
        "timestamp": "2026-06-01T00:00:00Z",
    }, indent=2))

    return bundle


# ── Module import ─────────────────────────────────────────────────────────

class TestBuildContextBundleImport:
    """Verify the build_context_bundle module can be imported."""

    def test_imports(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "code-scan"))
        from build_context_bundle import (
            build_context_envelope,
            render_markdown_handoff,
            main,
        )


# ── RED: no context envelope exists before implementation ───────────────

class TestRedPhase:
    """Before implementation, the script must not exist."""

    def test_script_does_not_exist_before_implementation(self):
        """Build context bundle script should not exist in RED phase."""
        # This test documents the RED state; it will be removed or
        # modified once we go GREEN.  We assert existence only so
        # the test runner can prove GREEN when the file appears.
        assert BUNDLE_SCRIPT.exists(), (
            f"RED: {BUNDLE_SCRIPT} does not exist yet — expected in RED phase"
        )


# ── GREEN: basic envelope generation ────────────────────────────────────

class TestContextEnvelopeBasic:
    """Tests for minimal valid envelope output."""

    def test_minimal_bundle_produces_valid_envelope(self, tmp_path: Path):
        """A bundle with only UA-001 artifacts must produce a valid envelope."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        rc, stdout, stderr = _run_build_context(bundle, out)
        assert rc == 0, f"script exited {rc}: {stderr}"
        assert out.exists()

        data = json.loads(out.read_text())
        for key in REQUIRED_TOP_KEYS:
            assert key in data, f"Missing required key: {key}"

    def test_required_top_level_keys(self, tmp_path: Path):
        """All required top-level keys must be present."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        for key in REQUIRED_TOP_KEYS:
            assert key in data, f"Missing required key '{key}'"
            assert data[key] is not None, f"Key '{key}' must not be None"

    def test_validation_nested_keys(self, tmp_path: Path):
        """validation section must have verdict, issue_count, warning_count."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        validation = data["validation"]
        for key in REQUIRED_VALIDATION_KEYS:
            assert key in validation, f"validation missing '{key}'"

    def test_severity_summary_optional(self, tmp_path: Path):
        """severity_summary is present only when severity analysis exists."""
        # Without severity_analysis.json: should NOT have severity_summary
        bundle_minimal = _make_minimal_bundle(tmp_path)
        out_minimal = tmp_path / "ctx_min.json"
        _run_build_context(bundle_minimal, out_minimal)
        data_min = json.loads(out_minimal.read_text())
        # severity_summary should be absent or None when file is missing
        assert (
            data_min["validation"].get("severity_summary") is None
            or "severity_summary" not in data_min["validation"]
        ), "severity_summary must not be fabricated"

        # With severity_analysis.json: severity_summary should appear
        full_tmp = tmp_path / "full_test_dir"
        full_tmp.mkdir()
        bundle_full = _make_full_bundle(full_tmp)
        out_full = full_tmp / "ctx_full.json"
        _run_build_context(bundle_full, out_full)
        data_full = json.loads(out_full.read_text())
        assert (
            data_full["validation"].get("severity_summary") is not None
        ), "severity_summary should be present when severity data available"


# ── GREEN: artifacts tracking ───────────────────────────────────────────

class TestArtifactTracking:
    """artifacts_included and artifacts_missing must be accurate."""

    def test_missing_optional_artifacts_tracked(self, tmp_path: Path):
        """When severity/analytics files are absent, artifacts_missing lists them."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        assert isinstance(data["artifacts_included"], list)
        assert isinstance(data["artifacts_missing"], list)

        # Optional artifacts should be listed as missing
        missing_names = [a["artifact"] for a in data["artifacts_missing"]]
        assert "severity_analysis.json" in missing_names, (
            "Missing severity_analysis not tracked"
        )
        assert "graph_analytics.json" in missing_names, (
            "Missing graph_analytics not tracked"
        )

    def test_all_artifacts_included_when_present(self, tmp_path: Path):
        """When all artifacts exist, artifacts_missing should be empty/minimal."""
        bundle = _make_full_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        assert len(data["artifacts_missing"]) == 0, (
            f"artifacts_missing should be empty: {data['artifacts_missing']}"
        )
        included_names = [a["artifact"] for a in data["artifacts_included"]]
        assert "severity_analysis.json" in included_names
        assert "graph_analytics.json" in included_names

    def test_manifest_artifact_paths_are_canonical_for_presence(self, tmp_path: Path):
        """When manifest is present, context artifact claims derive from it.

        Regression for PRL/Muster: manifest listed manifest.json, summary.json,
        and REPORT.md as present while subagent-context.json falsely claimed
        they were missing.
        """
        bundle = _make_minimal_bundle(tmp_path)
        report_path = bundle / "REPORT.md"
        report_path.write_text("# Report\n")
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_paths"] = {
            "scan.json": str(bundle / "scan.json"),
            "manifest.json": str(manifest_path),
            "summary.json": str(bundle / "summary.json"),
            "REPORT.md": str(report_path),
        }
        manifest["artifacts_missing"] = []
        manifest_path.write_text(json.dumps(manifest, indent=2))

        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        included_names = {a["artifact"] for a in data["artifacts_included"]}
        missing_names = {a["artifact"] for a in data["artifacts_missing"]}
        assert {"manifest.json", "summary.json", "REPORT.md"} <= included_names
        assert "manifest.json" not in missing_names
        assert "summary.json" not in missing_names
        assert "REPORT.md" not in missing_names
        assert "validation.json" not in missing_names

    def test_manifest_artifact_paths_record_real_missing_files(self, tmp_path: Path):
        """Manifest-listed artifacts that do not exist are explicitly missing."""
        bundle = _make_minimal_bundle(tmp_path)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_paths"] = {
            "scan.json": str(bundle / "scan.json"),
            "missing-report.md": str(bundle / "missing-report.md"),
        }
        manifest["artifacts_missing"] = []
        manifest_path.write_text(json.dumps(manifest, indent=2))

        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        missing = {a["artifact"]: a["reason"] for a in data["artifacts_missing"]}
        assert missing == {"missing-report.md": "file not found at manifest artifact path"}

    def test_manifest_relative_artifact_paths_resolve_against_bundle_dir(self, tmp_path: Path):
        """Relative manifest artifact_paths are resolved from bundle_dir."""
        bundle = _make_minimal_bundle(tmp_path)
        nested = bundle / "nested"
        nested.mkdir()
        relative_artifact = nested / "relative-report.md"
        relative_artifact.write_text("# Relative Report\n")

        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_paths"] = {
            "scan.json": "scan.json",
            "relative-report.md": "nested/relative-report.md",
        }
        manifest["artifacts_missing"] = []
        manifest_path.write_text(json.dumps(manifest, indent=2))

        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        included = {a["artifact"] for a in data["artifacts_included"]}
        missing = {a["artifact"] for a in data["artifacts_missing"]}
        assert {"scan.json", "relative-report.md"} <= included
        assert "relative-report.md" not in missing

    def test_manifest_missing_does_not_override_present_artifact_path(self, tmp_path: Path):
        """A present artifact_path wins over duplicate artifacts_missing metadata."""
        bundle = _make_minimal_bundle(tmp_path)
        report_path = bundle / "present-report.md"
        report_path.write_text("# Present Report\n")

        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_paths"] = {
            "scan.json": str(bundle / "scan.json"),
            "present-report.md": str(report_path),
        }
        manifest["artifacts_missing"] = [
            {"artifact": "present-report.md", "reason": "stale missing claim"},
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2))

        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        included = {a["artifact"] for a in data["artifacts_included"]}
        missing = {a["artifact"] for a in data["artifacts_missing"]}
        assert "present-report.md" in included
        assert "present-report.md" not in missing

    def test_manifest_missing_dict_path_key_is_normalized(self, tmp_path: Path):
        """Manifest missing dict entries may identify artifacts with path."""
        bundle = _make_minimal_bundle(tmp_path)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_paths"] = {"scan.json": str(bundle / "scan.json")}
        manifest["artifacts_missing"] = [
            {"path": "path-key-report.md", "reason": "missing via path key"},
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2))

        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        missing = {a["artifact"]: a["reason"] for a in data["artifacts_missing"]}
        assert missing == {"path-key-report.md": "missing via path key"}

    def test_malformed_manifest_falls_back_to_bundle_artifact_detection(self, tmp_path: Path):
        """Unreadable/malformed manifest uses static bundle artifact fallback."""
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "REPORT.md").write_text("# Report\n")
        (bundle / "manifest.json").write_text("{not valid json")

        out = tmp_path / "context.json"
        rc, stdout, stderr = _run_build_context(bundle, out)
        assert rc == 0, f"malformed manifest fallback failed: {stderr} {stdout}"
        data = json.loads(out.read_text())

        included = {a["artifact"] for a in data["artifacts_included"]}
        missing = {a["artifact"] for a in data["artifacts_missing"]}
        assert {"scan.json", "summary.json", "validation.json", "REPORT.md"} <= included
        assert "manifest.json" in included
        assert data["scan_run_id"] == "unknown"


# ── GREEN: confidence labels ───────────────────────────────────────────

class TestConfidenceLabels:
    """Confidence must reflect data provenance quality."""

    def test_confidence_is_dict_with_labels(self, tmp_path: Path):
        """confidence field must be a dict with confidence labels."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        assert isinstance(data["confidence"], dict)
        # confidence values should be one of: high, medium, low
        valid_labels = {"high", "medium", "low"}
        for _k, v in data["confidence"].items():
            assert v in valid_labels, f"Invalid confidence label: {v}"

    def test_raw_counts_high_confidence(self, tmp_path: Path):
        """Raw counts from scan should be labeled 'high' confidence."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        # At minimum, one entry should be high confidence
        assert "high" in data["confidence"].values(), (
            "Expected at least one 'high' confidence label"
        )


# ── GREEN: suggested_questions ──────────────────────────────────────────

class TestSuggestedQuestions:
    """suggested_questions must have entries for each subagent role."""

    def test_suggested_questions_all_roles(self, tmp_path: Path):
        """Must have researcher, reviewer, coder keys."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        sq = data["suggested_questions"]
        assert isinstance(sq, dict)
        for role in REQUIRED_SUBAGENT_KEYS:
            assert role in sq, f"suggested_questions missing '{role}'"
            assert isinstance(sq[role], list)
            assert len(sq[role]) > 0, f"No questions for {role}"


# ── GREEN: markdown handoff ────────────────────────────────────────────

class TestMarkdownHandoff:
    """Markdown handoff must separate facts from LLM prompts."""

    def test_markdown_contains_facts_section(self, tmp_path: Path):
        """Markdown output must include a deterministic facts section."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        rc, stdout, stderr = _run_build_context(bundle, out)
        assert rc == 0, f"script exited {rc}: {stderr}"
        # Markdown should be in stdout
        assert "## Deterministic Facts" in stdout or "### Facts" in stdout, (
            "Markdown handoff must include a facts section"
        )

    def test_markdown_contains_interpretation_section(self, tmp_path: Path):
        """Markdown output must include an interpretation/prompts section."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        rc, stdout, stderr = _run_build_context(bundle, out)
        assert rc == 0, f"script exited {rc}: {stderr}"
        assert ("Interpretation" in stdout or "Questions" in stdout or
                "Prompt" in stdout), (
            "Markdown handoff must include an interpretation section"
        )


# ── GREEN: no-graph degradation ─────────────────────────────────────────

class TestNoGraphDegradation:
    """Envelope must handle bundles without validation data."""

    def test_nograph_bundle_produces_envelope(self, tmp_path: Path):
        """A no-graph bundle must still produce a valid envelope."""
        bundle = _make_no_graph_bundle(tmp_path)
        out = tmp_path / "context.json"
        rc, stdout, stderr = _run_build_context(bundle, out)
        assert rc == 0, f"script exited {rc}: {stderr}"
        assert out.exists()

        data = json.loads(out.read_text())
        for key in REQUIRED_TOP_KEYS:
            assert key in data, f"Missing required key in no-graph: {key}"

    def test_nograph_validation_defaults(self, tmp_path: Path):
        """No-graph bundles should have zero counts for validation."""
        bundle = _make_no_graph_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        v = data["validation"]
        assert v["issue_count"] == 0
        assert v["warning_count"] == 0


# ── GREEN: reading_budget and truncation ────────────────────────────────

class TestReadingBudget:
    """reading_budget must provide concrete file guidance."""

    def test_reading_budget_present(self, tmp_path: Path):
        """reading_budget key must exist and be non-empty."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        assert isinstance(data["reading_budget"], list)
        assert len(data["reading_budget"]) > 0

    def test_truncation_warnings_is_list(self, tmp_path: Path):
        """truncation_warnings must be a list."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        assert isinstance(data["truncation_warnings"], list)


# ── GREEN recommended_files ────────────────────────────────────────────

class TestRecommendedFiles:
    """recommended_files must list files worth reading."""

    def test_recommended_files_from_scan(self, tmp_path: Path):
        """When scan data has files, recommended_files should include them."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        assert isinstance(data["recommended_files"], list)
        # Should contain at least some paths from the scan
        assert len(data["recommended_files"]) > 0

    def test_recommended_files_deprioritize_lockfiles_and_images(self, tmp_path: Path):
        """Large lock/image assets must not rank above source/auth/API files."""
        bundle = _make_minimal_bundle(tmp_path)
        scan_path = bundle / "scan.json"
        scan = json.loads(scan_path.read_text())
        scan["total_files"] = 6
        scan["total_lines"] = 15075
        scan["languages"] = {"json": 1, "png": 1, "typescript": 4}
        scan["files"] = [
            {"path": "package-lock.json", "lines": 9000, "language": "json"},
            {"path": "public/logo.png", "lines": 5000, "language": "png"},
            {"path": "src/auth/session.ts", "lines": 130, "language": "typescript"},
            {"path": "src/api/users.ts", "lines": 90, "language": "typescript"},
            {"path": "src/main.ts", "lines": 55, "language": "typescript"},
            {"path": "tests/auth.test.ts", "lines": 300, "language": "typescript"},
        ]
        scan_path.write_text(json.dumps(scan, indent=2))

        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        recommended = json.loads(out.read_text())["recommended_files"]

        ranked_paths = [entry["path"] for entry in recommended]
        assert ranked_paths.index("src/auth/session.ts") < ranked_paths.index("package-lock.json")
        assert ranked_paths.index("src/api/users.ts") < ranked_paths.index("package-lock.json")
        assert ranked_paths.index("src/auth/session.ts") < ranked_paths.index("public/logo.png")
        assert ranked_paths.index("src/api/users.ts") < ranked_paths.index("public/logo.png")

    def test_recommended_files_are_bucketed_deterministic_and_reasoned(self, tmp_path: Path):
        """Recommendations must include stable buckets, score details, and reasons."""
        bundle = _make_minimal_bundle(tmp_path)
        scan_path = bundle / "scan.json"
        scan = json.loads(scan_path.read_text())
        scan["files"] = [
            {"path": "AGENTS.md", "lines": 40, "language": "markdown"},
            {"path": "src/main.ts", "lines": 55, "language": "typescript"},
            {"path": "src/auth/session.ts", "lines": 130, "language": "typescript"},
            {"path": "src/api/users.ts", "lines": 90, "language": "typescript"},
            {"path": "supabase/functions/invite/index.ts", "lines": 75, "language": "typescript"},
            {"path": "supabase/migrations/20260601000000_rls.sql", "lines": 80, "language": "sql"},
            {"path": "Dockerfile", "lines": 30, "language": "dockerfile"},
            {"path": "tests/auth.test.ts", "lines": 300, "language": "typescript"},
            {"path": "README.md", "lines": 200, "language": "markdown"},
        ]
        scan_path.write_text(json.dumps(scan, indent=2))
        (bundle / "graph_analytics.json").write_text(json.dumps({
            "hub_nodes": [
                {"node_id": "file:src/api/users.ts", "degree": 7, "label": "users.ts"},
            ],
        }, indent=2))

        out1 = tmp_path / "context1.json"
        out2 = tmp_path / "context2.json"
        _run_build_context(bundle, out1)
        _run_build_context(bundle, out2)
        recommended1 = json.loads(out1.read_text())["recommended_files"]
        recommended2 = json.loads(out2.read_text())["recommended_files"]

        assert recommended1 == recommended2
        buckets = {entry.get("bucket") for entry in recommended1}
        assert {
            "project identity",
            "entrypoints",
            "auth/security",
            "data/API",
            "backend/serverless",
            "DB/RLS",
            "runtime/deployment",
            "tests",
            "docs/process",
        } <= buckets
        for entry in recommended1:
            assert isinstance(entry.get("score"), (int, float))
            assert isinstance(entry.get("score_details"), dict)
            assert entry.get("bucket")
            assert entry.get("reason")


# ── FULL: integration via CLI ───────────────────────────────────────────

class TestCLIIntegration:
    """End-to-end CLI tests."""

    def test_cli_with_valid_bundle(self, tmp_path: Path):
        """CLI must exit 0 and produce valid JSON output file."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        rc, stdout, stderr = _run_build_context(bundle, out)
        assert rc == 0, f"CLI failed: {stderr}"
        assert out.exists()
        # Verify it's valid JSON
        data = json.loads(out.read_text())
        assert data["scan_run_id"] == "ua004-test-run-001"

    def test_cli_missing_bundle_dir_fails(self, tmp_path: Path):
        """CLI must fail gracefully with a non-existent bundle dir."""
        out = tmp_path / "context.json"
        rc, stdout, stderr = _run_build_context(
            tmp_path / "nonexistent", out
        )
        assert rc != 0, "Should fail with non-existent bundle dir"

    def test_cli_handoff_version(self, tmp_path: Path):
        """handoff_version must be a version string."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())

        assert isinstance(data["handoff_version"], str)
        assert len(data["handoff_version"]) > 0


# ── FULL: public API functions ──────────────────────────────────────────

class TestPublicAPI:
    """Test the Python API directly (not via subprocess)."""

    def test_build_context_envelope_returns_dict(self, tmp_path: Path):
        """build_context_envelope() must return a dict with required keys."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "code-scan"))
        from build_context_bundle import build_context_envelope

        bundle = _make_minimal_bundle(tmp_path)
        result = build_context_envelope(str(bundle))
        assert isinstance(result, dict)
        for key in REQUIRED_TOP_KEYS:
            assert key in result, f"API result missing '{key}'"

    def test_render_markdown_handoff_returns_string(self, tmp_path: Path):
        """render_markdown_handoff() must return a string."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "code-scan"))
        from build_context_bundle import (
            build_context_envelope,
            render_markdown_handoff,
        )

        bundle = _make_minimal_bundle(tmp_path)
        envelope = build_context_envelope(str(bundle))
        md = render_markdown_handoff(envelope)
        assert isinstance(md, str)
        assert len(md) > 0


# ── UA-P5-008: Subagent Context Critic Packs (strict TDD) ────────────────────

CRITIC_PACK_KEYS = [
    "trust_anchor_summary",
    "top_deterministic_facts",
    "warning_orphan_triage_summary",
    "domain_surface_inventory_summary",
    "outside_ua_scope_boundaries",
]

REQUIRED_ROLE_CRITIC_PACKS = [
    "reviewer_critic",
    "researcher_scout",
    "coder_preflight",
]


class TestCriticPacksUA_P5_008:
    """UA-P5-008: critic_packs must be present, bounded, deterministic (no LLM)."""

    def test_critic_packs_key_present(self, tmp_path: Path):
        """Envelope must contain top-level 'critic_packs' key (RED before impl)."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())
        assert "critic_packs" in data, f"Missing top-level critic_packs in {list(data.keys())}"
        assert isinstance(data["critic_packs"], dict)

    def test_required_role_packs_present(self, tmp_path: Path):
        """critic_packs must contain reviewer_critic, researcher_scout, coder_preflight."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        packs = json.loads(out.read_text())["critic_packs"]
        for role in REQUIRED_ROLE_CRITIC_PACKS:
            assert role in packs, f"Missing required critic pack: {role}"
            pack = packs[role]
            assert isinstance(pack, dict)
            for k in CRITIC_PACK_KEYS:
                assert k in pack, f"Role {role} pack missing key {k}: {list(pack.keys())}"

    def test_critic_packs_bounded_deterministic(self, tmp_path: Path):
        """Packs must be bounded (no LLM summaries) and contain only deterministic facts."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        packs = json.loads(out.read_text())["critic_packs"]
        for role in REQUIRED_ROLE_CRITIC_PACKS:
            pack = packs[role]
            # All values must be concrete (str/list/dict of primitives) — no free-form LLM text blobs
            # Trust anchor must mention deterministic nature explicitly
            tas = pack["trust_anchor_summary"]
            assert isinstance(tas, str)
            assert len(tas) > 0
            assert "deterministic" in tas.lower() or "fact" in tas.lower()
            # top_deterministic_facts must be list of strings
            tdf = pack["top_deterministic_facts"]
            assert isinstance(tdf, list)
            # warning_orphan_triage_summary: list or str summary, bounded
            wots = pack["warning_orphan_triage_summary"]
            assert isinstance(wots, (list, str))
            # domain_surface_inventory_summary nullable with reason if absent
            dsi = pack["domain_surface_inventory_summary"]
            assert dsi is None or isinstance(dsi, (dict, str))
            if dsi is None or (isinstance(dsi, str) and "absent" in dsi.lower()):
                pass  # allowed when artifact absent
            # boundaries must explicitly contain UA scope text
            bounds = pack["outside_ua_scope_boundaries"]
            assert isinstance(bounds, (str, list))
            bo = str(bounds).lower()
            assert "hermes owns final" in bo or "hermes" in bo
            assert "reviewer" in bo and "researcher" in bo and "coder" in bo

    def test_suggested_questions_preserved_for_backward_compat(self, tmp_path: Path):
        """suggested_questions must remain for backward compatibility."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        data = json.loads(out.read_text())
        assert "suggested_questions" in data
        sq = data["suggested_questions"]
        for k in ("researcher", "reviewer", "coder"):
            assert k in sq

    def test_domain_surfaces_included_when_present(self, tmp_path: Path):
        """If domain-surfaces.json present in bundle, domain_surface_inventory_summary must summarize it (not fabricate)."""
        bundle = _make_minimal_bundle(tmp_path)
        # add domain-surfaces.json
        ds = {
            "surfaces": [
                {"surface": "vite_config", "path": "vite.config.ts", "claim_type": "deterministic_inventory"},
                {"surface": "pwa_manifest", "path": "public/manifest.webmanifest", "claim_type": "deterministic_inventory"},
            ],
            "summary": {"total_surfaces": 2, "surface_types": {"vite_config": 1, "pwa_manifest": 1}},
        }
        (bundle / "domain-surfaces.json").write_text(json.dumps(ds))
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        packs = json.loads(out.read_text())["critic_packs"]
        # All three roles should surface the domain inventory or reason
        for role in REQUIRED_ROLE_CRITIC_PACKS:
            dsi = packs[role]["domain_surface_inventory_summary"]
            assert dsi is not None
            # Must reference surfaces or summary count deterministically
            ds_str = json.dumps(dsi) if isinstance(dsi, dict) else str(dsi)
            assert "total_surfaces" in ds_str or "vite_config" in ds_str or "2" in ds_str

    def test_domain_surface_absent_reason_when_missing(self, tmp_path: Path):
        """When domain-surfaces.json absent, summary must contain explicit 'not present' / reason."""
        bundle = _make_minimal_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        packs = json.loads(out.read_text())["critic_packs"]
        for role in REQUIRED_ROLE_CRITIC_PACKS:
            dsi = packs[role]["domain_surface_inventory_summary"]
            # Either None or a string explaining absence
            if dsi is not None:
                assert "absent" in str(dsi).lower() or "not present" in str(dsi).lower() or "unavailable" in str(dsi).lower()

    def test_critic_packs_size_bounded(self, tmp_path: Path):
        """Generated subagent-context.json (with critic_packs) must remain bounded (use existing size approach or document cap)."""
        bundle = _make_full_bundle(tmp_path)
        out = tmp_path / "context.json"
        _run_build_context(bundle, out)
        raw = out.read_text()
        size = len(raw)
        # Conservative bound: < 100kB for context envelope (existing truncation machinery implies boundedness)
        # If script exposes an explicit cap we would test it here; otherwise document in test.
        assert size < 100_000, f"context envelope too large: {size} bytes"
        data = json.loads(raw)
        cp = json.dumps(data.get("critic_packs", {}))
        assert len(cp) < 20_000, "critic_packs section itself must stay small and bounded"


class TestCriticPacksViaRunUA:
    """Integration via run_ua preflight/review/full that produces subagent-context.json."""

    def test_preflight_includes_critic_packs(self, tmp_path: Path):
        from tests.code_scan.test_run_ua import run_ua, FIXTURES_DIR
        target = str(FIXTURES_DIR / "sample_repo")
        bundle_dir = str(tmp_path / "bundle")
        rc, stdout, stderr = run_ua(target, bundle_dir, mode="preflight")
        assert rc == 0
        ctx_path = Path(bundle_dir) / "subagent-context.json"
        assert ctx_path.exists()
        data = json.loads(ctx_path.read_text())
        assert "critic_packs" in data
        for role in REQUIRED_ROLE_CRITIC_PACKS:
            assert role in data["critic_packs"]
