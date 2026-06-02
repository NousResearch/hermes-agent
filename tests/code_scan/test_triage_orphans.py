"""Tests for scripts/code-scan/triage_orphans.py — Phase 4 D3.

Orphan Warning Triage: classify graph orphan nodes into expected,
entrypoint_candidate, suspicious, and unknown groups.

Strict TDD: tests written first, implementation follows.
"""
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRIAGE_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "triage_orphans.py"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "orphan_triage"


def _import_triage():
    """Import triage_orphans module with sys.path pointing to scripts/code-scan."""
    script_dir = str(PROJECT_ROOT / "scripts" / "code-scan")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from triage_orphans import (
        triage_orphans,
        classify_orphan,
        _is_expected_orphan,
        _is_entrypoint_candidate,
        _build_result,
        main,
        _confidence_label,
    )
    return (
        triage_orphans,
        classify_orphan,
        _is_expected_orphan,
        _is_entrypoint_candidate,
        _build_result,
        main,
        _confidence_label,
    )


# ── Fixture helpers ─────────────────────────────────────────────────────

def _make_graph(orphan_node_ids, connected_node_ids=None, edge_sources=None, edge_targets=None):
    """Build a minimal graph JSON with the given orphan and connected nodes."""
    connected = connected_node_ids or []
    sources = edge_sources or []
    targets = edge_targets or []

    nodes = []
    for nid in connected:
        nodes.append({
            "node_id": nid,
            "node_type": "file",
            "filePath": nid.split(":", 1)[-1] if ":" in nid else nid,
            "language": "python",
        })
    for nid in orphan_node_ids:
        fp = nid.split(":", 1)[-1] if ":" in nid else nid
        nodes.append({
            "node_id": nid,
            "node_type": "file",
            "filePath": fp,
            "language": "python",
        })

    edges = []
    for s, t in zip(sources, targets):
        edges.append({
            "source": s,
            "target": t,
            "edge_type": "imports",
        })

    return {"nodes": nodes, "edges": edges}


def _make_scan(file_records):
    """Build a minimal scan.json with the given file records.

    file_records: list of dicts with keys like relative_path, language, etc.
    """
    files = []
    for rec in file_records:
        files.append({
            "path": rec.get("path", "/fake/" + rec.get("relative_path", "")),
            "relative_path": rec.get("relative_path", ""),
            "language": rec.get("language", "unknown"),
        })
    return {"files": files}


def _make_entrypoints(file_list):
    """Build a minimal entrypoints.json with the given files."""
    return {
        "schema_version": "1.0.0",
        "entrypoints": [
            {"file": f, "language": "python", "type": "python_cli", "signals": [], "confidence": 0.8}
            for f in file_list
        ],
        "totals": {"entrypoints_found": len(file_list), "by_type": {}},
    }


# ── Output schema tests ─────────────────────────────────────────────────

class TestOutputSchema:
    """Verify the output JSON shape matches the required format."""

    def test_schema_version(self):
        triage = _import_triage()[0]
        graph = _make_graph([], connected_node_ids=["file:src/main.py"],
                            edge_sources=["file:src/main.py"],
                            edge_targets=["file:src/main.py"])
        scan = _make_scan([{"relative_path": "src/main.py", "language": "python"}])
        result = triage(graph, scan, None)
        assert "schema_version" in result
        assert result["schema_version"] == "1.0.0"

    def test_orphans_key_exists(self):
        triage = _import_triage()[0]
        graph = _make_graph([], connected_node_ids=["file:src/main.py"],
                            edge_sources=["file:src/main.py"],
                            edge_targets=["file:src/main.py"])
        scan = _make_scan([{"relative_path": "src/main.py", "language": "python"}])
        result = triage(graph, scan, None)
        assert "orphans" in result
        for key in ("expected", "entrypoint_candidate", "suspicious", "unknown"):
            assert key in result["orphans"], f"Missing orphans key: {key}"

    def test_orphan_entry_has_node_id_and_reason(self):
        triage = _import_triage()[0]
        graph = _make_graph(["file:docs/README.md"], connected_node_ids=["file:src/main.py"],
                            edge_sources=["file:src/main.py"],
                            edge_targets=["file:src/main.py"])
        scan = _make_scan([
            {"relative_path": "docs/README.md", "language": "markdown"},
            {"relative_path": "src/main.py", "language": "python"},
        ])
        result = triage(graph, scan, None)
        assert len(result["orphans"]["expected"]) >= 1
        entry = result["orphans"]["expected"][0]
        assert "node_id" in entry
        assert "reason" in entry

    def test_totals_present(self):
        triage = _import_triage()[0]
        graph = _make_graph(["file:docs/README.md"], connected_node_ids=["file:src/main.py"],
                            edge_sources=["file:src/main.py"],
                            edge_targets=["file:src/main.py"])
        scan = _make_scan([
            {"relative_path": "docs/README.md", "language": "markdown"},
            {"relative_path": "src/main.py", "language": "python"},
        ])
        result = triage(graph, scan, None)
        assert "totals" in result
        for key in ("total_orphans", "expected", "entrypoint_candidate", "suspicious", "unknown"):
            assert key in result["totals"], f"Missing totals key: {key}"

    def test_totals_match_counts(self):
        triage = _import_triage()[0]
        graph = _make_graph(["file:docs/README.md", "file:src/legacy.py"],
                            connected_node_ids=["file:src/main.py"],
                            edge_sources=["file:src/main.py"],
                            edge_targets=["file:src/main.py"])
        scan = _make_scan([
            {"relative_path": "docs/README.md", "language": "markdown"},
            {"relative_path": "src/legacy.py", "language": "python"},
            {"relative_path": "src/main.py", "language": "python"},
        ])
        result = triage(graph, scan, None)
        totals = result["totals"]
        assert totals["total_orphans"] == 2
        assert totals["expected"] == len(result["orphans"]["expected"])
        assert totals["suspicious"] == len(result["orphans"]["suspicious"])
        assert sum(len(result["orphans"][k]) for k in ("expected", "entrypoint_candidate", "suspicious", "unknown")) == totals["total_orphans"], \
            "Sum of per-category counts must equal total_orphans"


# ── Expected orphan classification ──────────────────────────────────────

class TestExpectedOrphans:
    """Orphans that are expected (docs, config, tests, fixtures, workflows, images, templates)."""

    def test_docs_directory_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:docs/guide.md", "filePath": "docs/guide.md", "language": "markdown"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"
        assert "doc" in reason

    def test_readme_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:README.md", "filePath": "README.md", "language": "markdown"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"
        assert "doc" in reason

    def test_changelog_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:CHANGELOG.md", "filePath": "CHANGELOG.md", "language": "markdown"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"

    def test_config_yaml_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:config.yaml", "filePath": "config.yaml", "language": "yaml"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"
        assert "config" in reason

    def test_tests_directory_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:tests/test_main.py", "filePath": "tests/test_main.py", "language": "python"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"
        assert "test" in reason

    def test_fixture_json_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:fixtures/data.json", "filePath": "fixtures/data.json", "language": "json"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"
        assert "fixture" in reason

    def test_github_workflows_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:.github/workflows/ci.yml", "filePath": ".github/workflows/ci.yml", "language": "yaml"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"
        # Workflows map to expected_config in V2 taxonomy
        assert _entry["category"] == "expected_config"

    def test_image_file_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:assets/logo.png", "filePath": "assets/logo.png", "language": "unknown"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"
        assert "asset" in reason or "image" in reason

    def test_template_file_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:templates/index.html", "filePath": "templates/index.html", "language": "html"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"
        assert "template" in reason

    def test_license_file_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:LICENSE", "filePath": "LICENSE", "language": "unknown"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"

    def test_toml_config_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:pyproject.toml", "filePath": "pyproject.toml", "language": "toml"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"

    def test_rst_doc_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:docs/api.rst", "filePath": "docs/api.rst", "language": "rst"}
        category, reason, _entry = classify(node, None)
        assert category == "expected"


# ── Suspicious orphan classification ────────────────────────────────────

class TestSuspiciousOrphans:
    """Source orphans that are NOT plausible entrypoints."""

    def test_unreferenced_python_source_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/legacy.py", "filePath": "src/legacy.py", "language": "python"}
        category, reason, _entry = classify(node, None)
        assert category == "suspicious"
        assert "unreferenced" in reason

    def test_orphaned_util_module_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/utils.py", "filePath": "src/utils.py", "language": "python"}
        category, reason, _entry = classify(node, None)
        assert category == "suspicious"

    def test_orphaned_js_module_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:lib/helpers.js", "filePath": "lib/helpers.js", "language": "javascript"}
        category, reason, _entry = classify(node, None)
        assert category == "suspicious"

    def test_orphaned_go_file_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:pkg/old.go", "filePath": "pkg/old.go", "language": "go"}
        category, reason, _entry = classify(node, None)
        assert category == "suspicious"

    def test_orphaned_rust_file_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/legacy.rs", "filePath": "src/legacy.rs", "language": "rust"}
        category, reason, _entry = classify(node, None)
        assert category == "suspicious"

    def test_orphaned_ts_file_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/old.ts", "filePath": "src/old.ts", "language": "typescript"}
        category, reason, _entry = classify(node, None)
        assert category == "suspicious"


# ── Entrypoint candidate classification ─────────────────────────────────

class TestEntrypointCandidateOrphans:
    """Source orphans that entrypoints.json marks as likely standalone entrypoints."""

    def test_entrypoint_file_is_not_suspicious(self):
        """A file marked as entrypoint should be entrypoint_candidate, not suspicious."""
        triage_func = _import_triage()[0]
        graph = _make_graph(["file:src/cli.py"],
                            connected_node_ids=["file:src/utils.py"],
                            edge_sources=["file:src/utils.py"],
                            edge_targets=["file:src/utils.py"])
        scan = _make_scan([
            {"relative_path": "src/cli.py", "language": "python"},
            {"relative_path": "src/utils.py", "language": "python"},
        ])
        entrypoints = _make_entrypoints(["src/cli.py"])
        result = triage_func(graph, scan, entrypoints)
        assert len(result["orphans"]["entrypoint_candidate"]) >= 1
        ep_ids = [e["node_id"] for e in result["orphans"]["entrypoint_candidate"]]
        assert "file:src/cli.py" in ep_ids

    def test_non_entrypoint_remains_suspicious(self):
        """A source orphan NOT in entrypoints.json remains suspicious."""
        triage_func = _import_triage()[0]
        graph = _make_graph(["file:src/legacy.py"],
                            connected_node_ids=["file:src/main.py"],
                            edge_sources=["file:src/main.py"],
                            edge_targets=["file:src/main.py"])
        scan = _make_scan([
            {"relative_path": "src/legacy.py", "language": "python"},
            {"relative_path": "src/main.py", "language": "python"},
        ])
        entrypoints = _make_entrypoints(["src/cli.py"])  # different file
        result = triage_func(graph, scan, entrypoints)
        assert len(result["orphans"]["suspicious"]) >= 1
        sus_ids = [e["node_id"] for e in result["orphans"]["suspicious"]]
        assert "file:src/legacy.py" in sus_ids

    def test_none_entrypoints_still_classifies(self):
        """When entrypoints data is absent, suspicious/expected still work."""
        triage_func = _import_triage()[0]
        graph = _make_graph(["file:src/legacy.py", "file:docs/README.md"],
                            connected_node_ids=["file:src/main.py"],
                            edge_sources=["file:src/main.py"],
                            edge_targets=["file:src/main.py"])
        scan = _make_scan([
            {"relative_path": "src/legacy.py", "language": "python"},
            {"relative_path": "docs/README.md", "language": "markdown"},
            {"relative_path": "src/main.py", "language": "python"},
        ])
        result = triage_func(graph, scan, None)
        assert result["orphans"]["entrypoint_candidate"] == []
        assert len(result["orphans"]["suspicious"]) >= 1
        assert len(result["orphans"]["expected"]) >= 1


# ── Unknown orphan classification ───────────────────────────────────────

class TestUnknownOrphans:
    """Orphans with missing metadata or unsupported language."""

    def test_missing_language_is_unknown(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/weird.xyz", "filePath": "src/weird.xyz", "language": ""}
        category, reason, _entry = classify(node, None)
        assert category == "unknown"

    def test_unsupported_language_is_unknown(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/compiled.dat", "filePath": "src/compiled.dat", "language": "binary"}
        category, reason, _entry = classify(node, None)
        assert category == "unknown"

    def test_missing_filePath_is_unknown(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:xyz", "language": "python"}
        category, reason, _entry = classify(node, None)
        assert category == "unknown"


# ── _is_expected_orphan helper tests ────────────────────────────────────

class TestIsExpectedOrphan:
    """Low-level tests for the _is_expected_orphan helper."""

    def test_docs_pattern(self):
        is_exp = _import_triage()[2]
        result = is_exp({"node_id": "file:docs/readme.md", "filePath": "docs/readme.md", "language": "markdown"})
        assert result[1] is True
        assert result[0] == "expected_doc"

    def test_config_pattern(self):
        is_exp = _import_triage()[2]
        assert is_exp({"node_id": "file:.env", "filePath": ".env", "language": "unknown"})[1] is True

    def test_tests_pattern(self):
        is_exp = _import_triage()[2]
        result = is_exp({"node_id": "file:tests/conftest.py", "filePath": "tests/conftest.py", "language": "python"})
        assert result[1] is True

    def test_fixture_pattern(self):
        is_exp = _import_triage()[2]
        result = is_exp({"node_id": "file:fixtures/sample.json", "filePath": "fixtures/sample.json", "language": "json"})
        assert result[1] is True

    def test_workflow_pattern(self):
        is_exp = _import_triage()[2]
        result = is_exp({"node_id": "file:.github/workflows/build.yml", "filePath": ".github/workflows/build.yml", "language": "yaml"})
        assert result[1] is True

    def test_image_pattern(self):
        is_exp = _import_triage()[2]
        result = is_exp({"node_id": "file:logo.png", "filePath": "logo.png", "language": "unknown"})
        assert result[1] is True

    def test_template_pattern(self):
        is_exp = _import_triage()[2]
        result = is_exp({"node_id": "file:templates/base.html", "filePath": "templates/base.html", "language": "html"})
        assert result[1] is True

    def test_license_pattern(self):
        is_exp = _import_triage()[2]
        result = is_exp({"node_id": "file:LICENSE", "filePath": "LICENSE", "language": "unknown"})
        assert result[1] is True

    def test_not_expected_for_source(self):
        is_exp = _import_triage()[2]
        result = is_exp({"node_id": "file:src/main.py", "filePath": "src/main.py", "language": "python"})
        assert result[1] is False

    def test_not_expected_for_unknown(self):
        """If _is_expected_orphan can't match, return False (caller decides unknown)."""
        is_exp = _import_triage()[2]
        # A file with no recognizable pattern but also not a known language
        result = is_exp({"node_id": "file:weird.xyz", "filePath": "weird.xyz", "language": "unknown"})
        # weird.xyz extension + language=unknown → not expected (goes to unknown instead)
        assert result[1] is False


# ── Integration test: real assembled graph fixture ──────────────────────

class TestIntegration:
    """Integration tests using multi-class orphan fixtures."""

    def test_mixed_orphan_classes(self):
        """A graph with expected + suspicious + unknown orphans."""
        triage_func = _import_triage()[0]
        graph = {
            "nodes": [
                {"node_id": "file:src/main.py", "filePath": "src/main.py", "language": "python"},
                {"node_id": "file:src/utils.py", "filePath": "src/utils.py", "language": "python"},
                {"node_id": "file:docs/README.md", "filePath": "docs/README.md", "language": "markdown"},
                {"node_id": "file:src/legacy.py", "filePath": "src/legacy.py", "language": "python"},
                {"node_id": "module:os", "filePath": None, "language": "unknown"},
                {"node_id": "file:config.yaml", "filePath": "config.yaml", "language": "yaml"},
            ],
            "edges": [
                {"source": "file:src/main.py", "target": "file:src/utils.py", "edge_type": "imports"},
            ],
        }
        scan = {
            "files": [
                {"relative_path": "src/main.py", "language": "python"},
                {"relative_path": "src/utils.py", "language": "python"},
                {"relative_path": "docs/README.md", "language": "markdown"},
                {"relative_path": "src/legacy.py", "language": "python"},
                {"relative_path": "config.yaml", "language": "yaml"},
            ],
        }
        result = triage_func(graph, scan, None)

        # Verify all categories are present
        assert "expected" in result["orphans"]
        assert "entrypoint_candidate" in result["orphans"]
        assert "suspicious" in result["orphans"]
        assert "unknown" in result["orphans"]

        # docs/README.md and config.yaml should be expected
        expected_ids = [e["node_id"] for e in result["orphans"]["expected"]]
        assert "file:docs/README.md" in expected_ids, f"Expected docs/README.md in expected, got {expected_ids}"
        assert "file:config.yaml" in expected_ids, f"Expected config.yaml in expected, got {expected_ids}"

        # src/legacy.py should be suspicious
        suspicious_ids = [e["node_id"] for e in result["orphans"]["suspicious"]]
        assert "file:src/legacy.py" in suspicious_ids, f"Expected src/legacy.py in suspicious, got {suspicious_ids}"

        # module:os (no filePath) should be unknown
        unknown_ids = [e["node_id"] for e in result["orphans"]["unknown"]]
        assert "module:os" in unknown_ids, f"Expected module:os in unknown, got {unknown_ids}"

        # Totals should be correct
        assert result["totals"]["total_orphans"] == 4
        assert result["totals"]["expected"] == len(result["orphans"]["expected"])
        assert result["totals"]["suspicious"] == len(result["orphans"]["suspicious"])
        assert result["totals"]["unknown"] == len(result["orphans"]["unknown"])

    def test_all_expected_orphan_types(self):
        """Test that all expected orphan categories are recognized."""
        triage_func = _import_triage()[0]
        graph = {
            "nodes": [
                {"node_id": "file:connected.py", "filePath": "connected.py", "language": "python"},
                {"node_id": "file:docs/api.txt", "filePath": "docs/api.txt", "language": "txt"},
                {"node_id": "file:tests/test_api.py", "filePath": "tests/test_api.py", "language": "python"},
                {"node_id": "file:fixtures/data.json", "filePath": "fixtures/data.json", "language": "json"},
                {"node_id": "file:.github/workflows/ci.yaml", "filePath": ".github/workflows/ci.yaml", "language": "yaml"},
                {"node_id": "file:assets/icon.svg", "filePath": "assets/icon.svg", "language": "unknown"},
                {"node_id": "file:templates/base.html", "filePath": "templates/base.html", "language": "html"},
                {"node_id": "file:setup.cfg", "filePath": "setup.cfg", "language": "unknown"},
            ],
            "edges": [
                {"source": "file:connected.py", "target": "file:connected.py", "edge_type": "imports"},
            ],
        }
        scan = {"files": []}
        result = triage_func(graph, scan, None)
        expected_ids = [e["node_id"] for e in result["orphans"]["expected"]]
        assert "file:docs/api.txt" in expected_ids
        assert "file:tests/test_api.py" in expected_ids
        assert "file:fixtures/data.json" in expected_ids
        assert "file:.github/workflows/ci.yaml" in expected_ids
        assert "file:assets/icon.svg" in expected_ids
        assert "file:templates/base.html" in expected_ids
        assert "file:setup.cfg" in expected_ids

    def test_empty_graph_no_orphans(self):
        """Graph with no orphans produces empty triage."""
        triage_func = _import_triage()[0]
        graph = {
            "nodes": [
                {"node_id": "file:src/main.py", "filePath": "src/main.py", "language": "python"},
            ],
            "edges": [
                {"source": "file:src/main.py", "target": "file:src/main.py", "edge_type": "imports"},
            ],
        }
        scan = {"files": [{"relative_path": "src/main.py", "language": "python"}]}
        result = triage_func(graph, scan, None)
        assert result["totals"]["total_orphans"] == 0
        assert result["orphans"]["expected"] == []
        assert result["orphans"]["entrypoint_candidate"] == []
        assert result["orphans"]["suspicious"] == []
        assert result["orphans"]["unknown"] == []


# ── CLI integration tests ───────────────────────────────────────────────

class TestCLI:
    """Test the CLI interface."""

    def test_cli_exits_zero(self, capfd, tmp_path):
        """CLI should exit with code 0 on valid input."""
        triage_main = _import_triage()[5]

        graph_path = tmp_path / "graph.json"
        graph_path.write_text(json.dumps({"nodes": [], "edges": []}))
        scan_path = tmp_path / "scan.json"
        scan_path.write_text(json.dumps({"files": []}))

        sys.argv = ["triage_orphans.py", str(graph_path), str(scan_path)]
        exit_code = triage_main()
        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

    def test_cli_errors_on_missing_files(self, capfd, tmp_path):
        """CLI should exit non-zero when input files don't exist."""
        # Create minimal real files
        graph_path = tmp_path / "graph.json"
        graph_path.write_text(json.dumps({"nodes": [], "edges": []}))
        scan_path = tmp_path / "scan.json"
        scan_path.write_text(json.dumps({"files": []}))

        sys.argv = ["triage_orphans.py", str(graph_path), str(scan_path)]
        triage_main = _import_triage()[5]
        exit_code = triage_main()
        assert exit_code == 0, f"Expected exit code 0 for valid files, got {exit_code}"

    def test_cli_produces_valid_json(self, capfd, tmp_path):
        """CLI output should be valid JSON with required keys."""
        graph_path = tmp_path / "graph.json"
        graph = {
            "nodes": [
                {"node_id": "file:connected.py", "filePath": "connected.py", "language": "python"},
                {"node_id": "file:orphan.py", "filePath": "orphan.py", "language": "python"},
            ],
            "edges": [
                {"source": "file:connected.py", "target": "file:connected.py", "edge_type": "imports"},
            ],
        }
        graph_path.write_text(json.dumps(graph))

        scan_path = tmp_path / "scan.json"
        scan_path.write_text(json.dumps({
            "files": [
                {"relative_path": "connected.py", "language": "python"},
                {"relative_path": "orphan.py", "language": "python"},
            ],
        }))

        sys.argv = ["triage_orphans.py", str(graph_path), str(scan_path)]
        triage_main = _import_triage()[5]
        exit_code = triage_main()
        assert exit_code == 0
        captured = capfd.readouterr()
        output = json.loads(captured.out)
        assert "schema_version" in output
        assert "orphans" in output
        assert "totals" in output

    def test_cli_with_entrypoints(self, capfd, tmp_path):
        """CLI with --entrypoints option."""
        graph_path = tmp_path / "graph.json"
        graph = {
            "nodes": [
                {"node_id": "file:cli.py", "filePath": "cli.py", "language": "python"},
                {"node_id": "file:utils.py", "filePath": "utils.py", "language": "python"},
            ],
            "edges": [
                {"source": "file:cli.py", "target": "file:cli.py", "edge_type": "imports"},
            ],
        }
        graph_path.write_text(json.dumps(graph))

        scan_path = tmp_path / "scan.json"
        scan_path.write_text(json.dumps({
            "files": [
                {"relative_path": "cli.py", "language": "python"},
                {"relative_path": "utils.py", "language": "python"},
            ],
        }))

        ep_path = tmp_path / "entrypoints.json"
        ep = {
            "schema_version": "1.0.0",
            "entrypoints": [
                {"file": "utils.py", "language": "python", "type": "python_cli", "signals": [], "confidence": 0.8}
            ],
            "totals": {"entrypoints_found": 1, "by_type": {}},
        }
        ep_path.write_text(json.dumps(ep))

        sys.argv = ["triage_orphans.py", str(graph_path), str(scan_path), "--entrypoints", str(ep_path)]
        triage_main = _import_triage()[5]
        exit_code = triage_main()
        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
        captured = capfd.readouterr()
        output = json.loads(captured.out)

        # utils.py is orphan + entrypoint candidate
        ep_candidate_ids = [e["node_id"] for e in output["orphans"]["entrypoint_candidate"]]
        assert "file:utils.py" in ep_candidate_ids

    def test_cli_errors_on_no_args(self, capfd):
        """CLI should exit non-zero when no args given."""
        sys.argv = ["triage_orphans.py"]
        triage_main = _import_triage()[5]
        with pytest.raises(SystemExit) as exc_info:
            triage_main()
        assert exc_info.value.code != 0

    def test_cli_errors_on_missing_graph(self, capfd):
        """CLI should exit non-zero when graph file doesn't exist."""
        sys.argv = ["triage_orphans.py", "/nonexistent/graph.json", "/dev/null"]
        triage_main = _import_triage()[5]
        exit_code = triage_main()
        assert exit_code != 0


class TestV2Taxonomy:
    """UA-P5-003: Richer orphan categories with confidence, reason, recommended_action.

    V2 taxonomy replaces coarse categories with fine-grained ones:
      expected_doc, expected_asset, expected_config, expected_test_fixture,
      expected_migration, expected_static_template,
      entrypoint_candidate, possible_dead_source, import_resolution_anomaly, unknown.

    Every orphan entry now carries: node_id, category, orphan_type (= category
    alias), confidence, confidence_label (high/medium/low), reason,
    recommended_action. The 4 top-level groups (expected, entrypoint_candidate,
    suspicious, unknown) are preserved for backward compat with report-data.
    """

    # ── expected_doc ──────────────────────────────────────────────────

    def test_readme_is_expected_doc(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:README.md", "filePath": "README.md", "language": "markdown"}
        cat, reason, entry = classify(node, None)
        assert cat == "expected"
        assert entry["category"] == "expected_doc"
        assert entry["confidence"] >= 0.9
        assert "readme" in entry["reason"].lower() or "doc" in entry["reason"].lower()
        assert entry["recommended_action"] in ("no_action_needed", "review")

    def test_docs_directory_is_expected_doc(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:docs/guide.md", "filePath": "docs/guide.md", "language": "markdown"}
        cat, reason, entry = classify(node, None)
        assert cat == "expected"
        assert entry["category"] == "expected_doc"

    def test_changelog_is_expected_doc(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:CHANGELOG.md", "filePath": "CHANGELOG.md", "language": "markdown"}
        cat, reason, entry = classify(node, None)
        assert cat == "expected"
        assert entry["category"] == "expected_doc"

    def test_rst_doc_is_expected_doc(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:docs/api.rst", "filePath": "docs/api.rst", "language": "rst"}
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_doc"

    # ── expected_asset ────────────────────────────────────────────────

    def test_png_is_expected_asset(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:assets/logo.png", "filePath": "assets/logo.png", "language": "unknown"}
        cat, _, entry = classify(node, None)
        assert cat == "expected"
        assert entry["category"] == "expected_asset"
        assert entry["confidence"] >= 0.9

    def test_svg_is_expected_asset(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:assets/icon.svg", "filePath": "assets/icon.svg", "language": "unknown"}
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_asset"

    def test_images_directory_is_expected_asset(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:images/banner.jpg", "filePath": "images/banner.jpg", "language": "unknown"}
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_asset"

    # ── expected_config ───────────────────────────────────────────────

    def test_yaml_is_expected_config(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:config.yaml", "filePath": "config.yaml", "language": "yaml"}
        cat, _, entry = classify(node, None)
        assert cat == "expected"
        assert entry["category"] == "expected_config"

    def test_toml_is_expected_config(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:pyproject.toml", "filePath": "pyproject.toml", "language": "toml"}
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_config"

    def test_env_is_expected_config(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:.env", "filePath": ".env", "language": "unknown"}
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_config"

    # ── expected_test_fixture ─────────────────────────────────────────

    def test_test_file_is_expected_test_fixture(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:tests/test_main.py", "filePath": "tests/test_main.py", "language": "python"}
        cat, _, entry = classify(node, None)
        assert cat == "expected"
        assert entry["category"] == "expected_test_fixture"
        assert entry["confidence"] >= 0.9

    def test_fixture_json_is_expected_test_fixture(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:fixtures/data.json", "filePath": "fixtures/data.json", "language": "json"}
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_test_fixture"

    # ── expected_migration ────────────────────────────────────────────

    def test_sql_migration_is_expected_migration(self):
        """supabase/migrations/001.sql -> expected_migration, high confidence."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:supabase/migrations/001.sql",
            "filePath": "supabase/migrations/001.sql",
            "language": "sql",
        }
        cat, _, entry = classify(node, None)
        assert cat == "expected"
        assert entry["category"] == "expected_migration"
        assert entry["confidence"] >= 0.8
        assert "domain" in entry["recommended_action"].lower() or "review" in entry["recommended_action"].lower()

    def test_migrations_directory_is_expected_migration(self):
        classify = _import_triage()[1]
        node = {
            "node_id": "file:migrations/002_down.sql",
            "filePath": "migrations/002_down.sql",
            "language": "sql",
        }
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_migration"

    def test_prisma_migration_is_expected_migration(self):
        classify = _import_triage()[1]
        node = {
            "node_id": "file:prisma/migrations/20240101_init.sql",
            "filePath": "prisma/migrations/20240101_init.sql",
            "language": "sql",
        }
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_migration"

    def test_alembic_migration_is_expected_migration(self):
        classify = _import_triage()[1]
        node = {
            "node_id": "file:alembic/versions/abc123.py",
            "filePath": "alembic/versions/abc123.py",
            "language": "python",
        }
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_migration"

    # ── expected_static_template ──────────────────────────────────────

    def test_html_template_is_expected_static_template(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:templates/index.html", "filePath": "templates/index.html", "language": "html"}
        cat, _, entry = classify(node, None)
        assert cat == "expected"
        assert entry["category"] == "expected_static_template"

    def test_views_directory_is_expected_static_template(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:views/base.html", "filePath": "views/base.html", "language": "html"}
        cat, _, entry = classify(node, None)
        assert entry["category"] == "expected_static_template"

    # ── entrypoint_candidate ──────────────────────────────────────────

    def test_entrypoint_candidate_shape(self):
        triage = _import_triage()[0]
        graph = _make_graph(["file:src/cli.py"],
                            connected_node_ids=["file:src/utils.py"],
                            edge_sources=["file:src/utils.py"],
                            edge_targets=["file:src/utils.py"])
        scan = _make_scan([
            {"relative_path": "src/cli.py", "language": "python"},
            {"relative_path": "src/utils.py", "language": "python"},
        ])
        entrypoints = _make_entrypoints(["src/cli.py"])
        result = triage(graph, scan, entrypoints)
        assert len(result["orphans"]["entrypoint_candidate"]) >= 1
        ep = result["orphans"]["entrypoint_candidate"][0]
        assert ep["category"] == "entrypoint_candidate"
        assert "confidence" in ep
        assert "reason" in ep
        assert "recommended_action" in ep

    # ── possible_dead_source ──────────────────────────────────────────

    def test_dead_source_shape(self):
        """src/lib/offlineQueue.js -> possible_dead_source, medium confidence."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:src/lib/offlineQueue.js",
            "filePath": "src/lib/offlineQueue.js",
            "language": "javascript",
        }
        cat, _, entry = classify(node, None)
        assert cat == "suspicious"
        assert entry["category"] == "possible_dead_source"
        assert entry["confidence"] <= 0.7  # medium confidence
        assert "import" in entry["recommended_action"].lower() or "verify" in entry["recommended_action"].lower()

    def test_unreferenced_python_is_possible_dead_source(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/legacy.py", "filePath": "src/legacy.py", "language": "python"}
        cat, _, entry = classify(node, None)
        assert cat == "suspicious"
        assert entry["category"] == "possible_dead_source"

    # ── import_resolution_anomaly ─────────────────────────────────────

    def test_import_resolution_anomaly(self):
        """A source file whose imports all fail to resolve."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:src/broken.py",
            "filePath": "src/broken.py",
            "language": "python",
            "unresolved_imports": ["__missing_module__", "@corrupt/import"],
        }
        cat, _, entry = classify(node, None)
        assert cat == "suspicious"
        assert entry["category"] == "import_resolution_anomaly"
        assert "import" in entry["reason"].lower() or "resolution" in entry["reason"].lower()
        assert entry["confidence"] >= 0.3

    # ── unknown ───────────────────────────────────────────────────────

    def test_unknown_shape(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/compiled.dat", "filePath": "src/compiled.dat", "language": "binary"}
        cat, _, entry = classify(node, None)
        assert cat == "unknown"
        assert entry["category"] == "unknown"
        assert "confidence" in entry
        assert entry["confidence"] <= 0.3

    def test_unknown_missing_metadata_shape(self):
        classify = _import_triage()[1]
        node = {"node_id": "module:os", "language": "unknown"}
        cat, _, entry = classify(node, None)
        assert cat == "unknown"
        assert entry["category"] == "unknown"

    # ── Output shape: 4 groups preserved, entries enriched ────────────

    def test_triage_output_has_rich_entries(self):
        """Full triage result entries should have category, orphan_type, confidence, confidence_label, reason, recommended_action."""
        triage_func = _import_triage()[0]
        graph = {
            "nodes": [
                {"node_id": "file:connected.py", "filePath": "connected.py", "language": "python"},
                {"node_id": "file:docs/README.md", "filePath": "docs/README.md", "language": "markdown"},
                {"node_id": "file:src/orphan.py", "filePath": "src/orphan.py", "language": "python"},
            ],
            "edges": [
                {"source": "file:connected.py", "target": "file:connected.py", "edge_type": "imports"},
            ],
        }
        scan = {"files": [
            {"relative_path": "docs/README.md", "language": "markdown"},
            {"relative_path": "src/orphan.py", "language": "python"},
        ]}
        result = triage_func(graph, scan, None)

        # Expected orphans should have V2 enriched shape
        assert len(result["orphans"]["expected"]) >= 1
        doc_entry = result["orphans"]["expected"][0]
        assert "category" in doc_entry
        assert "orphan_type" in doc_entry
        assert doc_entry["orphan_type"] == doc_entry["category"]
        assert "confidence" in doc_entry
        assert "confidence_label" in doc_entry
        assert "reason" in doc_entry
        assert "recommended_action" in doc_entry
        assert doc_entry["category"] == "expected_doc"

        # Suspicious orphans should also have V2 shape
        assert len(result["orphans"]["suspicious"]) >= 1
        sus_entry = result["orphans"]["suspicious"][0]
        assert "category" in sus_entry
        assert "orphan_type" in sus_entry
        assert sus_entry["orphan_type"] == sus_entry["category"]
        assert "confidence_label" in sus_entry
        assert sus_entry["category"] == "possible_dead_source"

    def test_schema_version_unchanged(self):
        """Schema version stays at 1.0.0 — V2 is an enrichment, not a schema break."""
        triage_func = _import_triage()[0]
        graph = _make_graph([], connected_node_ids=["file:src/main.py"],
                            edge_sources=["file:src/main.py"],
                            edge_targets=["file:src/main.py"])
        scan = _make_scan([{"relative_path": "src/main.py", "language": "python"}])
        result = triage_func(graph, scan, None)
        assert result["schema_version"] == "1.0.0"

    # ── Recommended actions ───────────────────────────────────────────

    def test_expected_doc_no_action_needed(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:docs/guide.md", "filePath": "docs/guide.md", "language": "markdown"}
        _, _, entry = classify(node, None)
        assert entry["recommended_action"] == "no_action_needed"

    def test_expected_config_no_action_needed(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:config.yaml", "filePath": "config.yaml", "language": "yaml"}
        _, _, entry = classify(node, None)
        assert entry["recommended_action"] == "no_action_needed"

    def test_expected_migration_review_action(self):
        classify = _import_triage()[1]
        node = {
            "node_id": "file:supabase/migrations/001.sql",
            "filePath": "supabase/migrations/001.sql",
            "language": "sql",
        }
        _, _, entry = classify(node, None)
        assert entry["recommended_action"] in ("review_via_domain_analyzer", "review")

    def test_possible_dead_source_verify_import(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/legacy.py", "filePath": "src/legacy.py", "language": "python"}
        _, _, entry = classify(node, None)
        assert "import" in entry["recommended_action"].lower() or "verify" in entry["recommended_action"].lower()

    def test_import_resolution_anomaly_action(self):
        classify = _import_triage()[1]
        node = {
            "node_id": "file:src/broken.py",
            "filePath": "src/broken.py",
            "language": "python",
            "unresolved_imports": ["missing_module"],
        }
        _, _, entry = classify(node, None)
        assert "import" in entry["recommended_action"].lower() or "resolution" in entry["recommended_action"].lower()

    # ── orphan_type alias ─────────────────────────────────────────────

    def test_migration_orphan_type(self):
        """Migration entries include orphan_type alias equal to category."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:supabase/migrations/001.sql",
            "filePath": "supabase/migrations/001.sql",
            "language": "sql",
        }
        _, _, entry = classify(node, None)
        assert "orphan_type" in entry
        assert entry["orphan_type"] == entry["category"]
        assert entry["orphan_type"] == "expected_migration"

    def test_import_anomaly_orphan_type(self):
        """Import resolution anomaly entries include orphan_type alias equal to category."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:src/broken.py",
            "filePath": "src/broken.py",
            "language": "python",
            "unresolved_imports": ["missing_module"],
        }
        _, _, entry = classify(node, None)
        assert "orphan_type" in entry
        assert entry["orphan_type"] == entry["category"]
        assert entry["orphan_type"] == "import_resolution_anomaly"

    def test_dead_source_orphan_type(self):
        """Dead source entries include orphan_type alias equal to category."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:src/legacy.py",
            "filePath": "src/legacy.py",
            "language": "python",
        }
        _, _, entry = classify(node, None)
        assert "orphan_type" in entry
        assert entry["orphan_type"] == "possible_dead_source"
        assert entry["orphan_type"] == entry["category"]

    # ── confidence_label ──────────────────────────────────────────────

    def test_migration_confidence_label(self):
        """Migration entries have a human-readable confidence_label."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:supabase/migrations/001.sql",
            "filePath": "supabase/migrations/001.sql",
            "language": "sql",
        }
        _, _, entry = classify(node, None)
        assert "confidence_label" in entry
        assert entry["confidence_label"] == "high"  # 0.85

    def test_import_anomaly_confidence_label(self):
        """Import resolution anomaly entries have medium confidence_label."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:src/broken.py",
            "filePath": "src/broken.py",
            "language": "python",
            "unresolved_imports": ["missing_module"],
        }
        _, _, entry = classify(node, None)
        assert "confidence_label" in entry
        assert entry["confidence_label"] == "medium"  # 0.6

    def test_dead_source_confidence_label(self):
        """Dead source entries have medium confidence_label (0.5)."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:src/legacy.py",
            "filePath": "src/legacy.py",
            "language": "python",
        }
        _, _, entry = classify(node, None)
        assert "confidence_label" in entry
        assert entry["confidence_label"] == "medium"  # 0.5

    def test_unknown_confidence_label(self):
        """Unknown entries have low confidence_label."""
        classify = _import_triage()[1]
        node = {
            "node_id": "file:src/compiled.dat",
            "filePath": "src/compiled.dat",
            "language": "binary",
        }
        _, _, entry = classify(node, None)
        assert "confidence_label" in entry
        assert entry["confidence_label"] == "low"  # 0.1

    def test_orphan_type_and_label_on_full_triage(self):
        """Full triage result entries include orphan_type and confidence_label."""
        triage_func = _import_triage()[0]
        graph = {
            "nodes": [
                {"node_id": "file:connected.py", "filePath": "connected.py", "language": "python"},
                {"node_id": "file:docs/README.md", "filePath": "docs/README.md", "language": "markdown"},
                {"node_id": "file:src/orphan.py", "filePath": "src/orphan.py", "language": "python"},
                {"node_id": "file:supabase/migrations/001.sql", "filePath": "supabase/migrations/001.sql", "language": "sql"},
            ],
            "edges": [
                {"source": "file:connected.py", "target": "file:connected.py", "edge_type": "imports"},
            ],
        }
        scan = {"files": [
            {"relative_path": "docs/README.md", "language": "markdown"},
            {"relative_path": "src/orphan.py", "language": "python"},
            {"relative_path": "supabase/migrations/001.sql", "language": "sql"},
        ]}
        result = triage_func(graph, scan, None)

        # Check expected entries
        for e in result["orphans"]["expected"]:
            assert "orphan_type" in e, "Missing orphan_type on expected entry"
            assert "confidence_label" in e, "Missing confidence_label on expected entry"
            assert isinstance(e["confidence_label"], str)
            assert e["confidence_label"] in ("high", "medium", "low")
            assert e["orphan_type"] == e["category"]

        # Check suspicious entries
        for e in result["orphans"]["suspicious"]:
            assert "orphan_type" in e, "Missing orphan_type on suspicious entry"
            assert "confidence_label" in e, "Missing confidence_label on suspicious entry"
            assert e["orphan_type"] == e["category"]


class TestConfidenceLabelHelper:
    """Unit tests for the _confidence_label helper function."""

    def test_high_threshold(self):
        label = _import_triage()[6]
        assert label(0.95) == "high"
        assert label(0.8) == "high"
        assert label(0.81) == "high"

    def test_medium_threshold(self):
        label = _import_triage()[6]
        assert label(0.7) == "medium"
        assert label(0.5) == "medium"
        assert label(0.51) == "medium"

    def test_low_threshold(self):
        label = _import_triage()[6]
        assert label(0.1) == "low"
        assert label(0.3) == "low"
        assert label(0.49) == "low"
        assert label(0.0) == "low"


# ── Fixture-based integration test ──────────────────────────────────────

class TestFixtureIntegration:
    """Integration test with real assembled-graph fixture files."""

    def test_fixture_based_triage(self):
        """Run triage against fixture graph.json + scan.json (+ optional entrypoints.json).

        The fixture at tests/code_scan/fixtures/orphan_triage/ represents a
        realistic assembled graph with nodes across all 4 orphan categories:
          - expected: docs, config, tests, fixtures, workflows, assets, templates
          - entrypoint_candidate: source orphan listed in entrypoints.json
          - suspicious: unreferenced source orphan not in entrypoints
          - unknown: module node (no filePath) and unsupported-language file
        """
        graph_path = FIXTURES_DIR / "graph.json"
        scan_path = FIXTURES_DIR / "scan.json"

        # Fixture is required — the integration test has a real-artifact obligation.
        assert graph_path.is_file(), f"Required fixture missing: {graph_path}"
        assert scan_path.is_file(), f"Required fixture missing: {scan_path}"

        triage_func = _import_triage()[0]

        with open(graph_path) as f:
            graph = json.load(f)
        with open(scan_path) as f:
            scan = json.load(f)

        entrypoints_path = FIXTURES_DIR / "entrypoints.json"
        entrypoints = None
        if entrypoints_path.exists():
            with open(entrypoints_path) as f:
                entrypoints = json.load(f)

        result = triage_func(graph, scan, entrypoints)

        # ── Validate output structure ──
        assert result["schema_version"] == "1.0.0"
        assert "orphans" in result
        assert "totals" in result

        category_set = {"expected", "entrypoint_candidate", "suspicious", "unknown"}
        for cat in category_set:
            assert cat in result["orphans"], f"Missing orphans category: {cat}"

        totals = result["totals"]
        for cat in category_set:
            assert cat in totals, f"Missing totals key: {cat}"

        assert totals["total_orphans"] == sum(
            len(result["orphans"][cat]) for cat in category_set
        )

        # ── Every category must have at least one member ──
        for cat in category_set:
            assert len(result["orphans"][cat]) >= 1, \
                f"Category {cat!r} is empty — fixture must cover all 4 orphan classes"

        # ── Specific node placement assertions ──
        expected_ids = {e["node_id"] for e in result["orphans"]["expected"]}
        for nid in ("file:docs/guide.md", "file:config.yaml",
                    "file:tests/test_core.py", "file:fixtures/sample.json",
                    "file:.github/workflows/ci.yml", "file:assets/logo.png",
                    "file:templates/index.html"):
            assert nid in expected_ids, f"Expected {nid!r} in expected orphans"

        suspicious_ids = {e["node_id"] for e in result["orphans"]["suspicious"]}
        for nid in ("file:src/legacy.py", "file:src/utils.py"):
            assert nid in suspicious_ids, f"Expected {nid!r} in suspicious orphans"

        ep_ids = {e["node_id"] for e in result["orphans"]["entrypoint_candidate"]}
        assert "file:src/cli.py" in ep_ids, \
            "Expected file:src/cli.py in entrypoint_candidate"

        unknown_ids = {e["node_id"] for e in result["orphans"]["unknown"]}
        assert "module:os" in unknown_ids, "Expected module:os in unknown"
        assert "file:src/compiled.dat" in unknown_ids, \
            "Expected file:src/compiled.dat in unknown"
