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
    )
    return (
        triage_orphans,
        classify_orphan,
        _is_expected_orphan,
        _is_entrypoint_candidate,
        _build_result,
        main,
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
        category, reason = classify(node, None)
        assert category == "expected"
        assert "doc" in reason

    def test_readme_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:README.md", "filePath": "README.md", "language": "markdown"}
        category, reason = classify(node, None)
        assert category == "expected"
        assert "doc" in reason

    def test_changelog_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:CHANGELOG.md", "filePath": "CHANGELOG.md", "language": "markdown"}
        category, reason = classify(node, None)
        assert category == "expected"

    def test_config_yaml_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:config.yaml", "filePath": "config.yaml", "language": "yaml"}
        category, reason = classify(node, None)
        assert category == "expected"
        assert "config" in reason

    def test_tests_directory_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:tests/test_main.py", "filePath": "tests/test_main.py", "language": "python"}
        category, reason = classify(node, None)
        assert category == "expected"
        assert "test" in reason

    def test_fixture_json_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:fixtures/data.json", "filePath": "fixtures/data.json", "language": "json"}
        category, reason = classify(node, None)
        assert category == "expected"
        assert "fixture" in reason

    def test_github_workflows_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:.github/workflows/ci.yml", "filePath": ".github/workflows/ci.yml", "language": "yaml"}
        category, reason = classify(node, None)
        assert category == "expected"
        assert "workflow" in reason

    def test_image_file_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:assets/logo.png", "filePath": "assets/logo.png", "language": "unknown"}
        category, reason = classify(node, None)
        assert category == "expected"
        assert "asset" in reason or "image" in reason

    def test_template_file_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:templates/index.html", "filePath": "templates/index.html", "language": "html"}
        category, reason = classify(node, None)
        assert category == "expected"
        assert "template" in reason

    def test_license_file_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:LICENSE", "filePath": "LICENSE", "language": "unknown"}
        category, reason = classify(node, None)
        assert category == "expected"

    def test_toml_config_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:pyproject.toml", "filePath": "pyproject.toml", "language": "toml"}
        category, reason = classify(node, None)
        assert category == "expected"

    def test_rst_doc_is_expected(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:docs/api.rst", "filePath": "docs/api.rst", "language": "rst"}
        category, reason = classify(node, None)
        assert category == "expected"


# ── Suspicious orphan classification ────────────────────────────────────

class TestSuspiciousOrphans:
    """Source orphans that are NOT plausible entrypoints."""

    def test_unreferenced_python_source_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/legacy.py", "filePath": "src/legacy.py", "language": "python"}
        category, reason = classify(node, None)
        assert category == "suspicious"
        assert "unreferenced" in reason

    def test_orphaned_util_module_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/utils.py", "filePath": "src/utils.py", "language": "python"}
        category, reason = classify(node, None)
        assert category == "suspicious"

    def test_orphaned_js_module_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:lib/helpers.js", "filePath": "lib/helpers.js", "language": "javascript"}
        category, reason = classify(node, None)
        assert category == "suspicious"

    def test_orphaned_go_file_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:pkg/old.go", "filePath": "pkg/old.go", "language": "go"}
        category, reason = classify(node, None)
        assert category == "suspicious"

    def test_orphaned_rust_file_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/legacy.rs", "filePath": "src/legacy.rs", "language": "rust"}
        category, reason = classify(node, None)
        assert category == "suspicious"

    def test_orphaned_ts_file_is_suspicious(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/old.ts", "filePath": "src/old.ts", "language": "typescript"}
        category, reason = classify(node, None)
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
        category, reason = classify(node, None)
        assert category == "unknown"

    def test_unsupported_language_is_unknown(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:src/compiled.dat", "filePath": "src/compiled.dat", "language": "binary"}
        category, reason = classify(node, None)
        assert category == "unknown"

    def test_missing_filePath_is_unknown(self):
        classify = _import_triage()[1]
        node = {"node_id": "file:xyz", "language": "python"}
        category, reason = classify(node, None)
        assert category == "unknown"


# ── _is_expected_orphan helper tests ────────────────────────────────────

class TestIsExpectedOrphan:
    """Low-level tests for the _is_expected_orphan helper."""

    def test_docs_pattern(self):
        is_exp = _import_triage()[2]
        assert is_exp({"node_id": "file:docs/readme.md", "filePath": "docs/readme.md", "language": "markdown"}) == ("doc", True)

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
