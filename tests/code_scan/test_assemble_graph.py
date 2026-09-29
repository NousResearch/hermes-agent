"""Tests for scripts/code-scan/assemble_graph.py — Phase 3 D3."""
import json
import os
import subprocess
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSEMBLE_SCRIPT = PROJECT_ROOT / "scripts" / "code-scan" / "assemble_graph.py"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "code_scan" / "fixtures" / "graph-batch"


def _import_assemble_graph():
    """Import assemble_graph module with sys.path pointing to scripts/code-scan."""
    script_dir = str(PROJECT_ROOT / "scripts" / "code-scan")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from assemble_graph import (
        load_batch_inputs,
        normalize_node_id,
        build_nodes_from_scan,
        build_nodes_from_imports,
        build_edges_from_imports,
        deduplicate_nodes,
        deduplicate_edges,
        merge_nodes,
        merge_edges,
        assemble_graph,
        count_orphans,
        build_summary,
        main,
    )
    return (
        load_batch_inputs,
        normalize_node_id,
        build_nodes_from_scan,
        build_nodes_from_imports,
        build_edges_from_imports,
        deduplicate_nodes,
        deduplicate_edges,
        merge_nodes,
        merge_edges,
        assemble_graph,
        count_orphans,
        build_summary,
        main,
    )


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def funcs():
    return _import_assemble_graph()


@pytest.fixture
def load_batch_inputs(funcs):
    return funcs[0]


@pytest.fixture
def normalize_node_id(funcs):
    return funcs[1]


@pytest.fixture
def build_nodes_from_scan(funcs):
    return funcs[2]


@pytest.fixture
def build_nodes_from_imports(funcs):
    return funcs[3]


@pytest.fixture
def build_edges_from_imports(funcs):
    return funcs[4]


@pytest.fixture
def deduplicate_nodes(funcs):
    return funcs[5]


@pytest.fixture
def deduplicate_edges(funcs):
    return funcs[6]


@pytest.fixture
def merge_nodes(funcs):
    return funcs[7]


@pytest.fixture
def merge_edges(funcs):
    return funcs[8]


@pytest.fixture
def assemble_graph(funcs):
    return funcs[9]


@pytest.fixture
def count_orphans(funcs):
    return funcs[10]


@pytest.fixture
def build_summary(funcs):
    return funcs[11]


@pytest.fixture
def main(funcs):
    return funcs[12]


# ── load_batch_inputs ───────────────────────────────────────────────────────

class TestLoadBatchInputs:
    def test_load_single_valid_file(self, load_batch_inputs):
        result = load_batch_inputs([str(FIXTURES_DIR / "batch1.json")])
        assert isinstance(result, list)
        assert len(result) == 1
        assert "files" in result[0]

    def test_load_multiple_valid_files(self, load_batch_inputs):
        result = load_batch_inputs([
            str(FIXTURES_DIR / "batch1.json"),
            str(FIXTURES_DIR / "batch2.json"),
        ])
        assert len(result) == 2

    def test_load_mixed_scan_and_imports(self, load_batch_inputs):
        result = load_batch_inputs([
            str(FIXTURES_DIR / "batch1.json"),
            str(FIXTURES_DIR / "imports1.json"),
        ])
        assert len(result) == 2

    def test_load_missing_file_raises(self, load_batch_inputs):
        with pytest.raises(ValueError):
            load_batch_inputs(["/no/such/file.json"])

    def test_load_invalid_json_raises(self, load_batch_inputs):
        with pytest.raises(ValueError):
            load_batch_inputs([str(FIXTURES_DIR / "bad_schema.json")])

    def test_load_empty_list(self, load_batch_inputs):
        result = load_batch_inputs([])
        assert result == []


# ── normalize_node_id ───────────────────────────────────────────────────────

class TestNormalizeNodeId:
    def test_file_prefix(self, normalize_node_id):
        assert normalize_node_id("file", "src/main.py") == "file:src/main.py"

    def test_module_prefix(self, normalize_node_id):
        assert normalize_node_id("module", "os") == "module:os"

    def test_func_prefix(self, normalize_node_id):
        result = normalize_node_id("func", "src/main.py:main")
        assert result == "func:src/main.py:main"

    def test_class_prefix(self, normalize_node_id):
        result = normalize_node_id("class", "src/main.py:ConfigParser")
        assert result == "class:src/main.py:ConfigParser"

    def test_unknown_prefix(self, normalize_node_id):
        assert normalize_node_id("bogus", "raw_id") == "unknown:raw_id"

    def test_lowercase_type(self, normalize_node_id):
        assert normalize_node_id("FILE", "src/main.py") == "file:src/main.py"
        assert normalize_node_id("MODULE", "os") == "module:os"

    def test_forward_slash_normalization(self, normalize_node_id):
        # Windows-style backslashes normalized to forward slashes
        result = normalize_node_id("file", "src\\main.py")
        assert result == "file:src/main.py"


# ── build_nodes_from_scan ───────────────────────────────────────────────────

class TestBuildNodesFromScan:
    def test_basic_file_nodes(self, build_nodes_from_scan):
        scan_data = {"files": [
            {"relative_path": "src/main.py", "path": "/tmp/src/main.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200},
        ]}
        nodes = build_nodes_from_scan(scan_data)
        assert len(nodes) == 1
        node = nodes[0]
        assert node["node_id"] == "file:src/main.py"
        assert node["node_type"] == "file"
        assert node["filePath"] == "src/main.py"
        assert node["label"] == "main.py"
        assert node["language"] == "python"

    def test_multiple_files(self, build_nodes_from_scan):
        scan_data = {"files": [
            {"relative_path": "a.py", "path": "/tmp/a.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200},
            {"relative_path": "b.py", "path": "/tmp/b.py",
             "language": "python", "category": "code",
             "lines": 20, "size_bytes": 400},
        ]}
        nodes = build_nodes_from_scan(scan_data)
        assert len(nodes) == 2
        ids = [n["node_id"] for n in nodes]
        assert "file:a.py" in ids
        assert "file:b.py" in ids

    def test_functions_included(self, build_nodes_from_scan):
        scan_data = {"files": [
            {"relative_path": "src/main.py", "path": "/tmp/src/main.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200,
             "functions": ["main", "helper"]},
        ]}
        nodes = build_nodes_from_scan(scan_data)
        assert "functions" in nodes[0]
        assert nodes[0]["functions"] == ["main", "helper"]

    def test_classes_included(self, build_nodes_from_scan):
        scan_data = {"files": [
            {"relative_path": "src/models.py", "path": "/tmp/src/models.py",
             "language": "python", "category": "code",
             "lines": 50, "size_bytes": 300,
             "classes": ["DataModel"]},
        ]}
        nodes = build_nodes_from_scan(scan_data)
        assert "classes" in nodes[0]
        assert nodes[0]["classes"] == ["DataModel"]

    def test_real_fixture(self, build_nodes_from_scan):
        scan_data = json.loads((FIXTURES_DIR / "batch1.json").read_text())
        nodes = build_nodes_from_scan(scan_data)
        assert len(nodes) == 5
        node_ids = [n["node_id"] for n in nodes]
        assert "file:src/main.py" in node_ids
        assert "file:src/utils.py" in node_ids
        assert "file:tests/test_main.py" in node_ids


# ── build_nodes_from_imports ────────────────────────────────────────────────

class TestBuildNodesFromImports:
    def test_unique_module_nodes(self, build_nodes_from_imports):
        import_data = {"files": {
            "a.py": {"imports": ["os", "sys"], "warnings": []},
            "b.py": {"imports": ["os", "json"], "warnings": []},
        }}
        nodes = build_nodes_from_imports(import_data)
        ids = [n["node_id"] for n in nodes]
        assert "module:os" in ids
        assert "module:sys" in ids
        assert "module:json" in ids
        # os appears in both files but should be deduplicated
        assert ids.count("module:os") == 1

    def test_module_node_schema(self, build_nodes_from_imports):
        import_data = {"files": {
            "a.py": {"imports": ["os"], "warnings": []},
        }}
        nodes = build_nodes_from_imports(import_data)
        assert len(nodes) == 1
        node = nodes[0]
        assert node["node_type"] == "module"
        assert node["filePath"] is None
        assert node["label"] == "os"

    def test_empty_imports(self, build_nodes_from_imports):
        import_data = {"files": {
            "a.py": {"imports": [], "warnings": []},
        }}
        nodes = build_nodes_from_imports(import_data)
        assert nodes == []

    def test_real_fixture(self, build_nodes_from_imports):
        import_data = json.loads((FIXTURES_DIR / "imports1.json").read_text())
        nodes = build_nodes_from_imports(import_data)
        # imports1 has 8 unique modules: os, sys, json, configparser,
        # datetime, typing, dataclasses, unittest, src.main, src.utils
        assert len(nodes) > 0
        ids = [n["node_id"] for n in nodes]
        assert "module:os" in ids


# ── build_edges_from_imports ────────────────────────────────────────────────

class TestBuildEdgesFromImports:
    def test_basic_edges(self, build_edges_from_imports):
        import_data = {"files": {
            "a.py": {"imports": ["os", "sys"], "warnings": []},
        }}
        edges = build_edges_from_imports(import_data)
        assert len(edges) == 2
        assert edges[0]["source"] == "file:a.py"
        assert edges[0]["target"] == "module:os"
        assert edges[0]["edge_type"] == "imports"
        assert edges[0]["meta"] == {}

    def test_multiple_files(self, build_edges_from_imports):
        import_data = {"files": {
            "a.py": {"imports": ["os"], "warnings": []},
            "b.py": {"imports": ["json"], "warnings": []},
        }}
        edges = build_edges_from_imports(import_data)
        assert len(edges) == 2
        sources = [e["source"] for e in edges]
        assert "file:a.py" in sources
        assert "file:b.py" in sources

    def test_edge_meta_empty(self, build_edges_from_imports):
        edges = build_edges_from_imports({"files": {
            "x.py": {"imports": ["os"], "warnings": []},
        }})
        assert edges[0]["meta"] == {}

    def test_real_fixture(self, build_edges_from_imports):
        import_data = json.loads((FIXTURES_DIR / "imports1.json").read_text())
        edges = build_edges_from_imports(import_data)
        assert len(edges) > 0
        targets = [e["target"] for e in edges]
        assert all(t.startswith("module:") for t in targets)


# ── deduplicate_nodes ───────────────────────────────────────────────────────

class TestDeduplicateNodes:
    def test_no_duplicates(self, deduplicate_nodes):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py"},
            {"node_id": "file:b.py", "node_type": "file",
             "filePath": "b.py", "label": "b.py"},
        ]
        deduped, count = deduplicate_nodes(nodes)
        assert len(deduped) == 2
        assert count == 0

    def test_duplicates_merged(self, deduplicate_nodes):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py",
             "functions": ["func1"]},
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py",
             "functions": ["func2"]},
        ]
        deduped, count = deduplicate_nodes(nodes)
        assert len(deduped) == 1
        assert count == 1
        assert deduped[0]["functions"] == ["func1", "func2"]

    def test_functions_sorted_deduped(self, deduplicate_nodes):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py",
             "functions": ["zebra", "alpha"]},
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py",
             "functions": ["alpha", "beta"]},
        ]
        deduped, count = deduplicate_nodes(nodes)
        assert deduped[0]["functions"] == ["alpha", "beta", "zebra"]

    def test_classes_merged(self, deduplicate_nodes):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py",
             "classes": ["ClassA"]},
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py",
             "classes": ["ClassB"]},
        ]
        deduped, count = deduplicate_nodes(nodes)
        assert deduped[0]["classes"] == ["ClassA", "ClassB"]

    def test_imports_merged(self, deduplicate_nodes):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py",
             "imports": ["os"]},
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py",
             "imports": ["sys"]},
        ]
        deduped, count = deduplicate_nodes(nodes)
        assert deduped[0]["imports"] == ["os", "sys"]

    def test_scalar_fields_keep_first(self, deduplicate_nodes):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py", "language": "python"},
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py", "language": "python3"},
        ]
        deduped, count = deduplicate_nodes(nodes)
        assert deduped[0]["language"] == "python"

    def test_empty_list(self, deduplicate_nodes):
        deduped, count = deduplicate_nodes([])
        assert deduped == []
        assert count == 0


# ── deduplicate_edges ───────────────────────────────────────────────────────

class TestDeduplicateEdges:
    def test_no_duplicates(self, deduplicate_edges):
        edges = [
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "imports", "meta": {}},
            {"source": "file:a.py", "target": "module:sys",
             "edge_type": "imports", "meta": {}},
        ]
        deduped, count = deduplicate_edges(edges)
        assert len(deduped) == 2
        assert count == 0

    def test_duplicates_removed(self, deduplicate_edges):
        edges = [
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "imports", "meta": {}},
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "imports", "meta": {}},
        ]
        deduped, count = deduplicate_edges(edges)
        assert len(deduped) == 1
        assert count == 1

    def test_keep_first_meta(self, deduplicate_edges):
        edges = [
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "imports", "meta": {"version": "1.0"}},
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "imports", "meta": {"version": "2.0"}},
        ]
        deduped, count = deduplicate_edges(edges)
        assert deduped[0]["meta"] == {"version": "1.0"}

    def test_different_edge_types_not_duplicates(self, deduplicate_edges):
        edges = [
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "imports", "meta": {}},
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "contains", "meta": {}},
        ]
        deduped, count = deduplicate_edges(edges)
        assert len(deduped) == 2
        assert count == 0

    def test_empty_list(self, deduplicate_edges):
        deduped, count = deduplicate_edges([])
        assert deduped == []
        assert count == 0


# ── merge_nodes ─────────────────────────────────────────────────────────────

class TestMergeNodes:
    def test_merge_two_batches(self, merge_nodes):
        batch1 = [{"node_id": "file:a.py", "node_type": "file",
                   "filePath": "a.py", "label": "a.py"}]
        batch2 = [{"node_id": "file:b.py", "node_type": "file",
                   "filePath": "b.py", "label": "b.py"}]
        merged = merge_nodes([batch1, batch2])
        assert len(merged) == 2

    def test_merge_overlapping(self, merge_nodes):
        batch1 = [{"node_id": "file:a.py", "node_type": "file",
                   "filePath": "a.py", "label": "a.py",
                   "functions": ["func1"]}]
        batch2 = [{"node_id": "file:a.py", "node_type": "file",
                   "filePath": "a.py", "label": "a.py",
                   "functions": ["func2"]}]
        merged = merge_nodes([batch1, batch2])
        assert len(merged) == 1
        assert merged[0]["functions"] == ["func1", "func2"]

    def test_merge_empty_batches(self, merge_nodes):
        merged = merge_nodes([[], []])
        assert merged == []


# ── merge_edges ─────────────────────────────────────────────────────────────

class TestMergeEdges:
    def test_merge_two_batches(self, merge_edges):
        batch1 = [{"source": "file:a.py", "target": "module:os",
                   "edge_type": "imports", "meta": {}}]
        batch2 = [{"source": "file:b.py", "target": "module:json",
                   "edge_type": "imports", "meta": {}}]
        merged = merge_edges([batch1, batch2])
        assert len(merged) == 2

    def test_merge_overlapping(self, merge_edges):
        batch1 = [{"source": "file:a.py", "target": "module:os",
                   "edge_type": "imports", "meta": {}}]
        batch2 = [{"source": "file:a.py", "target": "module:os",
                   "edge_type": "imports", "meta": {}}]
        merged = merge_edges([batch1, batch2])
        assert len(merged) == 1

    def test_merge_empty_batches(self, merge_edges):
        merged = merge_edges([[], []])
        assert merged == []


# ── count_orphans ───────────────────────────────────────────────────────────

class TestCountOrphans:
    def test_all_referenced(self, count_orphans):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py"},
            {"node_id": "module:os", "node_type": "module",
             "filePath": None, "label": "os"},
        ]
        edges = [
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "imports", "meta": {}},
        ]
        assert count_orphans(nodes, edges) == 0

    def test_one_orphan(self, count_orphans):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py"},
            {"node_id": "module:os", "node_type": "module",
             "filePath": None, "label": "os"},
            {"node_id": "module:sys", "node_type": "module",
             "filePath": None, "label": "sys"},
        ]
        edges = [
            {"source": "file:a.py", "target": "module:os",
             "edge_type": "imports", "meta": {}},
        ]
        assert count_orphans(nodes, edges) == 1

    def test_empty_nodes(self, count_orphans):
        assert count_orphans([], []) == 0

    def test_no_edges_all_orphans(self, count_orphans):
        nodes = [
            {"node_id": "file:a.py", "node_type": "file",
             "filePath": "a.py", "label": "a.py"},
            {"node_id": "file:b.py", "node_type": "file",
             "filePath": "b.py", "label": "b.py"},
        ]
        assert count_orphans(nodes, []) == 2


# ── build_summary ───────────────────────────────────────────────────────────

class TestBuildSummary:
    def test_basic_summary(self, build_summary):
        nodes = [{"node_id": "a"}, {"node_id": "b"}]
        edges = [{"source": "a", "target": "b", "edge_type": "imports", "meta": {}}]
        summary = build_summary(nodes, edges, 1, 2, 3)
        assert summary["total_nodes"] == 2
        assert summary["total_edges"] == 1
        assert summary["deduplicated_nodes"] == 1
        assert summary["deduplicated_edges"] == 2
        assert summary["orphan_nodes"] == 3

    def test_summary_keys(self, build_summary):
        summary = build_summary([], [], 0, 0, 0)
        expected_keys = {"total_nodes", "total_edges", "deduplicated_nodes",
                         "deduplicated_edges", "orphan_nodes"}
        assert set(summary.keys()) == expected_keys


# ── assemble_graph ──────────────────────────────────────────────────────────

class TestAssembleGraph:
    def test_with_scan_only(self, assemble_graph):
        scans = [{"files": [
            {"relative_path": "a.py", "path": "/tmp/a.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200},
        ]}]
        graph = assemble_graph(scans, None)
        assert graph["schema_version"] == "1.0.0"
        assert "generated_at" in graph
        assert "nodes" in graph
        assert "edges" in graph
        assert "summary" in graph
        assert graph["summary"]["total_nodes"] > 0

    def test_with_scan_and_imports(self, assemble_graph):
        scans = [{"files": [
            {"relative_path": "a.py", "path": "/tmp/a.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200},
        ]}]
        imports_list = [{"files": {
            "a.py": {"imports": ["os", "sys"], "warnings": []},
        }}]
        graph = assemble_graph(scans, imports_list)
        assert graph["schema_version"] == "1.0.0"
        assert graph["summary"]["total_nodes"] > 0
        assert graph["summary"]["total_edges"] > 0

    def test_deduplication_counts(self, assemble_graph):
        # Two scans with overlapping file
        scans = [
            {"files": [
                {"relative_path": "a.py", "path": "/tmp/a.py",
                 "language": "python", "category": "code",
                 "lines": 10, "size_bytes": 200, "functions": ["func1"]},
            ]},
            {"files": [
                {"relative_path": "a.py", "path": "/tmp/a.py",
                 "language": "python", "category": "code",
                 "lines": 10, "size_bytes": 200, "functions": ["func2"]},
            ]},
        ]
        graph = assemble_graph(scans, None)
        assert graph["summary"]["deduplicated_nodes"] >= 1

    def test_full_fixture_pipeline(self, assemble_graph):
        scan1 = json.loads((FIXTURES_DIR / "batch1.json").read_text())
        scan2 = json.loads((FIXTURES_DIR / "batch2.json").read_text())
        imp1 = json.loads((FIXTURES_DIR / "imports1.json").read_text())
        imp2 = json.loads((FIXTURES_DIR / "imports2.json").read_text())
        graph = assemble_graph([scan1, scan2], [imp1, imp2])
        assert graph["schema_version"] == "1.0.0"
        assert "generated_at" in graph
        assert "source_files" in graph
        assert "nodes" in graph
        assert "edges" in graph
        assert "summary" in graph
        # summary should have all required keys
        for key in ("total_nodes", "total_edges", "deduplicated_nodes",
                     "deduplicated_edges", "orphan_nodes"):
            assert key in graph["summary"], f"Missing summary key: {key}"
        # Should have deduplicated some nodes (overlapping files)
        assert graph["summary"]["deduplicated_nodes"] > 0

    def test_graph_schema_validation(self, assemble_graph):
        scans = [{"files": [
            {"relative_path": "a.py", "path": "/tmp/a.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200},
        ]}]
        imports_list = [{"files": {
            "a.py": {"imports": ["os"], "warnings": []},
        }}]
        graph = assemble_graph(scans, imports_list)
        # The graph should be valid for graph_schema
        assert graph["schema_version"] == "1.0.0"
        # Node types should be valid (file, function, class, module)
        for node in graph["nodes"]:
            assert node["node_type"] in ("file", "module", "function", "class")

    # ── Import-map key canonicalization (integration blocker) ─────────────

    def test_import_abs_path_key_canonicalized_to_relative(self, assemble_graph):
        """When import_data files key is an absolute path, edge source must
        use the scan's relative_path, not the absolute key.

        Repro: extract_imports.py emits keys using entry['path'] (absolute),
        while build_nodes_from_scan uses relative_path. Without canonicalization
        this produces edge source 'file:/tmp/proj/src/main.py' but node
        'file:src/main.py' → graph_schema source mismatch.
        """
        scans = [{"files": [
            {"path": "/tmp/proj/src/main.py", "relative_path": "src/main.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200},
        ]}]
        # Simulate real extract_imports output: key = absolute path
        imports_list = [{"files": {
            "/tmp/proj/src/main.py": {"imports": ["os"], "warnings": []},
        }}]
        graph = assemble_graph(scans, imports_list)

        # Edge source must be file:src/main.py (match scan node), not
        # file:/tmp/proj/src/main.py
        edge_sources = [e["source"] for e in graph["edges"] if e["edge_type"] == "imports"]
        assert "file:src/main.py" in edge_sources, (
            f"Expected edge source 'file:src/main.py', got sources: {edge_sources}"
        )
        assert "file:/tmp/proj/src/main.py" not in edge_sources

    def test_import_abs_path_no_schema_mismatch(self, assemble_graph):
        """Assembled graph for absolute-path import keys must have ZERO
        graph_schema issues about edge source not matching any node.
        """
        scans = [{"files": [
            {"path": "/tmp/proj/src/main.py", "relative_path": "src/main.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200},
            {"path": "/tmp/proj/src/utils.py", "relative_path": "src/utils.py",
             "language": "python", "category": "code",
             "lines": 20, "size_bytes": 400},
        ]}]
        imports_list = [{"files": {
            "/tmp/proj/src/main.py": {"imports": ["os", "src.utils"], "warnings": []},
            "/tmp/proj/src/utils.py": {"imports": ["json"], "warnings": []},
        }}]
        graph = assemble_graph(scans, imports_list)

        # No issues — especially no "Edge source … does not match any known node"
        from graph_schema import validate_graph
        validation = validate_graph(graph)
        source_mismatches = [
            i for i in validation["issues"]
            if "Edge source" in i and "does not match" in i
        ]
        assert source_mismatches == [], (
            f"Source-mismatch issues found: {source_mismatches}"
        )

    def test_import_key_canonicalization_preserves_unmatched(self, assemble_graph):
        """If an import map key has no matching scan record, the key should
        be preserved (normalized with forward slashes) rather than dropped.
        """
        scans = [{"files": [
            {"path": "/tmp/proj/src/main.py", "relative_path": "src/main.py",
             "language": "python", "category": "code",
             "lines": 10, "size_bytes": 200},
        ]}]
        # Key doesn't match any scan file — should still produce edges
        imports_list = [{"files": {
            "/tmp/other/external.py": {"imports": ["os"], "warnings": []},
        }}]
        graph = assemble_graph(scans, imports_list)

        # Should still have an edge from the unmatched file
        edge_sources = [e["source"] for e in graph["edges"] if e["edge_type"] == "imports"]
        assert len(edge_sources) == 1
        # Normalized key with forward slashes (unchanged absolute path)
        assert edge_sources[0] == "file:/tmp/other/external.py"


# ── main() ──────────────────────────────────────────────────────────────────

class TestMain:
    def test_main_with_scan_file(self, main, tmp_path):
        output_file = tmp_path / "graph.json"
        with patch.object(sys, "argv", [
            "assemble_graph.py",
            str(FIXTURES_DIR / "batch1.json"),
            "--output", str(output_file),
        ]):
            rc = main()
        assert rc == 0
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["schema_version"] == "1.0.0"

    def test_main_with_multiple_inputs(self, main, tmp_path):
        output_file = tmp_path / "graph.json"
        with patch.object(sys, "argv", [
            "assemble_graph.py",
            str(FIXTURES_DIR / "batch1.json"),
            str(FIXTURES_DIR / "imports1.json"),
            "--output", str(output_file),
        ]):
            rc = main()
        assert rc == 0

    def test_main_missing_input_raises(self, main):
        with patch.object(sys, "argv", [
            "assemble_graph.py", "/no/such/file.json",
        ]):
            rc = main()
        assert rc != 0

    def test_main_no_args(self, main):
        with patch.object(sys, "argv", ["assemble_graph.py"]):
            rc = main()
        assert rc != 0

    def test_main_verbose(self, main, tmp_path):
        output_file = tmp_path / "graph.json"
        with patch.object(sys, "argv", [
            "assemble_graph.py",
            str(FIXTURES_DIR / "batch1.json"),
            "--output", str(output_file),
            "--verbose",
        ]):
            rc = main()
        assert rc == 0

    def test_main_e2e_abs_path_import_keys_validate_issue_free(self, main, tmp_path):
        """CLI E2E: scan fixture with absolute-path import map key should
        produce a graph with zero edge-source-mismatch issues.

        Repro for the integration blocker: when extract_imports.py uses
        absolute paths as file keys, edge sources must be canonicalized to
        match scan node IDs, otherwise graph_schema reports source mismatches.
        """
        output_file = tmp_path / "graph.json"
        with patch.object(sys, "argv", [
            "assemble_graph.py",
            str(FIXTURES_DIR / "scan_abs_paths.json"),
            str(FIXTURES_DIR / "imports_abs_paths.json"),
            "--output", str(output_file),
        ]):
            rc = main()
        assert rc == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        # Build node set and check edges
        node_ids = {n["node_id"] for n in data["nodes"]}
        source_mismatches = []
        for edge in data["edges"]:
            if edge["source"] not in node_ids:
                source_mismatches.append(
                    f"Edge source '{edge['source']}' not in nodes"
                )
        assert source_mismatches == [], (
            f"E2E graph has source mismatches: {source_mismatches}"
        )

        # Verify at least some edges use relative-path node IDs
        abs_sources = [
            e["source"] for e in data["edges"]
            if e["source"].startswith("file:/")
        ]
        assert abs_sources == [], (
            f"Found absolute-path edge sources that should have been canonicalized: {abs_sources}"
        )
