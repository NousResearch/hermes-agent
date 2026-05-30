"""Tests for scripts/code-scan/graph_schema.py."""
import os
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "code-scan"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from graph_schema import (
    NodeType,
    EdgeType,
    NODE_TYPE_ALIASES,
    EDGE_TYPE_ALIASES,
    validate_node,
    validate_edge,
    validate_graph,
)


class TestEnums:
    """Test NodeType and EdgeType enum definitions."""

    def test_node_type_values(self):
        assert NodeType.FILE == "file"
        assert NodeType.FUNCTION == "function"
        assert NodeType.CLASS == "class"
        assert NodeType.MODULE == "module"

    def test_edge_type_values(self):
        assert EdgeType.IMPORTS == "imports"
        assert EdgeType.CONTAINS == "contains"
        assert EdgeType.CALLS == "calls"
        assert EdgeType.TESTED_BY == "tested_by"
        assert EdgeType.CONFIGURES == "configures"
        assert EdgeType.DOCUMENTS == "documents"


class TestAliasMaps:
    """Test NODE_TYPE_ALIASES and EDGE_TYPE_ALIASES."""

    def test_node_type_alias_func(self):
        assert NODE_TYPE_ALIASES["func"] == NodeType.FUNCTION
        assert NODE_TYPE_ALIASES["fn"] == NodeType.FUNCTION
        assert NODE_TYPE_ALIASES["method"] == NodeType.FUNCTION

    def test_node_type_alias_file(self):
        assert NODE_TYPE_ALIASES["src"] == NodeType.FILE

    def test_node_type_alias_class(self):
        assert NODE_TYPE_ALIASES["type"] == NodeType.CLASS

    def test_node_type_alias_module(self):
        assert NODE_TYPE_ALIASES["pkg"] == NodeType.MODULE
        assert NODE_TYPE_ALIASES["package"] == NodeType.MODULE

    def test_edge_type_alias_import(self):
        assert EDGE_TYPE_ALIASES["import"] == EdgeType.IMPORTS
        assert EDGE_TYPE_ALIASES["imports_from"] == EdgeType.IMPORTS

    def test_edge_type_alias_calls(self):
        assert EDGE_TYPE_ALIASES["invoke"] == EdgeType.CALLS
        assert EDGE_TYPE_ALIASES["invokes"] == EdgeType.CALLS

    def test_edge_type_alias_contains(self):
        assert EDGE_TYPE_ALIASES["has"] == EdgeType.CONTAINS


class TestValidateNode:
    """Test validate_node() function."""

    def test_valid_node(self):
        node = {"node_type": "file", "node_id": "n1", "filePath": "src/main.py"}
        assert validate_node(node) == []

    def test_valid_node_with_enum(self):
        node = {"node_type": NodeType.FILE, "node_id": "n2", "filePath": "src/utils.py"}
        assert validate_node(node) == []

    def test_missing_node_type(self):
        node = {"node_id": "n1", "filePath": "src/main.py"}
        issues = validate_node(node)
        assert any("node_type" in i for i in issues)

    def test_invalid_node_type(self):
        node = {"node_type": "bogus", "node_id": "n1", "filePath": "src/main.py"}
        issues = validate_node(node)
        assert any("node_type" in i for i in issues)

    def test_missing_node_id(self):
        node = {"node_type": "file", "filePath": "src/main.py"}
        issues = validate_node(node)
        assert any("node_id" in i for i in issues)

    def test_missing_filepath(self):
        node = {"node_type": "file", "node_id": "n1"}
        issues = validate_node(node)
        assert any("filePath" in i for i in issues)

    def test_alias_resolution(self):
        node = {"node_type": "func", "node_id": "n1", "filePath": "src/main.py"}
        assert validate_node(node) == []

    def test_node_id_not_string(self):
        node = {"node_type": "file", "node_id": 123, "filePath": "src/main.py"}
        issues = validate_node(node)
        assert any("node_id" in i for i in issues)


class TestValidateEdge:
    """Test validate_edge() function."""

    def test_valid_edge(self):
        edge = {"edge_type": "imports", "source": "n1", "target": "n2"}
        known = {"n1", "n2"}
        assert validate_edge(edge, known) == []

    def test_valid_edge_with_alias(self):
        edge = {"edge_type": "import", "source": "n1", "target": "n2"}
        known = {"n1", "n2"}
        assert validate_edge(edge, known) == []

    def test_missing_edge_type(self):
        edge = {"source": "n1", "target": "n2"}
        known = {"n1", "n2"}
        issues = validate_edge(edge, known)
        assert any("edge_type" in i for i in issues)

    def test_invalid_edge_type(self):
        edge = {"edge_type": "bogus", "source": "n1", "target": "n2"}
        known = {"n1", "n2"}
        issues = validate_edge(edge, known)
        assert any("edge_type" in i for i in issues)

    def test_source_not_in_known(self):
        edge = {"edge_type": "imports", "source": "n99", "target": "n2"}
        known = {"n1", "n2"}
        issues = validate_edge(edge, known)
        assert any("source" in i for i in issues)

    def test_target_not_in_known(self):
        edge = {"edge_type": "imports", "source": "n1", "target": "n99"}
        known = {"n1", "n2"}
        issues = validate_edge(edge, known)
        assert any("target" in i for i in issues)

    def test_self_referencing_edge(self):
        edge = {"edge_type": "calls", "source": "n1", "target": "n1"}
        known = {"n1"}
        issues = validate_edge(edge, known)
        assert any("self" in i.lower() or "same" in i.lower() for i in issues)


class TestValidateGraph:
    """Test validate_graph() function."""

    def test_empty_graph(self):
        graph = {"nodes": [], "edges": []}
        result = validate_graph(graph)
        assert result == {"issues": [], "warnings": []}

    def test_valid_graph(self):
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "n1", "filePath": "src/main.py"},
                {"node_type": "function", "node_id": "n2", "filePath": "src/main.py"},
            ],
            "edges": [
                {"edge_type": "contains", "source": "n1", "target": "n2"},
            ],
        }
        result = validate_graph(graph)
        assert result["issues"] == []

    def test_graph_with_node_issues(self):
        graph = {
            "nodes": [
                {"node_type": "bogus", "node_id": "n1", "filePath": "src/main.py"},
            ],
            "edges": [],
        }
        result = validate_graph(graph)
        assert len(result["issues"]) > 0

    def test_graph_with_edge_issues(self):
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "n1", "filePath": "src/main.py"},
            ],
            "edges": [
                {"edge_type": "bogus", "source": "n1", "target": "n99"},
            ],
        }
        result = validate_graph(graph)
        assert len(result["issues"]) > 0

    def test_graph_with_orphan_node(self):
        """Node not referenced by any edge should produce a warning."""
        graph = {
            "nodes": [
                {"node_type": "file", "node_id": "n1", "filePath": "src/main.py"},
                {"node_type": "file", "node_id": "n2", "filePath": "src/utils.py"},
            ],
            "edges": [
                {"edge_type": "contains", "source": "n1", "target": "n2"},
            ],
        }
        result = validate_graph(graph)
        assert result["issues"] == []

    def test_graph_missing_nodes_key(self):
        graph = {"edges": []}
        result = validate_graph(graph)
        assert len(result["issues"]) > 0

    def test_graph_missing_edges_key(self):
        graph = {"nodes": []}
        result = validate_graph(graph)
        assert len(result["issues"]) > 0
