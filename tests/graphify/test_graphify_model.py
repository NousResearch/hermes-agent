"""Tests for Graphify core model and persistence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys_path = Path(__file__).resolve().parents[2] / "src"
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from graphify.model import GraphEdge, GraphJsonRepository, GraphModel, GraphNode  # noqa: E402


@pytest.fixture()
def tiny_graph() -> GraphModel:
    graph = GraphModel()
    graph.add_node(GraphNode(label="repo", kind="root"))
    graph.add_node(GraphNode(label="agent", kind="module"))
    graph.add_node(GraphNode(label="cli", kind="module"))
    graph.add_node(GraphNode(label="tool", kind="module"))
    graph.add_edge(GraphEdge(source="repo", target="agent", relation="contains"))
    graph.add_edge(GraphEdge(source="repo", target="cli", relation="contains"))
    graph.add_edge(GraphEdge(source="cli", target="tool", relation="uses"))
    return graph


def test_graph_model_node_lookup(tiny_graph: GraphModel) -> None:
    assert tiny_graph.node("repo").kind == "root"
    assert tiny_graph.node("agent").properties == {}


def test_graph_model_missing_node_raises(tiny_graph: GraphModel) -> None:
    with pytest.raises(KeyError):
        tiny_graph.node("missing")


def test_graph_model_edge_requires_existing_nodes() -> None:
    graph = GraphModel()
    graph.add_node(GraphNode(label="a", kind="module"))
    with pytest.raises(KeyError):
        graph.add_edge(GraphEdge(source="a", target="missing", relation="links"))


def test_graph_model_neighbors_direction(tiny_graph: GraphModel) -> None:
    repo_neighbors = tiny_graph.neighbors("repo")
    neighbor_labels = [item["node"].label for item in repo_neighbors]
    assert set(neighbor_labels) == {"agent", "cli"}
    assert all(item["direction"] == "out" for item in repo_neighbors)

    cli_neighbors = tiny_graph.neighbors("cli")
    assert set(item["node"].label for item in cli_neighbors) == {"repo", "tool"}
    incoming = [item for item in cli_neighbors if item["direction"] == "in"]
    outgoing = [item for item in cli_neighbors if item["direction"] == "out"]
    assert len(incoming) == 1
    assert len(outgoing) == 1
    assert incoming[0]["node"].label == "repo"
    assert outgoing[0]["node"].label == "tool"


def test_graph_model_neighbors_filter(tiny_graph: GraphModel) -> None:
    repo_neighbors = tiny_graph.neighbors("repo", relation_filter="contains")
    assert len(repo_neighbors) == 2


def test_graph_model_graph_stats_deterministic(tiny_graph: GraphModel) -> None:
    first = tiny_graph.graph_stats()
    second = tiny_graph.graph_stats()
    assert first == second
    assert first["node_count"] == 4
    assert first["edge_count"] == 3
    assert first["avg_degree"] == 1.5


def test_graph_json_repository_round_trip(tmp_path: Path) -> None:
    repo = GraphJsonRepository(str(tmp_path / "graphify-out"))
    graph = GraphModel()
    graph.add_node(GraphNode(label="repo", kind="root"))
    graph.add_node(GraphNode(label="agent", kind="module"))
    graph.add_edge(GraphEdge(source="repo", target="agent", relation="contains"))
    repo.save(graph)
    loaded = repo.load()
    assert loaded.node("repo").kind == "root"
    assert loaded.node("agent").label == "agent"
    assert loaded.graph_stats() == graph.graph_stats()


def test_graph_json_repository_missing_file_raises(tmp_path: Path) -> None:
    repo = GraphJsonRepository(str(tmp_path / "graphify-out"))
    with pytest.raises(FileNotFoundError):
        repo.load()


def test_graph_json_repository_writes_sorted_json(tmp_path: Path) -> None:
    repo = GraphJsonRepository(str(tmp_path / "graphify-out"))
    graph = GraphModel()
    graph.add_node(GraphNode(label="z", kind="module"))
    graph.add_node(GraphNode(label="a", kind="module"))
    graph.add_edge(GraphEdge(source="z", target="a", relation="links"))
    repo.save(graph)
    contents = (tmp_path / "graphify-out" / "graph.json").read_text(encoding="utf-8")
    parsed = json.loads(contents)
    # Node array preserves insertion order; object keys are sorted for determinism.
    assert parsed["nodes"][0]["label"] == "z"
    assert parsed["nodes"][-1]["label"] == "a"
    assert list(parsed["nodes"][0].keys()) == sorted(parsed["nodes"][0].keys())
