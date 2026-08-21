"""Tests for Graphify query engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys_path = Path(__file__).resolve().parents[2] / "src"
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from graphify.model import GraphEdge, GraphModel, GraphNode  # noqa: E402
from graphify.query import GraphifyQueryEngine, QueryResult  # noqa: E402


@pytest.fixture()
def sample_graph() -> GraphModel:
    graph = GraphModel()
    graph.add_node(GraphNode(label="repo", kind="root", properties={"path": "/repo"}))
    graph.add_node(GraphNode(label="agent", kind="module", properties={"path": "/repo/agent"}))
    graph.add_node(GraphNode(label="cli", kind="module", properties={"path": "/repo/cli"}))
    graph.add_node(GraphNode(label="tool", kind="module", properties={"path": "/repo/cli/tool"}))
    graph.add_node(GraphNode(label="gateway", kind="module", properties={"path": "/repo/gateway"}))
    graph.add_node(GraphNode(label="state", kind="module", properties={"path": "/repo/hermes_state.py"}))
    graph.add_edge(GraphEdge(source="repo", target="agent", relation="contains"))
    graph.add_edge(GraphEdge(source="repo", target="cli", relation="contains"))
    graph.add_edge(GraphEdge(source="repo", target="gateway", relation="contains"))
    graph.add_edge(GraphEdge(source="cli", target="tool", relation="uses"))
    graph.add_edge(GraphEdge(source="gateway", target="state", relation="uses"))
    graph.add_edge(GraphEdge(source="agent", target="state", relation="uses"))
    return graph


def test_query_result_serialization(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    result = engine.query("repo", mode="bfs", depth=1, token_budget=10)
    serialized = result.to_dict()
    assert serialized["mode"] == "bfs"
    assert serialized["depth"] == 1


def test_bfs_query_includes_neighbors(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    result = engine.query("repo", mode="bfs", depth=1, token_budget=1000)
    labels = [node.label for node in result.nodes]
    assert "agent" in labels
    assert "cli" in labels
    assert "repo" in labels


def test_bfs_depth_limits_expansion(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    result = engine.query("repo", mode="bfs", depth=1, token_budget=1000)
    assert "tool" not in [node.label for node in result.nodes]
    result = engine.query("repo", mode="bfs", depth=2, token_budget=1000)
    assert "tool" in [node.label for node in result.nodes]


def test_dfs_query_returns_all_reachable_nodes(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    result = engine.query("repo", mode="dfs", depth=10, token_budget=1000)
    assert set(node.label for node in result.nodes) == sample_graph._nodes.keys()


def test_query_deterministic(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    first = engine.query("repo", mode="bfs", depth=2, token_budget=1000)
    second = engine.query("repo", mode="bfs", depth=2, token_budget=1000)
    assert [node.label for node in first.nodes] == [node.label for node in second.nodes]
    assert first.path == second.path


def test_invalid_mode_raises(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    with pytest.raises(ValueError):
        engine.query("repo", mode="invalid", depth=1, token_budget=10)


def test_get_node_returns_exact_match(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    node = engine.get_node("tool")
    assert node.label == "tool"
    assert node.kind == "module"


def test_get_neighbors_returns_directed_edges(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    neighbors = engine.get_neighbors("cli")
    directions = {item["direction"] for item in neighbors}
    relations = {item["relation"] for item in neighbors}
    labels = {item["node"].label for item in neighbors}
    assert directions == {"in", "out"}
    assert relations == {"contains", "uses"}
    assert labels == {"repo", "tool"}


def test_get_neighbors_filter_by_relation(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    neighbors = engine.get_neighbors("repo", relation="contains")
    assert {item["node"].label for item in neighbors} == {"agent", "cli", "gateway"}


def test_get_neighbors_missing_node_raises(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    with pytest.raises(KeyError):
        engine.get_neighbors("missing")


def test_shortest_path_same_node(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    assert engine.shortest_path("repo", "repo") == ["repo"]


def test_shortest_path_across_two_hops(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    path = engine.shortest_path("cli", "tool")
    assert path == ["cli", "tool"]


def test_shortest_path_missing_source_raises(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    with pytest.raises(KeyError):
        engine.shortest_path("missing", "tool")


def test_shortest_path_raises_when_no_route() -> None:
    graph = GraphModel()
    graph.add_node(GraphNode(label="a", kind="module"))
    graph.add_node(GraphNode(label="b", kind="module"))
    engine = GraphifyQueryEngine(graph)
    with pytest.raises(ValueError):
        engine.shortest_path("a", "b")


def test_get_community_returns_connected_component() -> None:
    graph = GraphModel()
    graph.add_node(GraphNode(label="c1", kind="community", properties={"id": 1}))
    graph.add_node(GraphNode(label="a", kind="module"))
    graph.add_node(GraphNode(label="b", kind="module"))
    graph.add_edge(GraphEdge(source="c1", target="a", relation="owns"))
    graph.add_edge(GraphEdge(source="a", target="b", relation="links"))
    engine = GraphifyQueryEngine(graph)
    community = engine.get_community(1)
    assert {node.label for node in community} == {"c1", "a", "b"}


def test_get_community_missing_returns_empty_list() -> None:
    graph = GraphModel()
    graph.add_node(GraphNode(label="repo", kind="root"))
    engine = GraphifyQueryEngine(graph)
    assert engine.get_community(99) == []


def test_god_nodes_sorted_by_degree() -> None:
    graph = GraphModel()
    graph.add_node(GraphNode(label="hub", kind="module"))
    graph.add_node(GraphNode(label="a", kind="module"))
    graph.add_node(GraphNode(label="b", kind="module"))
    graph.add_node(GraphNode(label="c", kind="module"))
    graph.add_edge(GraphEdge(source="hub", target="a", relation="uses"))
    graph.add_edge(GraphEdge(source="hub", target="b", relation="uses"))
    graph.add_edge(GraphEdge(source="hub", target="c", relation="uses"))
    engine = GraphifyQueryEngine(graph)
    ranked = engine.god_nodes(top_n=2)
    assert ranked[0]["label"] == "hub"
    assert ranked[0]["degree"] == 3


def test_graph_stats_density(sample_graph: GraphModel) -> None:
    engine = GraphifyQueryEngine(sample_graph)
    stats = engine.graph_stats()
    # sample_graph has 6 edges among 6 nodes => directed density = 6 / (6*5)
    assert stats["density"] == pytest.approx(6 / (6 * 5))
    assert stats["connected_components"] == 1


def test_graph_stats_multiple_components() -> None:
    graph = GraphModel()
    graph.add_node(GraphNode(label="a", kind="module"))
    graph.add_node(GraphNode(label="b", kind="module"))
    graph.add_edge(GraphEdge(source="a", target="b", relation="links"))
    graph.add_node(GraphNode(label="c", kind="module"))
    engine = GraphifyQueryEngine(graph)
    stats = engine.graph_stats()
    assert stats["connected_components"] == 2
