"""Graphify: repo-aware knowledge graph model, indexing, and persistence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class GraphNode:
    label: str
    kind: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)


class GraphModel:
    """Deterministic in-memory knowledge graph."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.label] = node

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise KeyError(
                "Both source and target nodes must exist before adding edge: %s -> %s"
                % (edge.source, edge.target)
            )
        self._edges.append(edge)

    def node(self, label: str) -> GraphNode:
        try:
            return self._nodes[label]
        except KeyError as exc:
            raise KeyError("Node not found: %s" % label) from exc

    def neighbors(self, label: str, relation_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if label not in self._nodes:
            raise KeyError("Node not found: %s" % label)
        result: List[Dict[str, Any]] = []
        for edge in self._edges:
            if edge.source == label:
                if relation_filter and edge.relation != relation_filter:
                    continue
                result.append(
                    {
                        "direction": "out",
                        "relation": edge.relation,
                        "node": self._nodes[edge.target],
                        "edge": edge,
                    }
                )
            elif edge.target == label:
                if relation_filter and edge.relation != relation_filter:
                    continue
                result.append(
                    {
                        "direction": "in",
                        "relation": edge.relation,
                        "node": self._nodes[edge.source],
                        "edge": edge,
                    }
                )
        return result

    def nodes(self) -> Iterable[GraphNode]:
        return list(self._nodes.values())

    def edges(self) -> List[GraphEdge]:
        return list(self._edges)

    def graph_stats(self) -> Dict[str, Any]:
        adjacency: Dict[str, int] = {label: 0 for label in self._nodes}
        for edge in self._edges:
            adjacency[edge.source] = adjacency.get(edge.source, 0) + 1
            adjacency[edge.target] = adjacency.get(edge.target, 0) + 1
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "avg_degree": round(sum(adjacency.values()) / max(len(adjacency), 1), 4),
        }


class GraphJsonRepository:
    """Persist GraphModel to graphify-out/graph.json."""

    def __init__(self, output_dir: str = "graphify-out") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, "graph.json")

    def save(self, graph: GraphModel) -> None:
        payload = {
            "nodes": [
                {
                    "label": node.label,
                    "kind": node.kind,
                    "properties": node.properties,
                }
                for node in graph.nodes()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "properties": edge.properties,
                }
                for edge in graph.edges()
            ],
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.path)

    def load(self) -> GraphModel:
        if not os.path.exists(self.path):
            raise FileNotFoundError("Graph file not found at %s" % self.path)
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        graph = GraphModel()
        for raw in payload.get("nodes", []):
            graph.add_node(
                GraphNode(
                    label=raw["label"],
                    kind=raw["kind"],
                    properties=raw.get("properties", {}),
                )
            )
        for raw in payload.get("edges", []):
            graph.add_edge(
                GraphEdge(
                    source=raw["source"],
                    target=raw["target"],
                    relation=raw["relation"],
                    properties=raw.get("properties", {}),
                )
            )
        return graph
