"""Research knowledge graph backed by graph.jsonl."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from research.models import GraphEdge
from research.state_store import ResearchStateStore


class KnowledgeGraph:
    """Operate on the project knowledge graph stored in graph.jsonl."""

    def __init__(self, store: ResearchStateStore, project_id: str):
        self.store = store
        self.project_id = project_id

    def _load_edges(self) -> list[dict[str, Any]]:
        return self.store.load_graph_edges(self.project_id)

    def add_edge(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        """Persist a new edge and return its record."""
        edge = GraphEdge(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relation=relation,
            weight=weight,
        )
        record = edge.model_dump(mode="json")
        self.store.append_jsonl(self.project_id, "graph", record)
        return record

    def get_neighbors(
        self,
        node_type: str,
        node_id: str,
        direction: str = "both",
        relation: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return connected nodes."""
        edges = self._load_edges()
        results: list[dict[str, Any]] = []
        for edge in edges:
            if relation and edge.get("relation") != relation:
                continue
            if direction in ("out", "both"):
                if edge.get("source_type") == node_type and edge.get("source_id") == node_id:
                    results.append({
                        "node_type": edge["target_type"],
                        "node_id": edge["target_id"],
                        "relation": edge["relation"],
                        "direction": "out",
                        "weight": edge.get("weight", 1.0),
                    })
            if direction in ("in", "both"):
                if edge.get("target_type") == node_type and edge.get("target_id") == node_id:
                    results.append({
                        "node_type": edge["source_type"],
                        "node_id": edge["source_id"],
                        "relation": edge["relation"],
                        "direction": "in",
                        "weight": edge.get("weight", 1.0),
                    })
        return results

    def get_subgraph(self, node_type: str, node_id: str, depth: int = 2) -> dict[str, Any]:
        """BFS traversal returning nodes and edges within *depth* hops."""
        visited: set[tuple[str, str]] = set()
        queue: deque[tuple[str, str, int]] = deque()
        queue.append((node_type, node_id, 0))
        visited.add((node_type, node_id))
        nodes: list[dict[str, Any]] = [{"type": node_type, "id": node_id, "depth": 0}]
        subgraph_edges: list[dict[str, Any]] = []

        edges = self._load_edges()

        while queue:
            ntype, nid, d = queue.popleft()
            if d >= depth:
                continue
            for edge in edges:
                neighbor_type = neighbor_id = None
                if edge.get("source_type") == ntype and edge.get("source_id") == nid:
                    neighbor_type = edge["target_type"]
                    neighbor_id = edge["target_id"]
                elif edge.get("target_type") == ntype and edge.get("target_id") == nid:
                    neighbor_type = edge["source_type"]
                    neighbor_id = edge["source_id"]
                if neighbor_type and (neighbor_type, neighbor_id) not in visited:
                    visited.add((neighbor_type, neighbor_id))
                    nodes.append({"type": neighbor_type, "id": neighbor_id, "depth": d + 1})
                    queue.append((neighbor_type, neighbor_id, d + 1))
                    subgraph_edges.append(edge)

        return {"nodes": nodes, "edges": subgraph_edges}

    def get_lineage(self, node_type: str, node_id: str) -> list[dict[str, Any]]:
        """Trace the research lineage chain: paper -> idea -> hypothesis -> experiment -> result."""
        lineage_order = ["paper", "idea", "hypothesis", "experiment", "run", "result", "eval"]
        chain: list[dict[str, Any]] = [{"type": node_type, "id": node_id}]
        visited: set[tuple[str, str]] = {(node_type, node_id)}
        edges = self._load_edges()

        current_type, current_id = node_type, node_id

        # Trace forward (outgoing)
        changed = True
        while changed:
            changed = False
            for edge in edges:
                if edge.get("source_type") == current_type and edge.get("source_id") == current_id:
                    target_type = edge["target_type"]
                    target_id = edge["target_id"]
                    if (target_type, target_id) not in visited:
                        visited.add((target_type, target_id))
                        chain.append({"type": target_type, "id": target_id, "relation": edge["relation"]})
                        current_type, current_id = target_type, target_id
                        changed = True
                        break

        # Also trace backward from the original node
        current_type, current_id = node_type, node_id
        backward: list[dict[str, Any]] = []
        changed = True
        while changed:
            changed = False
            for edge in edges:
                if edge.get("target_type") == current_type and edge.get("target_id") == current_id:
                    source_type = edge["source_type"]
                    source_id = edge["source_id"]
                    if (source_type, source_id) not in visited:
                        visited.add((source_type, source_id))
                        backward.append({"type": source_type, "id": source_id, "relation": edge["relation"]})
                        current_type, current_id = source_type, source_id
                        changed = True
                        break

        # Combine: backward (reversed) + original chain
        full_chain = list(reversed(backward)) + chain

        # Sort by lineage order if possible
        order_map = {t: i for i, t in enumerate(lineage_order)}
        full_chain.sort(key=lambda n: order_map.get(n["type"], 999))

        return full_chain

    def summary(self) -> dict[str, Any]:
        """Return node and edge counts by type."""
        edges = self._load_edges()
        node_types: defaultdict[str, set[str]] = defaultdict(set)
        edge_types: defaultdict[str, int] = defaultdict(int)

        for edge in edges:
            node_types[edge.get("source_type", "")].add(edge.get("source_id", ""))
            node_types[edge.get("target_type", "")].add(edge.get("target_id", ""))
            edge_types[edge.get("relation", "")] += 1

        return {
            "total_edges": len(edges),
            "node_counts": {k: len(v) for k, v in sorted(node_types.items())},
            "edge_counts": dict(sorted(edge_types.items())),
        }
