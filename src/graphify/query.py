"""Graphify query engine: BFS/DFS search, shortest path, communities, god nodes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.graphify.model import GraphModel, GraphNode


@dataclass
class QueryResult:
    question: str
    mode: str
    depth: int
    token_budget: int
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[Any] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "mode": self.mode,
            "depth": self.depth,
            "token_budget": self.token_budget,
            "nodes": [
                {"label": node.label, "kind": node.kind, "properties": node.properties}
                for node in self.nodes
            ],
            "path": self.path,
            "context": self.context,
        }


class GraphifyQueryEngine:
    """Query a GraphModel with bounded search."""

    def __init__(self, graph: GraphModel) -> None:
        self.graph = graph

    def query(self, question: str, mode: str = "bfs", depth: int = 3, token_budget: int = 2000, context_filter: Optional[List[str]] = None) -> QueryResult:
        if mode not in {"bfs", "dfs"}:
            raise ValueError("Unsupported mode: %s. Use bfs or dfs." % mode)
        start = self._seed_from_question(question)
        visited = {start: 0}
        path = [start]
        if mode == "bfs":
            frontier = deque([(start, 0)])
            while frontier:
                current, current_depth = frontier.popleft()
                if current_depth >= depth:
                    continue
                for neighbor in self.graph.neighbors(current):
                    candidate = neighbor["node"].label
                    if candidate in visited and visited[candidate] <= current_depth:
                        continue
                    if context_filter and not self._matches_filter(neighbor, context_filter):
                        continue
                    visited[candidate] = current_depth + 1
                    path.append(candidate)
                    frontier.append((candidate, current_depth + 1))
        else:
            stack = [(start, 0)]
            while stack:
                current, current_depth = stack.pop()
                if current_depth >= depth:
                    continue
                neighbors = self.graph.neighbors(current)
                for neighbor in neighbors:
                    candidate = neighbor["node"].label
                    if candidate in visited and visited[candidate] <= current_depth:
                        continue
                    if context_filter and not self._matches_filter(neighbor, context_filter):
                        continue
                    visited[candidate] = current_depth + 1
                    path.append(candidate)
                    stack.append((candidate, current_depth + 1))
        selected = self._select_within_budget(
            [self.graph.node(label) for label in path], token_budget
        )
        context = self._render_context(selected)
        return QueryResult(
            question=question,
            mode=mode,
            depth=depth,
            token_budget=token_budget,
            nodes=selected,
            path=path,
            context=context,
        )

    def get_node(self, label: str) -> GraphNode:
        return self.graph.node(label)

    def get_neighbors(self, label: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.graph.neighbors(label, relation_filter=relation)

    def shortest_path(self, source: str, target: str, max_hops: int = 8) -> List[str]:
        if source not in self.graph._nodes or target not in self.graph._nodes:
            raise KeyError("source and target nodes must exist")
        if source == target:
            return [source]
        visited = {source: None}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            path_length = 0
            walk = current
            while visited[walk] is not None:
                path_length += 1
                walk = visited[walk]
            if path_length >= max_hops:
                continue
            for neighbor in self.graph.neighbors(current):
                candidate = neighbor["node"].label
                if candidate in visited:
                    continue
                visited[candidate] = current
                if candidate == target:
                    path = [target]
                    node = target
                    while node is not None:
                        path.append(node)
                        node = visited[node]
                    return list(reversed(path))[:-1]
                queue.append(candidate)
        raise ValueError("No path within max_hops=%s" % max_hops)

    def get_community(self, community_id: int) -> List[GraphNode]:
        nodes = [node for node in self.graph.nodes() if node.kind == "community" and node.properties.get("id") == community_id]
        if not nodes:
            return []
        center = nodes[0].label
        visited = {center}
        queue = deque([center])
        community = [self.graph.node(center)]
        while queue:
            current = queue.popleft()
            for neighbor in self.graph.neighbors(current):
                candidate = neighbor["node"].label
                if candidate in visited:
                    continue
                visited.add(candidate)
                community.append(neighbor["node"])
                queue.append(candidate)
        return community

    def god_nodes(self, top_n: int = 10) -> List[Dict[str, Any]]:
        adjacency: Dict[str, int] = {node.label: 0 for node in self.graph.nodes()}
        for edge in self.graph.edges():
            adjacency[edge.source] = adjacency.get(edge.source, 0) + 1
            adjacency[edge.target] = adjacency.get(edge.target, 0) + 1
        ranked = sorted(adjacency.items(), key=lambda item: (-item[1], item[0]))[:top_n]
        return [
            {"label": label, "degree": degree, "node": self.graph.node(label)}
            for label, degree in ranked
        ]

    def graph_stats(self) -> Dict[str, Any]:
        stats = self.graph.graph_stats()
        stats.update({
            "connected_components": self._connected_components_count(),
            "density": round(
                stats["edge_count"] / (stats["node_count"] * (stats["node_count"] - 1))
                if stats["node_count"] > 1
                else 0.0,
                4,
            ),
        })
        return stats

    def _seed_from_question(self, question: str) -> str:
        candidates = [node.label for node in self.graph.nodes() if node.label.lower() in question.lower()]
        if candidates:
            return candidates[0]
        fallback = list(self.graph.nodes())[0]
        return fallback.label if fallback else "repo"

    def _matches_filter(self, neighbor: Dict[str, Any], context_filter: List[str]) -> bool:
        node_kind = neighbor["node"].kind
        return node_kind in context_filter or neighbor["relation"] in context_filter

    def _select_within_budget(self, nodes: List[GraphNode], token_budget: int) -> List[GraphNode]:
        selected: List[GraphNode] = []
        budget = token_budget
        for node in nodes:
            estimated = len(node.label.split()) + len(node.kind.split())
            if estimated > budget:
                continue
            budget -= estimated
            selected.append(node)
        return selected

    def _render_context(self, nodes: List[GraphNode]) -> str:
        lines = []
        for node in nodes:
            props = ", ".join("%s=%s" % (k, v) for k, v in node.properties.items())
            lines.append("%s (%s)%s" % (node.label, node.kind, " [%s]" % props if props else ""))
        return "\n".join(lines)

    def _connected_components_count(self) -> int:
        unvisited: set[str] = {node.label for node in self.graph.nodes()}
        count = 0
        while unvisited:
            count += 1
            start = next(iter(unvisited))
            queue = deque([start])
            unvisited.remove(start)
            while queue:
                current = queue.popleft()
                for neighbor in self.graph.neighbors(current):
                    candidate_label: str = neighbor["node"].label
                    if candidate_label in unvisited:
                        unvisited.remove(candidate_label)
                        queue.append(candidate_label)
        return count
