"""Behavior contracts for the learning-graph assembler.

Asserts invariants (edges resolve to real nodes, clusters cover every node,
memory cards are represented consistently), never a snapshot of the live skill
catalog — that catalog grows every release and a count assertion would be a
change-detector.
"""

from __future__ import annotations

from agent import learning_graph
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _node(name: str, category: str, related=None):
    n = learning_graph.SkillNode(name=name, category=category)
    n.related = list(related or [])
    return n




def test_density_stats_count_isolated_nodes():
    nodes = {
        "a": _node("a", "x", related=["b"]),
        "b": _node("b", "x", related=["a"]),
        "c": _node("c", "y"),
    }
    stats = learning_graph.density_stats(nodes, learning_graph.build_edges(nodes))

    assert stats["nodes"] == 3
    assert stats["linked_nodes"] == 2
    assert stats["isolated_pct"] == round(100 / 3, 1)




def test_memory_is_cards_split_on_separator(tmp_path):
    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text(
        "Project uses pytest with xdist\n§\nUser prefers concise responses",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    titles = [c["title"] for c in graph["memory"]]
    assert "Project uses pytest with xdist" in titles
    assert "User prefers concise responses" in titles
    # Memory cards remain typed cards and also appear as memory-kind nodes.
    assert all(c["source"] in {"memory", "profile"} for c in graph["memory"])
    assert all("timestamp" in c for c in graph["memory"])
    assert any(n["kind"] == "memory" for n in graph["nodes"])






def test_provider_cards_follow_file_cards(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("Project uses pytest", encoding="utf-8")

    class Provider:
        def journey_cards(self, limit=200):
            return [
                {"timestamp": 1_700_000_000, "title": "Learned from the demo provider", "body": "body text",
                 "kind": "lesson", "session_id": "demo-session"},
                {"body": "Only a body, dated by ISO string", "timestamp": "2023-11-14T22:13:20+00:00"},
                {"title": "no body"},  # dropped: body is required
            ]

    monkeypatch.setattr(learning_graph, "_load_active_provider", lambda: ("demo", Provider()))
    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    assert [c["source"] for c in graph["memory"]] == ["memory", "demo", "demo"]
    node = next(n for n in graph["nodes"] if n["id"] == "memory:demo:1")
    assert node["kind"] == "memory" and node["memorySource"] == "demo"
    assert node["label"] == "Learned from the demo provider" and node["timestamp"] == 1_700_000_000
    assert graph["memory"][1]["session_id"] == "demo-session"
    second = graph["memory"][2]
    assert second["title"] == "Only a body, dated by ISO string" and second["timestamp"] == 1_700_000_000
    assert graph["stats"]["memory_nodes"] == 3


def test_graph_survives_a_broken_provider(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()

    class Broken:
        def journey_cards(self, limit=200):
            raise RuntimeError("boom")

    monkeypatch.setattr(learning_graph, "_load_active_provider", lambda: ("broken", Broken()))
    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)
    assert graph["memory"] == []


def test_full_payload_shape_and_edge_integrity(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    ids = {n["id"] for n in graph["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    # Every node's category appears in the cluster list.
    cluster_cats = {c["category"] for c in graph["clusters"]}
    assert all(n["category"] in cluster_cats for n in graph["nodes"])
    skill_nodes = [n for n in graph["nodes"] if n["kind"] == "skill"]
    assert graph["stats"]["nodes"] == len(skill_nodes)
    assert graph["stats"]["memory_nodes"] == len(graph["memory"])
    assert all("timestamp" in n for n in graph["nodes"])
