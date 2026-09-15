"""Behavior contracts for the learning-graph assembler.

Asserts invariants (edges resolve to real nodes, clusters cover every node,
memory cards are represented consistently), never a snapshot of the live skill
catalog — that catalog grows every release and a count assertion would be a
change-detector.
"""

from __future__ import annotations

import json

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


def test_externally_installed_skill_not_learned(tmp_path):
    """Skills installed externally (e.g. ``npx skills add``) land in
    ``~/.hermes/skills/`` with ``source="profile"`` and ``provenance=None``.

    They must NOT appear as "learned" in the graph even when ``use_count > 0``
    from telemetry tracking — only agent-created skills (``provenance == "agent"``)
    should.
    """
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    ext_category = skills_dir / "tools"
    ext_category.mkdir(parents=True)

    # Externally installed skill — provenance=None, use_count > 0
    ext_skill = ext_category / "external-tool"
    ext_skill.mkdir()
    (ext_skill / "SKILL.md").write_text(
        "---\nname: external-tool\ncategory: tools\n---\nExternal skill body.\n",
        encoding="utf-8",
    )

    # Foreground agent-created skill — provenance="agent", created_by=None
    # (user-directed skill_manage(create): curator policy not set)
    fg_skill = ext_category / "fg-agent-skill"
    fg_skill.mkdir()
    (fg_skill / "SKILL.md").write_text(
        "---\nname: fg-agent-skill\ncategory: tools\n---\nForeground agent skill.\n",
        encoding="utf-8",
    )

    # Background-review agent-created skill — provenance="agent", created_by="agent"
    # (background review fork: curator policy also set)
    bg_skill = ext_category / "bg-agent-skill"
    bg_skill.mkdir()
    (bg_skill / "SKILL.md").write_text(
        "---\nname: bg-agent-skill\ncategory: tools\n---\nBackground agent skill.\n",
        encoding="utf-8",
    )

    usage = {
        "external-tool": {"use_count": 15, "created_by": None, "provenance": None, "state": "active"},
        "fg-agent-skill": {"use_count": 2, "created_by": None, "provenance": "agent", "state": "active"},
        "bg-agent-skill": {"use_count": 5, "created_by": "agent", "provenance": "agent", "state": "active"},
    }
    (skills_dir / ".usage.json").write_text(json.dumps(usage), encoding="utf-8")

    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    skill_names = {
        n["id"] for n in graph["nodes"] if n["kind"] == "skill"
    }
    # Both agent-created skills (foreground and background) ARE in the learned graph
    assert "fg-agent-skill" in skill_names
    assert "bg-agent-skill" in skill_names
    # Externally installed skill is NOT in the learned graph
    assert "external-tool" not in skill_names
