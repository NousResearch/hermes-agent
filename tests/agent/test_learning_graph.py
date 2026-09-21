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


def _vault_note(title, record_id, kind, agent, related=()):
    relations = "\n".join(f"  - {item}" for item in related) or "  []"
    execution = "reference-only" if kind == "skill-reference" else "not-applicable"
    return f"""---
title: {title}
schema_name: agent_learning_record_v1
record_id: {record_id}
kind: {kind}
agent: {agent}
memory_area: Runtime
status: current
authority: source-index
classification: diagnostic-only
sync_owned: true
execution_status: {execution}
related_record_ids:
{relations}
---

# {title}

Bounded shared summary.
"""


def test_shared_catalog_is_off_by_default():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["learning"] == {
        "vault_dir": "",
        "shared_catalog_enabled": False,
    }


def test_enabled_shared_catalog_adds_typed_nodes_and_explicit_edges(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    vault = tmp_path / "vault"
    memory = vault / "Memories" / "Codex" / "Runtime" / "lesson.md"
    skill = vault / "Reference" / "Agent Skills" / "Claude" / "playwright.md"
    memory.parent.mkdir(parents=True)
    skill.parent.mkdir(parents=True)
    memory.write_text(
        _vault_note("Lesson", "alr-memory", "memory", "codex", ("alr-skill",)),
        encoding="utf-8",
    )
    skill.write_text(
        _vault_note("Playwright", "alr-skill", "skill-reference", "claude"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "learning": {
                "vault_dir": str(vault),
                "shared_catalog_enabled": True,
            }
        },
    )
    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    by_id = {node["id"]: node for node in graph["nodes"]}
    assert by_id["vault-memory:alr-memory"]["originAgent"] == "codex"
    assert by_id["vault-memory:alr-memory"]["area"] == "Runtime"
    assert by_id["vault-skill:alr-skill"]["executionStatus"] == "reference-only"
    assert {(edge["source"], edge["target"]) for edge in graph["edges"]} >= {
        ("vault-memory:alr-memory", "vault-skill:alr-skill")
    }
    assert graph["shared_catalog"]["nodes"] == 2
    assert graph["shared_catalog"]["diagnostics"] == []
    assert not (home / "memories" / "MEMORY.md").exists()


def test_invalid_shared_catalog_config_preserves_local_graph(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("Local lesson", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "learning": {
                "vault_dir": "relative/vault",
                "shared_catalog_enabled": True,
            }
        },
    )
    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    assert any(node["label"] == "Local lesson" for node in graph["nodes"])
    assert graph["shared_catalog"]["nodes"] == 0
    assert graph["shared_catalog"]["diagnostics"] == ["vault_dir_not_absolute"]
