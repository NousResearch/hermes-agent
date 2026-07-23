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


def test_edges_only_connect_existing_nodes():
    nodes = {
        "a": _node("a", "x", related=["b", "ghost"]),
        "b": _node("b", "x", related=["a"]),
        "c": _node("c", "y"),
    }
    edges = learning_graph.build_edges(nodes)

    # The a→b link is kept once (deduped, undirected); a→ghost is dropped.
    assert edges == [("a", "b")]


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


def test_skill_node_timestamp_uses_iso_usage_activity(tmp_path, monkeypatch):
    skill_dir = tmp_path / "skills" / "dev" / "iso-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("---\nname: iso-skill\ncategory: dev\n---\n# ISO\n", encoding="utf-8")

    monkeypatch.setattr(
        learning_graph,
        "_load_usage",
        lambda: {
            "iso-skill": {
                "created_by": "agent",
                "last_used_at": "2026-04-30T12:00:00+00:00",
                "use_count": 1,
            }
        },
    )

    nodes = learning_graph.build_skill_nodes([("profile", tmp_path / "skills")])

    assert nodes["iso-skill"].timestamp == 1_777_550_400


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


def test_malformed_frontmatter_metadata_does_not_crash(tmp_path):
    """``parse_frontmatter``'s malformed-YAML fallback stores every value as a
    string, so ``metadata`` can be a str. The graph must tolerate that instead
    of crashing on chained ``.get()`` (the /journey base-CLI crash)."""
    skill_dir = tmp_path / "skills" / "misc" / "bad-skill"
    skill_dir.mkdir(parents=True)
    # The unterminated quote makes yaml_load raise → fallback → metadata is a str.
    skill_dir.joinpath("SKILL.md").write_text(
        '---\nname: bad-skill\nmetadata: not-a-dict\ndescription: "oops\n---\n# Bad\n',
        encoding="utf-8",
    )

    node = learning_graph.build_skill_nodes([("profile", tmp_path / "skills")])["bad-skill"]

    assert node.category == "misc"  # directory fallback, not a crash
    assert node.related == []


def test_hermes_meta_tolerates_non_dict():
    assert learning_graph._hermes_meta({"metadata": "junk"}) == {}
    assert learning_graph._hermes_meta({"metadata": {"hermes": "junk"}}) == {}
    assert learning_graph._hermes_meta({"metadata": {"hermes": {"category": "x"}}}) == {"category": "x"}


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


# ── External provider memory (journey_cards) ────────────────────────────────


class _FakeProvider:
    def __init__(self, cards):
        self._cards = cards

    def journey_cards(self, limit=200):
        return self._cards[:limit]


class _LegacyProvider:
    """A provider written before journey_cards existed — no such attribute."""


def _patch_active_provider(monkeypatch, name, provider):
    import plugins.memory as pm

    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: name)
    monkeypatch.setattr(pm, "load_memory_provider", lambda n: provider)


def test_provider_cards_normalized_and_tagged_with_provider_name(monkeypatch):
    _patch_active_provider(
        monkeypatch,
        "fakemem",
        _FakeProvider(
            [
                {"body": "User prefers rye bread", "timestamp": 1_770_000_000},
                {"body": "line one\nline two", "timestamp": "2026-04-30T12:00:00+00:00"},
                {"body": ""},          # dropped: empty body
                "not-a-dict",           # dropped: wrong shape
            ]
        ),
    )

    cards = learning_graph._provider_memory_cards()

    assert [c["source"] for c in cards] == ["fakemem", "fakemem"]
    assert cards[0]["body"] == "User prefers rye bread"
    assert cards[0]["title"] == "User prefers rye bread"
    assert cards[0]["timestamp"] == 1_770_000_000
    # Title defaults to the first line; ISO timestamps normalize to unix secs.
    assert cards[1]["title"] == "line one"
    assert cards[1]["timestamp"] == 1_777_550_400


def test_provider_cards_empty_when_no_provider_or_legacy_or_raising(monkeypatch):
    import plugins.memory as pm

    # No active provider configured.
    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: None)
    assert learning_graph._provider_memory_cards() == []

    # Older provider without the hook.
    _patch_active_provider(monkeypatch, "oldmem", _LegacyProvider())
    assert learning_graph._provider_memory_cards() == []

    # Provider whose hook raises (backend down) must not propagate.
    class _Boom:
        def journey_cards(self, limit=200):
            raise RuntimeError("backend down")

    _patch_active_provider(monkeypatch, "boommem", _Boom())
    assert learning_graph._provider_memory_cards() == []


def test_provider_cards_append_after_file_cards(tmp_path, monkeypatch):
    """Provider nodes must not shift MEMORY.md/USER.md indices — the mutation
    module's ``memory:<source>:<index>`` math depends on file cards first."""
    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("file fact", encoding="utf-8")
    _patch_active_provider(
        monkeypatch, "fakemem", _FakeProvider([{"body": "provider fact"}])
    )

    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    sources = [c["source"] for c in graph["memory"]]
    assert sources.index("memory") < sources.index("fakemem")
    # Provider node exists, carries provider source, and is memory-kind.
    node = next(n for n in graph["nodes"] if n["memorySource"] == "fakemem")
    assert node["kind"] == "memory"
    assert node["label"] == "provider fact"
    # Node ids stay positional over the combined list.
    assert node["id"] == f"memory:fakemem:{sources.index('fakemem')}"
    # Cluster count covers file + provider cards alike.
    mem_cluster = next(c for c in graph["clusters"] if c["category"] == "memory")
    assert mem_cluster["count"] == len(graph["memory"]) == 2
