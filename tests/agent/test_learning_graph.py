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
    # Provider gate: the payload always carries memoryProvider (None here — no
    # active external provider in the temp home) so the desktop never has to
    # infer the provider from node presence.
    assert "memoryProvider" in graph
    assert graph["memoryProvider"] is None


def test_memory_provider_name_reflects_active_provider(monkeypatch, tmp_path):
    """memoryProvider echoes the active provider name (lowercased), or None."""
    home = tmp_path / ".hermes"
    home.mkdir()

    import plugins.memory as pm

    # Active provider present → its name flows to the payload, normalized.
    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "Honcho")
    token = set_hermes_home_override(home)
    try:
        assert learning_graph.build_learning_graph()["memoryProvider"] == "honcho"
    finally:
        reset_hermes_home_override(token)

    # No active provider → None (never inferred from nodes).
    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: None)
    token = set_hermes_home_override(home)
    try:
        assert learning_graph.build_learning_graph()["memoryProvider"] is None
    finally:
        reset_hermes_home_override(token)


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


def test_provider_card_level_propagates_to_node(monkeypatch, tmp_path):
    """Honcho's conclusion 'level' must flow card → node as ``memoryLevel`` so
    the desktop can separate true memories ('explicit') from derived
    conclusions ('inductive'/'deductive'). A card without a level omits the
    field entirely (older backend → treated as a plain memory downstream)."""
    _patch_active_provider(
        monkeypatch,
        "honcho",
        _FakeProvider(
            [
                {"body": "toby texts himself instructions", "level": "explicit"},
                {"body": "toby values reproducibility", "level": "INDUCTIVE"},
                {"body": "a fact from an older backend"},  # no level
            ]
        ),
    )

    cards = learning_graph._provider_memory_cards()
    # Level is normalized to lowercase; absent stays absent (no empty string).
    assert cards[0]["level"] == "explicit"
    assert cards[1]["level"] == "inductive"
    assert "level" not in cards[2]

    home = tmp_path / ".hermes"
    home.mkdir()
    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    mem_nodes = [n for n in graph["nodes"] if n["kind"] == "memory"]
    by_level = {n["label"][:20]: n.get("memoryLevel") for n in mem_nodes}
    # explicit + inductive carry through; the level-less card has no memoryLevel.
    assert "explicit" in by_level.values()
    assert "inductive" in by_level.values()
    levelless = [n for n in mem_nodes if "older backend" in n["label"]]
    assert levelless and "memoryLevel" not in levelless[0]


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


def test_provider_session_provenance_reaches_graph_nodes(tmp_path, monkeypatch):
    """A provider card's session_id must surface as ``sessionId`` on its graph
    node (the journey drill-down resolves the originating conversation from
    it), while cards without one — and file-based memory — stay bare."""
    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("file fact", encoding="utf-8")
    _patch_active_provider(
        monkeypatch,
        "fakemem",
        _FakeProvider(
            [
                {"body": "sourced fact", "session_id": "20260101_000000_abc123"},
                {"body": "orphan fact", "session_id": None},
            ]
        ),
    )

    token = set_hermes_home_override(home)
    try:
        graph = learning_graph.build_learning_graph()
    finally:
        reset_hermes_home_override(token)

    by_label = {n["label"]: n for n in graph["nodes"] if n["kind"] == "memory"}
    assert by_label["sourced fact"]["sessionId"] == "20260101_000000_abc123"
    assert "sessionId" not in by_label["orphan fact"]
    assert "sessionId" not in by_label["file fact"]


def test_card_origin_detection():
    """Origin is extensible by convention: explicit `origin` field wins, then
    the `<source>-import-…` session-id convention (any future importer that
    follows it — claude, gemini — needs zero code changes), else hermes."""
    origin = learning_graph._card_origin

    # Convention: session id prefix names the import source.
    assert origin({}, "chatgpt-import-6a2593e8") == "chatgpt"
    assert origin({}, "claude-import-abc") == "claude"
    assert origin({}, "gemini2-import-xyz") == "gemini2"

    # Hermes-born (per-session sync uses the Hermes session id).
    assert origin({}, "20260707_232752_9533a6") == "hermes"
    assert origin({}, "") == "hermes"

    # Explicit origin on the card overrides everything.
    assert origin({"origin": "Notion"}, "20260707_232752_9533a6") == "notion"
    assert origin({"origin": "slack"}, "chatgpt-import-abc") == "slack"


def test_provider_cards_carry_origin(monkeypatch):
    _patch_active_provider(
        monkeypatch,
        "honcho",
        _FakeProvider(
            [
                {"body": "imported fact", "session_id": "chatgpt-import-42"},
                {"body": "hermes fact", "session_id": "20260101_000000_abc123"},
                {"body": "orphan fact"},
            ]
        ),
    )

    by_body = {c["body"]: c for c in learning_graph._provider_memory_cards()}

    assert by_body["imported fact"]["origin"] == "chatgpt"
    assert by_body["hermes fact"]["origin"] == "hermes"
    assert by_body["orphan fact"]["origin"] == "hermes"
