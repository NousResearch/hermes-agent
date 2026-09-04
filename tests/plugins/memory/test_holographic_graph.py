"""Native directed-graph contract for the holographic MemoryStore.

Every test uses a temporary SQLite database. No configured Hermes store is opened.
"""

import json
import multiprocessing
import sqlite3

import pytest

from plugins.memory.holographic.store import MemoryStore


def _concurrent_edge_add(db_path, source, target, start, results):
    store = MemoryStore(db_path)
    try:
        start.wait(timeout=10)
        edge = store.add_edge(source, target, "progresses_to")
        results.put(("ok", edge["edge_id"]))
    except BaseException as exc:  # noqa: BLE001 - child result is asserted
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        store.close()


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


def _facts(store: MemoryStore, category="exercise") -> tuple[int, int, int, int]:
    return tuple(
        store.add_fact(name, category=category)
        for name in ("Base", "Progression", "Advanced", "Variation")
    )


def test_schema_is_additive_indexed_and_id_preserving(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO facts (fact_id, content, category) VALUES
            (1287, 'Roll To Squat', 'exercise'),
            (1294, 'Exercise Database', 'exercise');
        """
    )
    conn.commit()
    conn.close()

    with MemoryStore(db_path) as store:
        assert [
            tuple(row)
            for row in store._conn.execute(
                "SELECT fact_id, content FROM facts ORDER BY fact_id"
            ).fetchall()
        ] == [(1287, "Roll To Squat"), (1294, "Exercise Database")]
        columns = {
            row[1] for row in store._conn.execute("PRAGMA table_info(edges)").fetchall()
        }
        assert {
            "edge_id", "source_fact_id", "target_fact_id", "relation_type",
            "metadata_json", "status", "archived_at", "archive_reason",
        } <= columns
        indexes = {
            row[1] for row in store._conn.execute("PRAGMA index_list(edges)").fetchall()
        }
        assert {
            "idx_edges_active_unique", "idx_edges_outgoing",
            "idx_edges_incoming", "idx_edges_status",
        } <= indexes
        assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_add_archive_and_readd_preserve_history(tmp_path):
    with MemoryStore(tmp_path / "graph.db") as store:
        source, target, _, _ = _facts(store)
        first = store.add_edge(
            source, target, "progresses-to", metadata={"seed": "test"}
        )
        assert first["created"] is True
        assert first["relation_type"] == "progresses_to"
        assert first["metadata"] == {"seed": "test"}

        duplicate = store.add_edge(source, target, "progresses_to")
        assert duplicate["edge_id"] == first["edge_id"]
        assert duplicate["created"] is False
        assert duplicate["reactivated"] is False

        archived = store.archive_edge(first["edge_id"], reason="superseded")
        assert archived["status"] == "archived"
        assert archived["deleted"] is False
        assert store.archive_edge(first["edge_id"])["already"] is True
        assert store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_id = ?", (first["edge_id"],)
        ).fetchone()[0] == 1

        readded = store.add_edge(source, target, "progresses_to")
        assert readded["edge_id"] != first["edge_id"]
        assert readded["created"] is True
        assert readded["reactivated"] is False
        assert readded["previous_archived_edge_id"] == first["edge_id"]
        old = store._conn.execute(
            "SELECT status, archived_at, archive_reason FROM edges WHERE edge_id = ?",
            (first["edge_id"],),
        ).fetchone()
        assert tuple(old) == ("archived", archived["archived_at"], "superseded")
        assert store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2

        store.archive_edge(readded["edge_id"], reason="second history record")
        third = store.add_edge(source, target, "progresses_to")
        assert third["previous_archived_edge_id"] == readded["edge_id"]
        history = store._conn.execute(
            "SELECT edge_id, status, archive_reason FROM edges ORDER BY edge_id"
        ).fetchall()
        assert [tuple(row) for row in history] == [
            (first["edge_id"], "archived", "superseded"),
            (readded["edge_id"], "archived", "second history record"),
            (third["edge_id"], "active", None),
        ]


def test_add_edge_validates_endpoints_relation_and_metadata(tmp_path):
    with MemoryStore(tmp_path / "graph.db") as store:
        source, target, _, _ = _facts(store)
        with pytest.raises(KeyError, match="not found"):
            store.add_edge(source, 999999, "progresses_to")
        with pytest.raises(ValueError, match="relation_type"):
            store.add_edge(source, target, "not a relation")
        with pytest.raises(ValueError, match="metadata"):
            store.add_edge(source, target, "progresses_to", metadata=["bad"])
        for invalid_id in (True, False, 1.0, "1.0", "1e0", " 1", "+1"):
            with pytest.raises(ValueError, match="integer"):
                store.add_edge(invalid_id, target, "progresses_to")
        with pytest.raises(ValueError, match="between"):
            store.add_edge(2**63, target, "progresses_to")


def test_neighbors_preserve_direction_and_type_filters(tmp_path):
    with MemoryStore(tmp_path / "graph.db") as store:
        base, progression, advanced, variation = _facts(store)
        incoming = store.add_edge(base, progression, "progresses_to")
        outgoing = store.add_edge(progression, advanced, "progresses_to")
        store.add_edge(progression, variation, "variation_of")

        both = store.neighbors(progression, direction="both")
        assert [item["fact"]["fact_id"] for item in both["neighbors"]] == [
            base, advanced, variation
        ]
        assert [item["direction"] for item in both["neighbors"]] == ["in", "out", "out"]

        typed = store.neighbors(
            progression, relation_types=["progresses_to"], direction="both"
        )
        assert [item["edge"]["edge_id"] for item in typed["neighbors"]] == [
            incoming["edge_id"], outgoing["edge_id"]
        ]

        store.archive_edge(outgoing["edge_id"])
        assert store.neighbors(progression, direction="out")["count"] == 1
        assert store.neighbors(
            progression, direction="out", active_only=False
        )["count"] == 2


def test_traverse_is_cycle_safe_depth_bounded_typed_and_capped(tmp_path):
    with MemoryStore(tmp_path / "graph.db") as store:
        base, progression, advanced, variation = _facts(store)
        store.add_edge(base, progression, "progresses_to")
        store.add_edge(progression, advanced, "progresses_to")
        store.add_edge(advanced, base, "progresses_to")
        store.add_edge(progression, variation, "variation_of")

        depth_zero = store.traverse(base, max_depth=0)
        assert [node["fact"]["fact_id"] for node in depth_zero["nodes"]] == [base]
        assert depth_zero["edges"] == []

        depth_one = store.traverse(base, max_depth=1)
        assert [node["fact"]["fact_id"] for node in depth_one["nodes"]] == [
            base, progression
        ]

        typed = store.traverse(
            base, relation_types="progresses_to", max_depth=10
        )
        assert [node["fact"]["fact_id"] for node in typed["nodes"]] == [
            base, progression, advanced
        ]
        assert typed["depth_by_fact_id"] == {base: 0, progression: 1, advanced: 2}
        assert typed["truncated"] is False

        capped = store.traverse(base, max_depth=10, max_nodes=2)
        assert capped["node_count"] == 2
        assert capped["truncated"] is True


def test_list_subgraph_is_category_induced_and_keeps_isolated_nodes(tmp_path):
    with MemoryStore(tmp_path / "graph.db") as store:
        base, progression, isolated, _ = _facts(store)
        outside = store.add_fact("Outside", category="project")
        internal = store.add_edge(base, progression, "progresses_to")
        store.add_edge(progression, outside, "prerequisite")

        graph = store.list_subgraph("exercise")
        assert {node["fact_id"] for node in graph["nodes"]} == {
            base, progression, isolated, _
        }
        assert [edge["edge_id"] for edge in graph["edges"]] == [internal["edge_id"]]

        connected = store.list_subgraph("exercise", include_isolated=False)
        assert {node["fact_id"] for node in connected["nodes"]} == {base, progression}

        limited = store.list_subgraph("exercise", limit_nodes=2)
        assert limited["node_count"] == 2
        assert limited["truncated"] is True


def test_category_change_rebuilds_source_and_destination_banks(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "graph.db") as store:
        fact_id = store.add_fact("Category move", category="hybrid_movement")
        rebuilt = []
        monkeypatch.setattr(store, "_rebuild_bank", rebuilt.append)
        assert store.update_fact(fact_id, category="exercise") is True
        assert rebuilt == ["exercise", "hybrid_movement"]


def test_cross_process_duplicate_add_is_idempotent(tmp_path):
    db_path = tmp_path / "graph.db"
    with MemoryStore(db_path) as store:
        source, target, _, _ = _facts(store)

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    workers = [
        context.Process(
            target=_concurrent_edge_add,
            args=(str(db_path), source, target, start, results),
        )
        for _ in range(6)
    ]
    for worker in workers:
        worker.start()
    start.set()
    outcomes = [results.get(timeout=20) for _ in workers]
    for worker in workers:
        worker.join(timeout=20)
        assert worker.exitcode == 0

    assert {kind for kind, _ in outcomes} == {"ok"}
    assert len({edge_id for _, edge_id in outcomes}) == 1
    with MemoryStore(db_path) as store:
        assert store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE status='active'"
        ).fetchone()[0] == 1


def test_multiple_instances_share_graph_writes(tmp_path):
    db_path = tmp_path / "graph.db"
    first = MemoryStore(db_path)
    second = MemoryStore(db_path)
    try:
        source, target, _, _ = _facts(first)
        edge = first.add_edge(source, target, "progresses_to")
        assert second.neighbors(source)["neighbors"][0]["edge"]["edge_id"] == edge["edge_id"]
        assert json.loads(
            second._conn.execute(
                "SELECT metadata_json FROM edges WHERE edge_id = ?", (edge["edge_id"],)
            ).fetchone()[0]
        ) == {}
    finally:
        first.close()
        second.close()
