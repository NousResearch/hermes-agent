import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from plugins.memory import discover_memory_providers, load_memory_provider
from plugins.memory.subconscious.consolidate import run_consolidation
from plugins.memory.subconscious.store import SubconsciousStore


def test_subconscious_provider_discovered_and_available():
    providers = {name: available for name, _desc, available in discover_memory_providers()}
    assert providers.get("subconscious") is True
    provider = load_memory_provider("subconscious")
    assert provider is not None
    assert provider.name == "subconscious"
    assert provider.is_available()


def test_store_layers_search_counts_and_status(tmp_path):
    store = SubconsciousStore(tmp_path / "subconscious.db")
    try:
        fact_id = store.add_memory(
            "semantic",
            "Mikhail prefers concise reports",
            tags=["user", "preference"],
            metadata={"platform": "telegram", "chat_id": "-1003772186616", "topic_id": "1347"},
        )
        proc_id = store.add_memory(
            "procedural",
            "Run pytest before reporting implementation complete",
            tags="tests",
            metadata={"platform": "telegram", "chat_id": "-1003772186616", "topic_id": "1347"},
        )
        store.add_edge(fact_id, proc_id, relation="supports", weight=0.9, source="test")
        assert store.counts()["semantic"] == 1
        assert store.counts()["procedural"] == 1
        results = store.search("Mikhail", limit=5)
        assert len(results) == 1
        assert results[0]["layer"] == "semantic"
        duplicate_id = store.add_memory(
            "semantic",
            "  Mikhail   prefers concise reports  ",
            tags=["user", "preference"],
            metadata={"platform": "telegram", "chat_id": "-1003772186616", "topic_id": "1347"},
        )
        assert duplicate_id == fact_id
        assert store.counts()["semantic"] == 1
        scoped = store.search("Mikhail", topic_id="1347", platform="telegram", limit=5)
        assert len(scoped) == 1
        assert store.search("Mikhail", topic_id="642", limit=5) == []
        hybrid = store.hybrid_search("Mikhail", topic_id="1347", limit=5)
        assert {row["retrieval"] for row in hybrid} == {"direct", "graph"}
        assert any(row.get("relation") == "supports" for row in hybrid)
        assert all("score" in row for row in hybrid)
        candidates = store.skill_candidates()
        assert candidates[0]["layer"] == "procedural"
        status = store.status()
        assert Path(status["db_path"]).exists()
        assert status["edge_count"] == 1
        assert status["duplicate_groups"] == 0
    finally:
        store.close()


def test_store_metrics_expire_and_conflict_detection(tmp_path):
    store = SubconsciousStore(tmp_path / "subconscious.db")
    try:
        working_id = store.add_memory("working", "Pending follow-up should expire", tags=["open-loop"])
        procedural_id = store.add_memory("procedural", "Run old cleanup workflow", tags=["workflow"])
        semantic_id = store.add_memory("semantic", "Bud must use decision preflight", confidence=0.8)
        semantic_conflict_id = store.add_memory("semantic", "Bud must not use decision preflight", confidence=0.75)
        should_id = store.add_memory("semantic", "Hermes should coordinate with Bud after approval", confidence=0.8)
        should_noise_id = store.add_memory("semantic", "Routine reports should not mention Hindsight unless relevant", confidence=0.8)
        store.add_memory(
            "semantic",
            '{"success": false, "error": "Bud must not use decision preflight"}',
            tags=["hygiene:noisy"],
            confidence=0.9,
        )
        store.add_memory("semantic", "Bud must not use decision preflight in a noisy draft", confidence=0.4)
        store._conn.execute(
            """
            INSERT INTO detected_conflicts
            (memory_id_1, memory_id_2, conflict_type, severity, detected_at)
            VALUES (?, ?, 'should_vs_should_not', 'high', ?)
            """,
            (should_id, should_noise_id, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        store._conn.commit()

        rows = store._conn.execute(
            "SELECT id, ttl_days FROM memories WHERE id IN (?, ?, ?)",
            (working_id, procedural_id, semantic_id),
        ).fetchall()
        ttl_by_id = {int(row["id"]): row["ttl_days"] for row in rows}
        assert ttl_by_id[working_id] == 7
        assert ttl_by_id[procedural_id] == 90
        assert ttl_by_id[semantic_id] is None

        old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat(timespec="seconds")
        store._conn.execute(
            "UPDATE memories SET created_at=?, updated_at=?, last_seen_at=? WHERE id IN (?, ?)",
            (old, old, old, working_id, procedural_id),
        )
        store._conn.commit()

        metrics = store.capture_metrics_snapshot()
        assert metrics["stale_memory_count"] == 2
        assert metrics["total_memories"] == 8

        expired = store.expire_stale_memories()
        assert expired["expired_count"] == 2
        assert set(expired["expired_ids"]) == {working_id, procedural_id}

        conflicts = store.detect_conflicts()
        assert conflicts["new_conflict_count"] == 1
        assert conflicts["conflict_count"] == 1
        assert conflicts["skipped_memory_count"] == 2
        assert conflicts["rejected_by_gate_count"] == 1
        assert conflicts["stale_resolved_count"] == 1
        assert conflicts["conflicts"][0]["memory_id_1"] == semantic_id
        assert conflicts["conflicts"][0]["memory_id_2"] == semantic_conflict_id
        assert store.status()["conflict_count"] == 1
    finally:
        store.close()


def test_provider_tool_status_add_search_and_memory_write(tmp_path):
    provider = load_memory_provider("subconscious")
    assert provider is not None
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="test", agent_context="primary")
    try:
        add_result = json.loads(provider.handle_tool_call("subconscious", {
            "action": "add",
            "layer": "semantic",
            "content": "Agentic Stack uses Hermes as coordinator",
            "tags": ["agentic-stack"],
            "topic_id": "1347",
            "chat_id": "-1003772186616",
            "platform": "telegram",
        }))
        assert add_result["success"] is True
        proc_result = json.loads(provider.handle_tool_call("subconscious", {
            "action": "add",
            "layer": "procedural",
            "content": "Bud executes implementation after Hermes approval",
            "tags": ["agentic-stack"],
            "topic_id": "1347",
            "chat_id": "-1003772186616",
            "platform": "telegram",
        }))
        assert proc_result["success"] is True
        link_result = json.loads(provider.handle_tool_call("subconscious", {
            "action": "link",
            "source_id": add_result["id"],
            "target_id": proc_result["id"],
            "relation": "delegates_to",
            "weight": 0.8,
        }))
        assert link_result["success"] is True

        search_result = json.loads(provider.handle_tool_call("subconscious", {
            "action": "search",
            "query": "coordinator",
            "topic_id": "1347",
        }))
        assert search_result["success"] is True
        assert search_result["results"][0]["layer"] == "semantic"
        hybrid_result = json.loads(provider.handle_tool_call("subconscious", {
            "action": "hybrid_search",
            "query": "coordinator",
            "topic_id": "1347",
        }))
        assert hybrid_result["success"] is True
        assert {row["retrieval"] for row in hybrid_result["results"]} == {"direct", "graph"}
        related_result = json.loads(provider.handle_tool_call("subconscious", {
            "action": "related",
            "memory_id": add_result["id"],
        }))
        assert related_result["success"] is True
        assert related_result["results"][0]["relation"] == "delegates_to"
        skill_candidates = json.loads(provider.handle_tool_call("subconscious", {
            "action": "skill_candidates",
        }))
        assert skill_candidates["success"] is True
        assert skill_candidates["results"]

        provider.on_memory_write("add", "user", "User prefers concise reports", metadata={"session_id": "s2"})
        status_result = json.loads(provider.handle_tool_call("subconscious", {"action": "status"}))
        assert status_result["success"] is True
        assert status_result["counts"]["semantic"] >= 2

        provider.handle_tool_call("subconscious", {
            "action": "add",
            "layer": "semantic",
            "content": "Hermes always coordinates review",
        })
        provider.handle_tool_call("subconscious", {
            "action": "add",
            "layer": "semantic",
            "content": "Hermes never coordinates review",
        })
        conflict_result = json.loads(provider.handle_tool_call("subconscious", {"action": "conflicts"}))
        assert conflict_result["success"] is True
        assert conflict_result["conflict_count"] >= 1
        metrics_result = json.loads(provider.handle_tool_call("subconscious", {"action": "metrics"}))
        assert metrics_result["success"] is True
        assert metrics_result["conflict_count"] >= 1
        expire_result = json.loads(provider.handle_tool_call("subconscious", {"action": "expire"}))
        assert expire_result["success"] is True
    finally:
        provider.shutdown()


def test_consolidation_is_deterministic_without_state_db(tmp_path):
    result = run_consolidation(tmp_path, run_key="test-run")
    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["stats"]["messages_scanned"] == 0
    assert result["stats"]["metrics"]["total_memories"] == 0
    assert result["stats"]["expire"]["expired_count"] == 0
    assert result["stats"]["conflicts"]["conflict_count"] == 0

    second = run_consolidation(tmp_path, run_key="test-run")
    assert second["success"] is True
    assert second["status"] == "skipped"
