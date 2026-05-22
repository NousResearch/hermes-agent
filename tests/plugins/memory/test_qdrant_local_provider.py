"""Tests for the local Qdrant-backed memory provider."""

from __future__ import annotations

import json
import sqlite3

from agent.memory_manager import MemoryManager
from plugins.memory import load_memory_provider
from plugins.memory.qdrant_local import QdrantLocalMemoryProvider


def _registry(tmp_path):
    return tmp_path / "memory" / "qdrant_local" / "registry.sqlite"


def _counts(registry):
    with sqlite3.connect(registry) as conn:
        return {
            "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "chunks": conn.execute("SELECT COUNT(*) FROM chunks WHERE deleted_at IS NULL").fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM ingest_events").fetchone()[0],
        }


def test_qdrant_local_provider_loads_with_no_tool_schema_bloat():
    provider = load_memory_provider("qdrant_local")

    assert provider is not None
    assert provider.name == "qdrant_local"
    assert provider.is_available() is True
    assert provider.get_tool_schemas() == []
    assert provider.system_prompt_block() == ""


def test_qdrant_local_initialize_uses_profile_scoped_storage(tmp_path):
    provider = load_memory_provider("qdrant_local")
    assert provider is not None

    provider.initialize(
        session_id="session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_context="primary",
        agent_identity="argus-test",
    )
    try:
        provider_dir = tmp_path / "memory" / "qdrant_local"
        registry = provider_dir / "registry.sqlite"
        config = provider_dir / "qdrant_local.json"

        assert provider_dir.is_dir()
        assert registry.is_file()
        assert config.is_file()
        saved = json.loads(config.read_text(encoding="utf-8"))
        assert saved["schema_version"] == 1
        assert saved["mode"] == "shadow"

        with sqlite3.connect(registry) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
        assert {"documents", "chunks", "chunks_fts", "ingest_events", "audit_events"} <= tables
    finally:
        provider.shutdown()


def test_qdrant_local_lifecycle_is_safe_noop_before_vector_mode(tmp_path):
    provider = load_memory_provider("qdrant_local")
    assert provider is not None
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="telegram")
    try:
        provider.sync_turn("hello", "world", session_id="session-1")
        provider.queue_prefetch("hello", session_id="session-1")
        assert provider.prefetch("hello", session_id="session-1") == ""
        provider.on_session_switch("session-2", parent_session_id="session-1", reset=False)
        provider.on_pre_compress([{"role": "user", "content": "old"}])
        provider.on_session_end([{"role": "assistant", "content": "done"}])
        provider.on_memory_write("add", "memory", "stable fact", metadata={"source": "test"})
    finally:
        provider.shutdown()


def test_qdrant_local_integrates_with_memory_manager(tmp_path):
    provider = load_memory_provider("qdrant_local")
    assert provider is not None

    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all(session_id="session-1", hermes_home=str(tmp_path), platform="cli")

    try:
        assert manager.get_provider("qdrant_local") is provider
        assert manager.get_all_tool_schemas() == []
        assert manager.build_system_prompt() == ""
        assert manager.prefetch_all("anything", session_id="session-1") == ""
        manager.queue_prefetch_all("anything", session_id="session-1")
        manager.sync_all("user", "assistant", session_id="session-1")
    finally:
        manager.shutdown_all()


def test_qdrant_local_sync_turn_indexes_chunks_idempotently(tmp_path):
    provider = QdrantLocalMemoryProvider({"mode": "fts_only"})
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        provider.sync_turn(
            "remember project zebra uses blue routers",
            "noted: project zebra uses blue routers",
            session_id="session-1",
        )
        provider.sync_turn(
            "remember project zebra uses blue routers",
            "noted: project zebra uses blue routers",
            session_id="session-1",
        )

        assert _counts(_registry(tmp_path)) == {"documents": 1, "chunks": 2, "events": 2}
    finally:
        provider.shutdown()


def test_qdrant_local_fts_only_prefetch_returns_cited_context(tmp_path):
    provider = QdrantLocalMemoryProvider({"mode": "fts_only"})
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        provider.sync_turn(
            "project zebra needs blue router firmware",
            "I will remember the blue router firmware detail for project zebra.",
            session_id="session-1",
        )
        provider.queue_prefetch("blue router", session_id="future-session")
        recall = provider.prefetch("blue router", session_id="future-session")

        assert "Local private memory recall" in recall
        assert "project zebra" in recall
        assert "blue router" in recall
        assert "source=session" in recall
    finally:
        provider.shutdown()


def test_qdrant_local_shadow_mode_indexes_but_does_not_inject(tmp_path):
    provider = QdrantLocalMemoryProvider({"mode": "shadow"})
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        provider.sync_turn("project shadow has silent memory", "stored silently", session_id="session-1")
        provider.queue_prefetch("project shadow", session_id="session-2")

        assert _counts(_registry(tmp_path))["chunks"] == 2
        assert provider.prefetch("project shadow", session_id="session-2") == ""
    finally:
        provider.shutdown()


def test_qdrant_local_blocks_secret_like_content_from_chunks(tmp_path):
    provider = QdrantLocalMemoryProvider({"mode": "fts_only"})
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        provider.sync_turn("token sk-proj-abcdefghijklmnopqrstuvwxyz1234567890FAKEFAKE", "do not store", session_id="session-1")

        assert _counts(_registry(tmp_path))["chunks"] == 0
        with sqlite3.connect(_registry(tmp_path)) as conn:
            audit_results = conn.execute(
                "SELECT result FROM audit_events WHERE event_type = 'ingest_blocked'"
            ).fetchall()
        assert audit_results
    finally:
        provider.shutdown()


def test_qdrant_local_hash_embeddings_are_deterministic_and_normalized():
    provider = QdrantLocalMemoryProvider({"embedding_dimensions": 64})

    first = provider.embed_text("project zebra blue router")
    second = provider.embed_text("project zebra blue router")
    other = provider.embed_text("completely different memory")

    assert first == second
    assert first != other
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_qdrant_local_vector_mode_rebuilds_projection_and_searches(tmp_path):
    provider = QdrantLocalMemoryProvider({"mode": "vector", "embedding_dimensions": 64})
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        provider.sync_turn(
            "project vector-lantern stores amber packet routes",
            "noted amber packet routes for project vector-lantern",
            session_id="session-1",
        )

        rebuilt = provider.rebuild_vector_index()
        status = provider.vector_status()
        provider.queue_prefetch("amber packet routes", session_id="session-2")
        recall = provider.prefetch("amber packet routes", session_id="session-2")

        assert rebuilt["success"] is True
        assert rebuilt["indexed_chunks"] == 2
        assert status["qdrant_available"] is True
        assert status["collection"] == "hermes_qdrant_local"
        assert status["vector_size"] == 64
        assert "vector-lantern" in recall
        assert "Local private memory recall" in recall
    finally:
        provider.shutdown()


def test_qdrant_local_rebuild_is_repeatable_without_duplicate_points(tmp_path):
    provider = QdrantLocalMemoryProvider({"mode": "vector", "embedding_dimensions": 32})
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        provider.sync_turn("project repeatable has stable points", "stable points stored", session_id="session-1")

        first = provider.rebuild_vector_index()
        second = provider.rebuild_vector_index()

        assert first["indexed_chunks"] == 2
        assert second["indexed_chunks"] == 2
        assert provider.vector_status()["points_count"] == 2
    finally:
        provider.shutdown()

def test_qdrant_local_explicit_tools_are_opt_in_and_bounded(tmp_path):
    default_provider = QdrantLocalMemoryProvider({"mode": "vector"})
    assert default_provider.get_tool_schemas() == []

    provider = QdrantLocalMemoryProvider({"mode": "vector", "enable_tools": True, "embedding_dimensions": 32})
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        schemas = provider.get_tool_schemas()
        names = {schema["name"] for schema in schemas}

        assert names == {"vector_memory_status", "vector_memory_rebuild", "vector_memory_search"}
        assert all(schema["parameters"]["additionalProperties"] is False for schema in schemas)
    finally:
        provider.shutdown()


def test_qdrant_local_tool_calls_return_json_and_hydrate_from_sqlite(tmp_path):
    provider = QdrantLocalMemoryProvider({"mode": "vector", "enable_tools": True, "embedding_dimensions": 32})
    provider.initialize(session_id="session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        provider.sync_turn(
            "project tool-lantern stores cyan gateway routes",
            "noted cyan gateway routes for project tool-lantern",
            session_id="session-1",
        )

        rebuild = json.loads(provider.handle_tool_call("vector_memory_rebuild", {}))
        status = json.loads(provider.handle_tool_call("vector_memory_status", {}))
        search = json.loads(provider.handle_tool_call("vector_memory_search", {"query": "cyan gateway routes", "limit": 3}))

        assert rebuild["success"] is True
        assert status["success"] is True
        assert status["points_count"] == 2
        assert search["success"] is True
        assert search["backend"] in {"qdrant", "fts"}
        assert search["results"]
        assert any("tool-lantern" in result["content"] for result in search["results"])
        assert all("vector" not in result for result in search["results"])
    finally:
        provider.shutdown()

