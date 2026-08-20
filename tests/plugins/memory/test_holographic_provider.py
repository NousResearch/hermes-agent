"""Regression tests for the bundled holographic memory provider."""

import json
from pathlib import Path

from plugins.memory.holographic import HolographicMemoryProvider, _resolve_profile_db_path
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def test_profile_shorthand_db_path_resolves_under_hermes_home(tmp_path):
    hermes_home = tmp_path / "profiles" / "worker"

    assert _resolve_profile_db_path("~/.hermes/memory_store.db", str(hermes_home)) == str(
        hermes_home / "memory_store.db"
    )
    assert _resolve_profile_db_path("~/.hermes", str(hermes_home)) == str(hermes_home)


def test_non_hermes_tilde_path_is_left_for_expanduser(tmp_path):
    hermes_home = tmp_path / "profiles" / "worker"

    assert _resolve_profile_db_path("~/custom/memory_store.db", str(hermes_home)) == "~/custom/memory_store.db"


def test_provider_initialize_uses_profile_safe_shorthand_path(tmp_path, monkeypatch):
    hermes_home = tmp_path / "profiles" / "responsible_memory_rollout"
    os_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(os_home))

    provider = HolographicMemoryProvider(config={"db_path": "~/.hermes/memory_store.db"})
    provider.initialize("session-1")

    try:
        assert provider._store is not None
        assert provider._store.db_path == hermes_home / "memory_store.db"
        assert not (os_home / ".hermes" / "memory_store.db").exists()
    finally:
        provider.shutdown()


def test_search_fallback_recovers_partial_overlap_when_strict_fts_and_query_misses(tmp_path):
    store = MemoryStore(tmp_path / "memory_store.db", default_trust=0.8)
    try:
        fact_id = store.add_fact(
            "Matrix E2EE room key sharing requires recovery key import for the bot device.",
            category="project",
            tags="matrix,e2ee,device,recovery",
        )
        store.add_fact(
            "Kanban blocked means human intervention is required.",
            category="project",
            tags="kanban,status",
        )
        retriever = FactRetriever(store, hrr_weight=0.0)

        # The token "session" is absent, so strict FTS5 AND matching returns no
        # rows. The deterministic overlap fallback should still surface the
        # Matrix fact instead of producing a false negative.
        results = retriever.search("Matrix E2EE device session recovery", category="project", limit=3)

        assert results
        assert results[0]["fact_id"] == fact_id
        assert "Matrix E2EE" in results[0]["content"]
    finally:
        store.close()


def test_probe_and_reason_suppress_unrelated_hrr_false_positives(tmp_path):
    store = MemoryStore(tmp_path / "memory_store.db", default_trust=0.5)
    try:
        store.add_fact(
            "Hermes Maintenance holographic rollout verification marker for task t_96dfd135.",
            category="project",
            tags="rollout,verification",
        )
        retriever = FactRetriever(store, min_hrr_score=0.35)

        assert retriever.probe("Lineage status", category="project") == []
        assert retriever.reason(["Lineage status", "working waiting"], category="project") == []
    finally:
        store.close()


def test_fact_store_tool_returns_empty_for_unrelated_probe(tmp_path):
    provider = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "default_trust": 0.5, "min_hrr_score": 0.35}
    )
    provider.initialize("session-1")
    try:
        provider._store.add_fact(
            "Hermes Maintenance holographic rollout verification marker for task t_96dfd135.",
            category="project",
            tags="rollout,verification",
        )

        result = json.loads(provider.handle_tool_call("fact_store", {"action": "probe", "entity": "Lineage status"}))

        assert result == {"results": [], "count": 0}
    finally:
        provider.shutdown()
