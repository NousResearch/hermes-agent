"""End-to-end tests for retrieval_count tracking on the fact_store tool path.

Issue #78801: retrieval_count was never incremented when the agent retrieved
facts, because every read path (FactRetriever search/probe/related/reason)
only SELECTed and the one UPDATE in the codebase (the old store.search_facts)
had zero callers. The memory janitor therefore flagged healthy facts as
"stale / never retrieved".

Fix shape: a locked MemoryStore.bump_retrieval(ids) helper, called from the
tool-handler layer (HolographicMemoryProvider._track_retrieval) after each
successful agent retrieval. Counting at the handler boundary means:

- search/probe/related/reason count, regardless of internal numpy/HRR or FTS5
  fallback paths,
- auto-prefetch (which calls retriever.search() directly) is intentionally
  NOT counted,
- contradict/list (diagnostics/browse) are intentionally NOT counted,
- a failing bump never breaks the retrieval result (advisory telemetry).

These tests drive the real handle_tool_call dispatch and assert exact
retrieval_count values in the DB.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic import HolographicMemoryProvider


@pytest.fixture
def provider(tmp_path):
    """Provider with a fresh DB and a few facts, initialized like a real session."""
    db_path = tmp_path / "memory_store.db"
    prov = HolographicMemoryProvider(config={"db_path": str(db_path), "hrr_dim": 64})
    prov.initialize("test-session")
    prov.handle_tool_call("fact_store", {"action": "add", "content": "The deployment rollback failed because of stale migration state.", "category": "project"})
    prov.handle_tool_call("fact_store", {"action": "add", "content": "Compaction settings tuned to 0.85 threshold.", "category": "tool"})
    prov.handle_tool_call("fact_store", {"action": "add", "content": "Jordan Miller and Alex Smith pair on database tuning.", "category": "project"})
    yield prov
    prov.shutdown()


def _counts(prov) -> dict:
    """fact_id -> retrieval_count straight from the DB."""
    return {r["fact_id"]: r["retrieval_count"] for r in prov._store.list_facts()}


def _ids(prov, raw: str) -> list[int]:
    return [r["fact_id"] for r in json.loads(raw)["results"]]


class TestRetrievalCountToolPath:
    def test_add_does_not_bump(self, provider):
        counts = _counts(provider)
        assert set(counts.values()) == {0}

    def test_search_bumps_returned_facts_exactly_once(self, provider):
        raw = provider.handle_tool_call("fact_store", {"action": "search", "query": "deployment rollback"})
        returned = _ids(provider, raw)
        assert returned, "search should return the deployment fact"
        counts = _counts(provider)
        for fact_id in returned:
            assert counts[fact_id] == 1
        # the other facts were not returned, so they stay at zero
        for fact_id, count in counts.items():
            if fact_id not in returned:
                assert count == 0

    def test_repeated_search_accumulates(self, provider):
        provider.handle_tool_call("fact_store", {"action": "search", "query": "deployment"})
        provider.handle_tool_call("fact_store", {"action": "search", "query": "deployment"})
        returned = _ids(provider, provider.handle_tool_call("fact_store", {"action": "search", "query": "deployment"}))
        counts = _counts(provider)
        for fact_id in returned:
            assert counts[fact_id] == 3

    def test_probe_bumps(self, provider):
        raw = provider.handle_tool_call("fact_store", {"action": "probe", "entity": "Jordan Miller"})
        returned = _ids(provider, raw)
        assert returned
        counts = _counts(provider)
        for fact_id in returned:
            assert counts[fact_id] == 1

    def test_related_bumps(self, provider):
        raw = provider.handle_tool_call("fact_store", {"action": "related", "entity": "Jordan Miller"})
        returned = _ids(provider, raw)
        assert returned
        counts = _counts(provider)
        for fact_id in returned:
            assert counts[fact_id] == 1

    def test_reason_bumps(self, provider):
        raw = provider.handle_tool_call("fact_store", {"action": "reason", "entities": ["Jordan Miller", "Alex Smith"]})
        returned = _ids(provider, raw)
        assert returned
        counts = _counts(provider)
        for fact_id in returned:
            assert counts[fact_id] == 1

    def test_list_does_not_bump(self, provider):
        provider.handle_tool_call("fact_store", {"action": "list"})
        assert set(_counts(provider).values()) == {0}

    def test_prefetch_does_not_bump(self, provider):
        provider.prefetch("deployment rollback")
        assert set(_counts(provider).values()) == {0}

    def test_empty_search_is_a_noop(self, provider):
        provider.handle_tool_call("fact_store", {"action": "search", "query": "zzz-no-match-zzz"})
        assert set(_counts(provider).values()) == {0}

    def test_contradict_does_not_bump(self, provider):
        provider.handle_tool_call("fact_store", {"action": "contradict"})
        assert set(_counts(provider).values()) == {0}

    def test_failing_bump_never_breaks_retrieval(self, provider, monkeypatch):
        """Advisory contract: even a hard bump failure must leave the result intact."""
        def boom(fact_ids):
            raise sqlite3.Error("simulated store failure")

        monkeypatch.setattr(provider._store, "bump_retrieval", boom)
        raw = provider.handle_tool_call("fact_store", {"action": "search", "query": "deployment rollback"})
        returned = _ids(provider, raw)
        assert returned, "search result must survive a failed counter bump"

    def test_bump_survives_oversized_id_list(self, provider):
        """The batched UPDATE must handle id lists beyond SQLite's variable limit."""
        fact_id = _ids(provider, provider.handle_tool_call(
            "fact_store", {"action": "search", "query": "deployment rollback"}))[0]
        provider._store.bump_retrieval(list(range(10, 1210)) + [fact_id])  # mostly nonexistent ids
        counts = _counts(provider)
        assert counts[fact_id] == 2  # one from search, one from the direct bump
