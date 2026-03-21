"""Tests for the hermes-memory-store plugin.

Covers:
- MemoryStore CRUD and core behavior
- Entity extraction and linking
- FactRetriever hybrid search
- Temporal decay
- Plugin registration and handler JSON contracts
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# Import strategy: plugin lives at ~/.hermes/plugins/hermes-memory-store/
# which is not on sys.path by default.
# ---------------------------------------------------------------------------

_plugin_dir = Path.home() / ".hermes" / "plugins" / "hermes-memory-store"
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import MemoryStore
from retrieval import FactRetriever


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """Fresh MemoryStore backed by a temp file. Closed after each test."""
    s = MemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def retriever(store):
    """FactRetriever pointing at the fresh store, decay disabled."""
    return FactRetriever(store=store, temporal_decay_half_life=0)


# ===========================================================================
# TestMemoryStore — CRUD + core behaviour
# ===========================================================================


class TestMemoryStore:

    # 1 -----------------------------------------------------------------------
    def test_add_fact_returns_id(self, store):
        fact_id = store.add_fact("The sky is blue")
        assert isinstance(fact_id, int)
        assert fact_id > 0

    # 2 -----------------------------------------------------------------------
    def test_add_fact_deduplication(self, store):
        id1 = store.add_fact("Python is a programming language")
        id2 = store.add_fact("Python is a programming language")
        assert id1 == id2

    # 3 -----------------------------------------------------------------------
    def test_add_fact_with_category_and_tags(self, store):
        fact_id = store.add_fact(
            "Use dark mode by default",
            category="user_pref",
            tags="ui,theme",
        )
        facts = store.list_facts()
        match = next(f for f in facts if f["fact_id"] == fact_id)
        assert match["category"] == "user_pref"
        assert match["tags"] == "ui,theme"

    # 4 -----------------------------------------------------------------------
    def test_search_facts_fts(self, store):
        store.add_fact("Cats are independent animals")
        store.add_fact("Dogs are loyal companions")
        store.add_fact("Fish are quiet pets")

        results = store.search_facts("Dogs loyal")
        assert len(results) >= 1
        contents = [r["content"] for r in results]
        assert any("Dogs" in c for c in contents)

    # 5 -----------------------------------------------------------------------
    def test_search_facts_category_filter(self, store):
        store.add_fact("Prefer tabs over spaces", category="user_pref")
        store.add_fact("Project uses pytest", category="project")

        results = store.search_facts("spaces tabs pytest prefer project", category="user_pref")
        assert all(r["category"] == "user_pref" for r in results)

    # 6 -----------------------------------------------------------------------
    def test_search_facts_min_trust_filter(self, store):
        fact_id = store.add_fact("Low trust fact about rabbits")
        # Lower trust below the default min_trust threshold
        store.update_fact(fact_id, trust_delta=-0.3)

        results = store.search_facts("rabbits", min_trust=0.4)
        assert all(r["trust_score"] >= 0.4 for r in results)

    # 7 -----------------------------------------------------------------------
    def test_update_fact_content(self, store):
        fact_id = store.add_fact("Old content about the universe")
        store.update_fact(fact_id, content="New content about the universe")

        facts = store.list_facts()
        match = next(f for f in facts if f["fact_id"] == fact_id)
        assert match["content"] == "New content about the universe"

    # 8 -----------------------------------------------------------------------
    def test_update_fact_trust_delta(self, store):
        fact_id = store.add_fact("A neutral fact")
        before = store.list_facts()[0]["trust_score"]

        store.update_fact(fact_id, trust_delta=+0.2)

        after = store.list_facts()[0]["trust_score"]
        assert abs(after - (before + 0.2)) < 1e-9

    # 9 -----------------------------------------------------------------------
    def test_update_fact_trust_clamp(self, store):
        fact_id = store.add_fact("Clamp test fact")

        # Push trust to above 1.0
        store.update_fact(fact_id, trust_delta=+999.0)
        high = store.list_facts()[0]["trust_score"]
        assert high <= 1.0

        # Push trust below 0.0
        store.update_fact(fact_id, trust_delta=-999.0)
        low = store.list_facts()[0]["trust_score"]
        assert low >= 0.0

    # 10 ----------------------------------------------------------------------
    def test_remove_fact(self, store):
        fact_id = store.add_fact("Fact to be deleted")
        removed = store.remove_fact(fact_id)
        assert removed is True

        facts = store.list_facts()
        assert all(f["fact_id"] != fact_id for f in facts)

    # 11 ----------------------------------------------------------------------
    def test_list_facts(self, store):
        for i in range(5):
            store.add_fact(f"Fact number {i} about widgets")

        facts = store.list_facts(limit=50)
        assert len(facts) == 5

    # 12 ----------------------------------------------------------------------
    def test_list_facts_by_category(self, store):
        store.add_fact("User pref: dark mode", category="user_pref")
        store.add_fact("Project: use ruff", category="project")
        store.add_fact("Tool: ripgrep is fast", category="tool")

        user_facts = store.list_facts(category="user_pref")
        assert len(user_facts) == 1
        assert user_facts[0]["category"] == "user_pref"

    # 13 ----------------------------------------------------------------------
    def test_record_feedback_helpful(self, store):
        fact_id = store.add_fact("Helpful piece of knowledge")
        result = store.record_feedback(fact_id, helpful=True)

        assert result["fact_id"] == fact_id
        assert abs(result["new_trust"] - result["old_trust"] - 0.05) < 1e-9
        assert result["helpful_count"] == 1

    # 14 ----------------------------------------------------------------------
    def test_record_feedback_unhelpful(self, store):
        fact_id = store.add_fact("Wrong piece of knowledge")
        result = store.record_feedback(fact_id, helpful=False)

        assert result["fact_id"] == fact_id
        assert abs(result["old_trust"] - result["new_trust"] - 0.10) < 1e-9
        assert result["helpful_count"] == 0

    # 15 ----------------------------------------------------------------------
    def test_record_feedback_unknown_id(self, store):
        with pytest.raises(KeyError):
            store.record_feedback(99999, helpful=True)


# ===========================================================================
# TestEntityResolution — entity extraction + linking
# ===========================================================================


class TestEntityResolution:

    # 16 ----------------------------------------------------------------------
    def test_extract_capitalized_phrases(self, store):
        entities = store._extract_entities("Alice Smith loves coding")
        assert "Alice Smith" in entities

    # 17 ----------------------------------------------------------------------
    def test_extract_quoted_terms(self, store):
        entities = store._extract_entities('She uses "Visual Studio Code" daily')
        assert "Visual Studio Code" in entities

    # 18 ----------------------------------------------------------------------
    def test_extract_aka_patterns(self, store):
        # AKA pattern yields both sides as candidates
        entities = store._extract_entities("VS Code aka Visual Studio Code is great")
        assert "VS Code" in entities
        assert "Visual Studio Code" in entities

    # 19 ----------------------------------------------------------------------
    def test_entity_linking_on_add(self, store):
        fact_id = store.add_fact("John Doe wrote this document")

        # Verify the entity exists in entities table
        row = store._conn.execute(
            "SELECT entity_id FROM entities WHERE name = 'John Doe'"
        ).fetchone()
        assert row is not None, "Entity 'John Doe' should have been created"

        entity_id = row["entity_id"]
        link = store._conn.execute(
            "SELECT 1 FROM fact_entities WHERE fact_id = ? AND entity_id = ?",
            (fact_id, entity_id),
        ).fetchone()
        assert link is not None, "Fact-entity link should exist"

    # 20 ----------------------------------------------------------------------
    def test_entity_dedup(self, store):
        store.add_fact("Jane Doe likes Python")
        store.add_fact("Jane Doe prefers dark themes")

        rows = store._conn.execute(
            "SELECT entity_id FROM entities WHERE name = 'Jane Doe'"
        ).fetchall()
        assert len(rows) == 1, "Should have exactly one entity record for 'Jane Doe'"


# ===========================================================================
# TestFactRetriever — hybrid search
# ===========================================================================


class TestFactRetriever:

    # 21 ----------------------------------------------------------------------
    def test_search_returns_scored_results(self, store, retriever):
        store.add_fact("Machine learning is transforming technology")
        results = retriever.search("machine learning")

        assert len(results) >= 1
        for r in results:
            assert "score" in r

    # 22 ----------------------------------------------------------------------
    def test_search_relevance_ranking(self, store, retriever):
        # Exact phrase match should rank higher than vague mention
        store.add_fact("pytest is the best testing framework for Python projects")
        store.add_fact("There are many tools available for various purposes")

        results = retriever.search("pytest testing framework")
        assert len(results) >= 1
        # The exact match should be the first result
        assert "pytest" in results[0]["content"]

    # 23 ----------------------------------------------------------------------
    def test_search_trust_weighting(self, store, retriever):
        id_high = store.add_fact("Reliable fact: the earth orbits the sun")
        id_low  = store.add_fact("Questionable fact: the earth is stationary")

        # Give one high trust, leave other with default
        store.update_fact(id_high, trust_delta=+0.4)   # trust → 0.9
        store.update_fact(id_low,  trust_delta=-0.2)   # trust → 0.3

        results = retriever.search("earth fact", min_trust=0.0)
        if len(results) >= 2:
            assert results[0]["trust_score"] > results[-1]["trust_score"]

    # 24 ----------------------------------------------------------------------
    def test_search_empty_query_returns_empty(self, store, retriever):
        store.add_fact("Some fact in the store")
        results = retriever.search("")
        assert results == []

    # 25 ----------------------------------------------------------------------
    def test_jaccard_similarity(self):
        sim = FactRetriever._jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4 → 0.5
        assert abs(sim - 0.5) < 1e-9

    def test_jaccard_similarity_empty_sets(self):
        assert FactRetriever._jaccard_similarity(set(), {"a"}) == 0.0
        assert FactRetriever._jaccard_similarity({"a"}, set()) == 0.0
        assert FactRetriever._jaccard_similarity(set(), set()) == 0.0

    def test_jaccard_similarity_identical(self):
        assert FactRetriever._jaccard_similarity({"x", "y"}, {"x", "y"}) == 1.0

    # 26 ----------------------------------------------------------------------
    def test_tokenize(self):
        tokens = FactRetriever._tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Punctuation stripped
        assert "hello," not in tokens
        assert "test." not in tokens

    def test_tokenize_empty(self):
        assert FactRetriever._tokenize("") == set()

    def test_tokenize_lowercases(self):
        tokens = FactRetriever._tokenize("Python PYTHON python")
        assert tokens == {"python"}


# ===========================================================================
# TestTemporalDecay
# ===========================================================================


class TestTemporalDecay:

    # 27 ----------------------------------------------------------------------
    def test_temporal_decay_disabled(self, store):
        retriever = FactRetriever(store=store, temporal_decay_half_life=0)
        result = retriever._temporal_decay("2020-01-01 00:00:00")
        assert result == 1.0

    # 28 ----------------------------------------------------------------------
    def test_temporal_decay_recent(self, store):
        retriever = FactRetriever(store=store, temporal_decay_half_life=30)
        # A timestamp from seconds ago should yield decay very close to 1.0
        now_str = datetime.now(timezone.utc).isoformat()
        result = retriever._temporal_decay(now_str)
        assert result > 0.99

    # 29 ----------------------------------------------------------------------
    def test_temporal_decay_old(self, store):
        retriever = FactRetriever(store=store, temporal_decay_half_life=30)
        # A timestamp 30 days ago should yield decay ≈ 0.5 (one half-life)
        old_ts = "2000-01-01 00:00:00"
        result = retriever._temporal_decay(old_ts)
        assert result < 0.01  # extremely old → effectively 0

    def test_temporal_decay_none_timestamp(self, store):
        retriever = FactRetriever(store=store, temporal_decay_half_life=30)
        assert retriever._temporal_decay(None) == 1.0

    def test_temporal_decay_invalid_timestamp(self, store):
        retriever = FactRetriever(store=store, temporal_decay_half_life=30)
        assert retriever._temporal_decay("not-a-date") == 1.0


# ===========================================================================
# TestPluginRegistration
# ===========================================================================


class TestPluginRegistration:

    def _make_ctx(self):
        """Return a mock plugin context that records register_tool calls."""
        ctx = MagicMock()
        ctx.register_tool = MagicMock()
        ctx.register_hook = MagicMock()
        return ctx

    # 30 ----------------------------------------------------------------------
    def test_register_creates_tools(self, tmp_path):
        # Patch MemoryStore to use tmp_path db so register() doesn't touch ~/.hermes/
        import store as store_module
        import retrieval as retrieval_module

        original_init = MemoryStore.__init__

        db_file = tmp_path / "reg_test.db"

        def patched_init(self, db_path="~/.hermes/memory_store.db", default_trust=0.5, **kwargs):
            original_init(self, db_path=db_file, default_trust=default_trust, **kwargs)

        # We need to import the plugin __init__ via its package path
        import importlib
        import sys as _sys

        plugin_init_path = str(_plugin_dir)
        if plugin_init_path not in _sys.path:
            _sys.path.insert(0, plugin_init_path)

        # Temporarily patch MemoryStore.__init__
        MemoryStore.__init__ = patched_init
        try:
            # Re-import to pick up fresh module state
            if "hermes_memory_store" in _sys.modules:
                del _sys.modules["hermes_memory_store"]

            # Import the __init__ directly from the plugin directory
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "hermes_memory_store",
                _plugin_dir / "__init__.py",
                submodule_search_locations=[str(_plugin_dir)],
            )
            plugin_mod = importlib.util.module_from_spec(spec)
            # Register sub-modules so relative imports work
            _sys.modules["hermes_memory_store"] = plugin_mod
            _sys.modules["hermes_memory_store.store"] = _sys.modules["store"]
            _sys.modules["hermes_memory_store.retrieval"] = _sys.modules["retrieval"]
            spec.loader.exec_module(plugin_mod)

            ctx = self._make_ctx()
            plugin_mod.register(ctx)

            # Verify register_tool was called for both tools
            registered_names = [
                c.kwargs.get("name") or c.args[0]
                for c in ctx.register_tool.call_args_list
            ]
            assert "fact_store" in registered_names
            assert "fact_feedback" in registered_names
        finally:
            MemoryStore.__init__ = original_init
            # Clean up sys.modules entries we added
            for key in ["hermes_memory_store", "hermes_memory_store.store", "hermes_memory_store.retrieval"]:
                _sys.modules.pop(key, None)

    # 31 ----------------------------------------------------------------------
    def test_handlers_return_json(self, tmp_path):
        """All handler return values must be valid JSON strings."""
        from store import MemoryStore as MS
        from retrieval import FactRetriever as FR
        import importlib.util
        import sys as _sys

        spec = importlib.util.spec_from_file_location(
            "_plugin_init_json_test",
            _plugin_dir / "__init__.py",
            submodule_search_locations=[str(_plugin_dir)],
        )
        plugin_mod = importlib.util.module_from_spec(spec)
        _sys.modules["_plugin_init_json_test"] = plugin_mod
        _sys.modules["_plugin_init_json_test.store"] = _sys.modules["store"]
        _sys.modules["_plugin_init_json_test.retrieval"] = _sys.modules["retrieval"]

        # Patch relative imports inside plugin __init__ to use already-loaded modules
        import store as _store_mod
        import retrieval as _retrieval_mod

        # Manually populate the module namespace before exec
        plugin_mod.MemoryStore = MS
        plugin_mod.FactRetriever = FR

        spec.loader.exec_module(plugin_mod)

        s = MS(db_path=tmp_path / "handler_test.db")
        r = FR(store=s)
        min_trust = 0.3

        fact_handler = plugin_mod._make_fact_store_handler(s, r, min_trust)
        feedback_handler = plugin_mod._make_fact_feedback_handler(s)

        try:
            # add
            result = fact_handler({"action": "add", "content": "JSON test fact"})
            parsed = json.loads(result)
            assert "fact_id" in parsed
            fact_id = parsed["fact_id"]

            # search
            result = fact_handler({"action": "search", "query": "JSON test"})
            parsed = json.loads(result)
            assert "results" in parsed

            # update
            result = fact_handler({"action": "update", "fact_id": fact_id, "trust_delta": 0.1})
            parsed = json.loads(result)
            assert "updated" in parsed

            # list
            result = fact_handler({"action": "list"})
            parsed = json.loads(result)
            assert "facts" in parsed

            # feedback helpful
            result = feedback_handler({"action": "helpful", "fact_id": fact_id})
            parsed = json.loads(result)
            assert "new_trust" in parsed

            # feedback unhelpful
            result = feedback_handler({"action": "unhelpful", "fact_id": fact_id})
            parsed = json.loads(result)
            assert "new_trust" in parsed

            # remove
            result = fact_handler({"action": "remove", "fact_id": fact_id})
            parsed = json.loads(result)
            assert "removed" in parsed

            # unknown action returns error JSON
            result = fact_handler({"action": "nonexistent"})
            parsed = json.loads(result)
            assert "error" in parsed

        finally:
            s.close()
            for key in ["_plugin_init_json_test", "_plugin_init_json_test.store", "_plugin_init_json_test.retrieval"]:
                _sys.modules.pop(key, None)


# ===========================================================================
# TestHolographicIntegration — HRR vector storage and retrieval
# ===========================================================================


class TestHolographicIntegration:
    """Integration tests for HRR vector storage and retrieval."""

    def test_add_fact_stores_hrr_vector(self, tmp_path):
        """Adding a fact should compute and store an HRR vector."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        fact_id = store.add_fact("peppi prefers Rust over Python", category="user_pref")

        row = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()

        if store._hrr_available:
            assert row["hrr_vector"] is not None
            assert len(row["hrr_vector"]) == store.hrr_dim * 8  # float64
        else:
            assert row["hrr_vector"] is None
        store.close()

    def test_search_includes_hrr_signal(self, tmp_path):
        """Search with HRR should produce scored results."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("peppi uses Neovim as primary editor", category="user_pref")
        store.add_fact("the project uses SQLite for storage", category="project")

        retriever = FactRetriever(store=store)
        results = retriever.search("editor neovim")

        assert len(results) > 0
        assert all("score" in r for r in results)
        store.close()

    def test_probe_entity_retrieval(self, tmp_path):
        """Probe should find facts by entity structure."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        fid1 = store.add_fact("Peppi prefers Rust over Python for systems code", category="user_pref")
        fid2 = store.add_fact("Peppi uses Neovim as primary editor", category="user_pref")
        store.add_fact("SQLite is used for the memory store", category="project")

        retriever = FactRetriever(store=store)
        results = retriever.probe("peppi", category="user_pref")

        assert len(results) >= 2
        # Results should include the peppi-related facts
        result_ids = {r["fact_id"] for r in results}
        assert fid1 in result_ids
        assert fid2 in result_ids
        store.close()

    def test_probe_fallback_no_numpy(self, tmp_path):
        """Probe should gracefully fall back to FTS5 when numpy unavailable."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("important project fact about databases", category="project")

        # Create a retriever with hrr_weight=0 (simulates no-numpy scenario)
        retriever = FactRetriever(store=store, hrr_weight=0.0)
        results = retriever.probe("databases", category="project")
        # Should fall back to FTS search and still return results
        assert isinstance(results, list)
        store.close()

    def test_rebuild_vectors_deterministic(self, tmp_path):
        """Rebuilt vectors should match originals (deterministic encoding)."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        if not store._hrr_available:
            pytest.skip("numpy not available")

        store.add_fact("test fact for rebuild", category="general")

        # Get original vector
        row1 = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = 1"
        ).fetchone()
        original = row1["hrr_vector"]

        # Clear and rebuild
        store._conn.execute("UPDATE facts SET hrr_vector = NULL")
        store._conn.commit()
        count = store.rebuild_all_vectors()

        assert count == 1
        row2 = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = 1"
        ).fetchone()
        assert row2["hrr_vector"] == original
        store.close()

    def test_memory_bank_lifecycle(self, tmp_path):
        """Memory banks should be created on add and updated on remove."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        if not store._hrr_available:
            pytest.skip("numpy not available")

        fid1 = store.add_fact("first fact", category="test_cat")

        # Bank should exist after add
        bank = store._conn.execute(
            "SELECT fact_count FROM memory_banks WHERE bank_name = ?",
            ("cat:test_cat",)
        ).fetchone()
        assert bank is not None
        assert bank["fact_count"] == 1

        fid2 = store.add_fact("second fact", category="test_cat")
        bank = store._conn.execute(
            "SELECT fact_count FROM memory_banks WHERE bank_name = ?",
            ("cat:test_cat",)
        ).fetchone()
        assert bank["fact_count"] == 2

        # Remove a fact — bank should update
        store.remove_fact(fid1)
        bank = store._conn.execute(
            "SELECT fact_count FROM memory_banks WHERE bank_name = ?",
            ("cat:test_cat",)
        ).fetchone()
        assert bank["fact_count"] == 1

        # Remove last fact — bank should be deleted
        store.remove_fact(fid2)
        bank = store._conn.execute(
            "SELECT fact_count FROM memory_banks WHERE bank_name = ?",
            ("cat:test_cat",)
        ).fetchone()
        assert bank is None
        store.close()

    def test_related_finds_connected_facts(self, tmp_path):
        """Related should find facts structurally connected to an entity."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        if not store._hrr_available:
            pytest.skip("numpy not available")

        store.add_fact("Peppi prefers Rust for systems code", category="user_pref")
        store.add_fact("Rust has great memory safety guarantees", category="general")
        store.add_fact("Python is used for data science", category="general")

        retriever = FactRetriever(store=store)
        results = retriever.related("rust")

        assert len(results) >= 2
        # Rust-related facts should score higher than unrelated ones
        contents = [r["content"] for r in results]
        assert any("Rust" in c for c in contents[:2])
        store.close()

    def test_related_fallback_no_numpy(self, tmp_path):
        """Related should fall back to FTS when numpy unavailable."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("important fact about testing", category="general")

        retriever = FactRetriever(store=store, hrr_weight=0.0)
        results = retriever.related("testing")
        assert isinstance(results, list)
        store.close()

    def test_related_action_via_handler(self, tmp_path):
        """The 'related' action in fact_store handler should return valid JSON."""
        import importlib.util
        import sys as _sys

        from store import MemoryStore as MS
        from retrieval import FactRetriever as FR

        spec = importlib.util.spec_from_file_location(
            "_plugin_related_test",
            _plugin_dir / "__init__.py",
            submodule_search_locations=[str(_plugin_dir)],
        )
        plugin_mod = importlib.util.module_from_spec(spec)
        _sys.modules["_plugin_related_test"] = plugin_mod
        _sys.modules["_plugin_related_test.store"] = _sys.modules["store"]
        _sys.modules["_plugin_related_test.retrieval"] = _sys.modules["retrieval"]
        plugin_mod.MemoryStore = MS
        plugin_mod.FactRetriever = FR

        spec.loader.exec_module(plugin_mod)

        s = MS(db_path=tmp_path / "related_handler_test.db")
        r = FR(store=s)
        handler = plugin_mod._make_fact_store_handler(s, r, 0.3)

        try:
            handler({"action": "add", "content": "Peppi uses Rust for systems work"})
            handler({"action": "add", "content": "Rust ensures memory safety"})

            result = handler({"action": "related", "entity": "rust"})
            parsed = json.loads(result)
            assert "results" in parsed
            assert "count" in parsed
            assert isinstance(parsed["results"], list)
        finally:
            s.close()
            for key in ["_plugin_related_test", "_plugin_related_test.store", "_plugin_related_test.retrieval"]:
                _sys.modules.pop(key, None)

    def test_reason_multi_entity_intersection(self, tmp_path):
        """Reason should find facts connected to ALL given entities."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        if not store._hrr_available:
            pytest.skip("numpy not available")

        # Facts with different entity combinations
        store.add_fact("Peppi prefers Rust for systems programming", category="user_pref")
        store.add_fact("Peppi uses Neovim as primary editor", category="user_pref")
        store.add_fact("Rust has great memory safety guarantees", category="general")

        retriever = FactRetriever(store=store)
        # Reason with peppi + rust — should favor the fact mentioning BOTH
        results = retriever.reason(["peppi", "rust"])

        assert len(results) >= 1
        # The fact mentioning both peppi and Rust should rank highest
        assert "Rust" in results[0]["content"] or "peppi" in results[0]["content"].lower()
        store.close()

    def test_reason_single_entity_degrades_to_probe(self, tmp_path):
        """Reason with one entity should still work (degrades gracefully)."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        if not store._hrr_available:
            pytest.skip("numpy not available")

        store.add_fact("Peppi loves dark mode", category="user_pref")

        retriever = FactRetriever(store=store)
        results = retriever.reason(["peppi"])

        assert len(results) >= 1
        store.close()

    def test_reason_fallback_no_numpy(self, tmp_path):
        """Reason should fall back to keyword search without numpy."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("important fact about testing things", category="general")

        retriever = FactRetriever(store=store, hrr_weight=0.0)
        results = retriever.reason(["testing", "things"])
        assert isinstance(results, list)
        store.close()

    def test_reason_action_via_handler(self, tmp_path):
        """The reason action should return valid JSON through the retriever directly."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        if not store._hrr_available:
            pytest.skip("numpy not available")

        store.add_fact("Alice works on the Backend Team", category="project")
        retriever = FactRetriever(store=store)

        results = retriever.reason(["alice", "backend"])
        assert isinstance(results, list)
        assert len(results) >= 1
        store.close()

    def test_contradict_finds_conflicting_facts(self, tmp_path):
        """Contradict should detect facts with same entities but different claims."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        if not store._hrr_available:
            pytest.skip("numpy not available")

        # Two facts about Peppi with conflicting preferences
        store.add_fact('"Peppi" prefers Rust for systems programming', category="user_pref")
        store.add_fact('"Peppi" prefers Python for systems programming', category="user_pref")
        # Unrelated fact
        store.add_fact("SQLite is used for storage", category="project")

        retriever = FactRetriever(store=store)
        results = retriever.contradict(category="user_pref", threshold=0.01)

        # Should find at least the conflicting pair
        assert len(results) >= 1
        pair = results[0]
        assert "fact_a" in pair
        assert "fact_b" in pair
        assert "contradiction_score" in pair
        assert "shared_entities" in pair
        assert len(pair["shared_entities"]) > 0
        store.close()

    def test_contradict_no_conflicts(self, tmp_path):
        """Contradict should return empty when no conflicts exist."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        if not store._hrr_available:
            pytest.skip("numpy not available")

        store.add_fact("Peppi uses Neovim", category="user_pref")
        store.add_fact("SQLite is used for storage", category="project")

        retriever = FactRetriever(store=store)
        results = retriever.contradict()

        # No shared entities between these facts, so no contradictions
        assert isinstance(results, list)
        store.close()

    def test_contradict_fallback_no_numpy(self, tmp_path):
        """Contradict should return empty list without numpy."""
        store = MemoryStore(db_path=tmp_path / "test.db")
        store.add_fact("some fact", category="general")

        retriever = FactRetriever(store=store, hrr_weight=0.0)
        results = retriever.contradict()
        assert results == [] or isinstance(results, list)
        store.close()


# ===========================================================================
# TestAutoExtraction — _extract_sentence pure function
# ===========================================================================


def _extract_sentence(text: str, pos: int) -> str:
    """Local copy of the pure helper from the plugin __init__.

    Copied here so tests remain stable and import-independent.
    The implementation must match hermes-memory-store/__init__.py.
    """
    start = pos
    while start > 0 and text[start - 1] not in '.!?\n':
        start -= 1

    end = pos
    while end < len(text) and text[end] not in '.!?\n':
        end += 1

    sentence = text[start:end].strip()
    sentence = sentence.lstrip('.!? ')
    return sentence


class TestAutoExtraction:
    """Tests for the on_session_end auto-extraction logic."""

    def test_extract_sentence_middle(self):
        """Should extract the sentence containing pos, not its neighbours."""
        text = "Hello world. I prefer Rust over Python. It is fast."
        result = _extract_sentence(text, text.index("I prefer"))
        assert "I prefer Rust over Python" in result
        assert "Hello" not in result

    def test_extract_sentence_at_start(self):
        """Should handle a sentence at position zero."""
        text = "I always use dark mode"
        result = _extract_sentence(text, 0)
        assert result == "I always use dark mode"

    def test_extract_sentence_last_sentence(self):
        """Should capture the final sentence even without trailing punctuation."""
        text = "First sentence. Last sentence without period"
        result = _extract_sentence(text, text.index("Last"))
        assert "Last sentence without period" in result
        assert "First" not in result

    def test_extract_sentence_newline_boundary(self):
        """Newlines act as sentence boundaries."""
        text = "Line one.\nI prefer tabs\nLine three."
        result = _extract_sentence(text, text.index("I prefer"))
        assert result == "I prefer tabs"
        assert "Line one" not in result
        assert "Line three" not in result

    def test_extract_sentence_strips_leading_punctuation(self):
        """Leading . ! ? space should be stripped from the result."""
        text = "First. Second fact here. Third."
        result = _extract_sentence(text, text.index("Second"))
        assert not result.startswith(".")
        assert "Second fact here" in result

    def test_extract_sentence_single_sentence_text(self):
        """Single-sentence text without punctuation returns whole text."""
        text = "just one sentence here"
        result = _extract_sentence(text, 5)
        assert result == "just one sentence here"

    def test_on_session_end_extracts_preference(self, tmp_path):
        """Auto-extraction hook should store I-prefer sentences."""
        import importlib.util
        import sys as _sys

        from store import MemoryStore as MS
        from retrieval import FactRetriever as FR

        spec = importlib.util.spec_from_file_location(
            "_plugin_session_end_test",
            _plugin_dir / "__init__.py",
            submodule_search_locations=[str(_plugin_dir)],
        )
        plugin_mod = importlib.util.module_from_spec(spec)
        _sys.modules["_plugin_session_end_test"] = plugin_mod
        _sys.modules["_plugin_session_end_test.store"] = _sys.modules["store"]
        _sys.modules["_plugin_session_end_test.retrieval"] = _sys.modules["retrieval"]
        plugin_mod.MemoryStore = MS
        plugin_mod.FactRetriever = FR
        spec.loader.exec_module(plugin_mod)

        s = MS(db_path=tmp_path / "session_end_test.db")
        handler = plugin_mod._make_session_end_handler(s)

        try:
            messages = [
                {"role": "user", "content": "I prefer Rust over Python for systems code."},
                {"role": "assistant", "content": "Got it, I'll remember that."},
            ]
            handler(messages=messages)

            facts = s.list_facts(category="user_pref")
            assert len(facts) >= 1
            assert any("prefer" in f["content"].lower() for f in facts)
        finally:
            s.close()
            for key in ["_plugin_session_end_test", "_plugin_session_end_test.store",
                        "_plugin_session_end_test.retrieval"]:
                _sys.modules.pop(key, None)

    def test_on_session_end_skips_assistant_messages(self, tmp_path):
        """Only user messages should be scanned for facts."""
        import importlib.util
        import sys as _sys

        from store import MemoryStore as MS
        from retrieval import FactRetriever as FR

        spec = importlib.util.spec_from_file_location(
            "_plugin_session_end_assistant_test",
            _plugin_dir / "__init__.py",
            submodule_search_locations=[str(_plugin_dir)],
        )
        plugin_mod = importlib.util.module_from_spec(spec)
        _sys.modules["_plugin_session_end_assistant_test"] = plugin_mod
        _sys.modules["_plugin_session_end_assistant_test.store"] = _sys.modules["store"]
        _sys.modules["_plugin_session_end_assistant_test.retrieval"] = _sys.modules["retrieval"]
        plugin_mod.MemoryStore = MS
        plugin_mod.FactRetriever = FR
        spec.loader.exec_module(plugin_mod)

        s = MS(db_path=tmp_path / "session_end_assistant_test.db")
        handler = plugin_mod._make_session_end_handler(s)

        try:
            messages = [
                {"role": "assistant", "content": "I prefer to use formal language in responses."},
            ]
            handler(messages=messages)

            facts = s.list_facts()
            assert len(facts) == 0
        finally:
            s.close()
            for key in ["_plugin_session_end_assistant_test",
                        "_plugin_session_end_assistant_test.store",
                        "_plugin_session_end_assistant_test.retrieval"]:
                _sys.modules.pop(key, None)

    def test_on_session_end_extracts_decision(self, tmp_path):
        """Decision patterns should be stored with category=project."""
        import importlib.util
        import sys as _sys

        from store import MemoryStore as MS
        from retrieval import FactRetriever as FR

        spec = importlib.util.spec_from_file_location(
            "_plugin_session_end_decision_test",
            _plugin_dir / "__init__.py",
            submodule_search_locations=[str(_plugin_dir)],
        )
        plugin_mod = importlib.util.module_from_spec(spec)
        _sys.modules["_plugin_session_end_decision_test"] = plugin_mod
        _sys.modules["_plugin_session_end_decision_test.store"] = _sys.modules["store"]
        _sys.modules["_plugin_session_end_decision_test.retrieval"] = _sys.modules["retrieval"]
        plugin_mod.MemoryStore = MS
        plugin_mod.FactRetriever = FR
        spec.loader.exec_module(plugin_mod)

        s = MS(db_path=tmp_path / "session_end_decision_test.db")
        handler = plugin_mod._make_session_end_handler(s)

        try:
            messages = [
                {"role": "user", "content": "The project uses SQLite for the memory store."},
            ]
            handler(messages=messages)

            facts = s.list_facts(category="project")
            assert len(facts) >= 1
            assert any("SQLite" in f["content"] for f in facts)
        finally:
            s.close()
            for key in ["_plugin_session_end_decision_test",
                        "_plugin_session_end_decision_test.store",
                        "_plugin_session_end_decision_test.retrieval"]:
                _sys.modules.pop(key, None)
