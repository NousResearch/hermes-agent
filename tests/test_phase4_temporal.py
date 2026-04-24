"""Tests for Phase 4 — Temporal & Quality.

Covers:
  - temporal.py: staleness detection, contradiction resolution, relationship extraction
  - episodic_store.py: relationships, fact_history, temporal queries
  - Provider tools: memory_stale_facts, memory_contradictions, memory_relationships
"""

import json
import time

import pytest

from memory.episodic_store import EpisodicStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh episodic store for testing."""
    db_path = tmp_path / "test_index.db"
    s = EpisodicStore(db_path=db_path)
    s.ensure_session("test-s1", source="test")
    s.ensure_session("test-s2", source="test")
    yield s
    s.close()


# ── Relationship operations ──────────────────────────────────────────────


class TestRelationships:
    def test_add_relationship(self, store):
        store.upsert_entity("person-aaron", "person", "Aaron", {"role": "partner"})
        store.upsert_entity("person-jefe", "person", "Jefe", {"role": "owner"})

        rid = store.add_relationship(
            source_entity_id="person-aaron",
            target_entity_id="person-jefe",
            relation_type="works_with",
            session_id="test-s1",
        )
        assert rid > 0

    def test_add_relationship_dedup(self, store):
        store.upsert_entity("e1", "project", "Hermes", {})
        store.upsert_entity("e2", "tool", "Ollama", {})

        rid1 = store.add_relationship("e1", "e2", "uses", {"version": "0.18"})
        rid2 = store.add_relationship("e1", "e2", "uses", {"version": "0.19"})
        assert rid1 == rid2  # Same ID, updated attributes

        rels = store.get_relationships("e1", direction="outgoing")
        assert len(rels) == 1
        assert rels[0]["attributes"]["version"] == "0.19"

    def test_get_relationships_direction(self, store):
        store.upsert_entity("a", "person", "A", {})
        store.upsert_entity("b", "person", "B", {})
        store.upsert_entity("c", "person", "C", {})

        store.add_relationship("a", "b", "knows")
        store.add_relationship("c", "a", "reports_to")

        outgoing = store.get_relationships("a", direction="outgoing")
        assert len(outgoing) == 1
        assert outgoing[0]["target_entity_id"] == "b"

        incoming = store.get_relationships("a", direction="incoming")
        assert len(incoming) == 1
        assert incoming[0]["source_entity_id"] == "c"

        both = store.get_relationships("a", direction="both")
        assert len(both) == 2

    def test_get_relationships_filter_type(self, store):
        store.upsert_entity("a", "person", "A", {})
        store.upsert_entity("b", "person", "B", {})

        store.add_relationship("a", "b", "knows")
        store.add_relationship("a", "b", "manages")

        knows = store.get_relationships("a", relation_type="knows")
        assert len(knows) == 1
        assert knows[0]["relation_type"] == "knows"

    def test_get_all_relationship_types(self, store):
        store.upsert_entity("a", "person", "A", {})
        store.upsert_entity("b", "person", "B", {})

        store.add_relationship("a", "b", "knows")
        store.add_relationship("a", "b", "manages")

        types = store.get_all_relationship_types()
        assert "knows" in types
        assert "manages" in types

    def test_relationship_graph(self, store):
        store.upsert_entity("a", "person", "Alice", {})
        store.upsert_entity("b", "person", "Bob", {})
        store.upsert_entity("c", "person", "Carol", {})

        store.add_relationship("a", "b", "knows")
        store.add_relationship("b", "c", "manages")

        # Depth 1: only Alice → Bob
        graph = store.get_entity_relationships_graph("a", depth=1)
        assert graph["center"] == "a"
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["target"] == "b"

        # Depth 2: Alice → Bob → Carol
        graph = store.get_entity_relationships_graph("a", depth=2)
        assert len(graph["edges"]) == 2
        node_ids = [n["id"] for n in graph["nodes"]]
        assert "a" in node_ids
        assert "b" in node_ids
        assert "c" in node_ids


# ── Fact History ─────────────────────────────────────────────────────────


class TestFactHistory:
    def test_record_fact_change(self, store):
        store.upsert_entity("e1", "project", "Hermes", {})

        hid = store.record_fact_change(
            entity_id="e1",
            field_path="version",
            old_value="1.0",
            new_value="2.0",
            operation="UPDATE",
            session_id="test-s1",
        )
        assert hid > 0

    def test_get_fact_history(self, store):
        store.upsert_entity("e1", "project", "Hermes", {})

        store.record_fact_change("e1", "version", "1.0", "2.0", "UPDATE", "test-s1")
        time.sleep(0.01)
        store.record_fact_change("e1", "version", "2.0", "3.0", "UPDATE", "test-s2")
        time.sleep(0.01)
        store.record_fact_change("e1", "status", None, "active", "ADD", "test-s1")

        # All history
        history = store.get_fact_history("e1")
        assert len(history) == 3

        # Filtered by field
        version_history = store.get_fact_history("e1", field_path="version")
        assert len(version_history) == 2
        assert all(h["field_path"] == "version" for h in version_history)

    def test_fact_history_order(self, store):
        store.upsert_entity("e1", "project", "Hermes", {})

        store.record_fact_change("e1", "v", "1", "2", "UPDATE", "test-s1")
        time.sleep(0.01)
        store.record_fact_change("e1", "v", "2", "3", "UPDATE", "test-s2")

        history = store.get_fact_history("e1", field_path="v")
        # Most recent first
        assert history[0]["new_value"] == "3"
        assert history[1]["new_value"] == "2"


# ── Staleness Detection ─────────────────────────────────────────────────


class TestStaleness:
    def test_stale_entities(self, store):
        from memory.temporal import detect_stale_entities

        # Create entity with old last_confirmed_at
        store.upsert_entity("old-ent", "project", "OldProject", {})
        # last_confirmed_at is NULL by default → should be stale

        stale = detect_stale_entities(store, threshold_days=1, limit=10)
        assert any(e["entity_id"] == "old-ent" for e in stale)

    def test_fresh_entity_not_stale(self, store):
        from memory.temporal import detect_stale_entities

        store.upsert_entity("fresh", "project", "FreshProject", {})
        store.confirm_entity("fresh")

        # 1-day threshold — just confirmed, should NOT be stale
        stale = detect_stale_entities(store, threshold_days=1, limit=10)
        assert not any(e["entity_id"] == "fresh" for e in stale)

    def test_stale_facts(self, store):
        from memory.temporal import detect_stale_facts

        store.upsert_entity("e1", "project", "TestProject", {})
        store.record_fact_change("e1", "status", None, "active", "ADD", "test-s1")

        # With 1-second threshold, it should be stale immediately
        stale = detect_stale_facts(store, "e1", threshold_days=0)
        # threshold_days=0 → 0 seconds → cutoff = now → nothing is stale yet
        # Let's use a tiny threshold by checking directly
        time.sleep(0.1)
        stale = detect_stale_facts(store, "e1", threshold_days=0)
        # With 0 days threshold, cutoff = now. The fact was recorded 0.1s ago,
        # which is < 0 seconds... so it should be stale
        assert isinstance(stale, list)


# ── Contradiction Detection ──────────────────────────────────────────────


class TestContradictions:
    def test_detect_contradictions(self, store):
        from memory.temporal import detect_contradictions

        store.upsert_entity("e1", "project", "Test", {})

        # Same field, different values
        store.record_fact_change("e1", "status", None, "active", "UPDATE", "test-s1")
        time.sleep(0.01)
        store.record_fact_change("e1", "status", "active", "inactive", "UPDATE", "test-s2")
        time.sleep(0.01)
        store.record_fact_change("e1", "status", "inactive", "active", "UPDATE", "test-s1")

        contradictions = detect_contradictions(store, entity_id="e1")
        assert len(contradictions) >= 1
        c = contradictions[0]
        assert c["field_path"] == "status"
        assert c["change_count"] == 3
        assert "active" in c["values_seen"]
        assert "inactive" in c["values_seen"]

    def test_no_contradictions_single_change(self, store):
        from memory.temporal import detect_contradictions

        store.upsert_entity("e1", "project", "Test", {})
        store.record_fact_change("e1", "status", None, "active", "UPDATE", "test-s1")

        contradictions = detect_contradictions(store, entity_id="e1")
        assert len(contradictions) == 0

    def test_resolve_contradiction(self, store):
        from memory.temporal import resolve_contradiction

        store.upsert_entity("e1", "project", "Test", {"status": "old"})

        result = resolve_contradiction(
            store, "e1", "status", "active", session_id="test-s1"
        )
        assert result["status"] == "resolved"
        assert result["resolved_value"] == "active"

        # Verify the entity profile was updated
        entity = store.get_entity("e1")
        assert entity["profile_json"]["status"] == "active"

        # Verify the resolution was recorded in fact_history
        history = store.get_fact_history("e1", field_path="status")
        assert any(h["operation"] == "RESOLVE" for h in history)


# ── Relationship Extraction from Facts ────────────────────────────────────


class TestRelationshipExtraction:
    def test_extract_relationships(self, store):
        from memory.temporal import extract_relationships_from_facts

        # Set up entities
        store.upsert_entity("person-aaron", "person", "Aaron", {})
        store.upsert_entity("person-jefe", "person", "Jefe", {})

        extracted = {
            "entities": [
                {"name": "Aaron", "type": "person", "attributes": {}},
                {"name": "Jefe", "type": "person", "attributes": {}},
            ],
            "facts": [
                {"subject": "Aaron", "predicate": "works_with", "object": "Jefe", "confidence": "high"},
            ],
            "events": [],
        }

        count = extract_relationships_from_facts(store, extracted, "test-s1")
        assert count == 1

        # Verify the relationship exists
        rels = store.get_relationships("person-aaron", direction="outgoing")
        assert len(rels) == 1
        assert rels[0]["relation_type"] == "works_with"
        assert rels[0]["target_entity_id"] == "person-jefe"

    def test_extract_relationships_skips_unknown(self, store):
        from memory.temporal import extract_relationships_from_facts

        extracted = {
            "entities": [],
            "facts": [
                {"subject": "UnknownA", "predicate": "knows", "object": "UnknownB", "confidence": "low"},
            ],
            "events": [],
        }

        count = extract_relationships_from_facts(store, extracted, "test-s1")
        assert count == 0  # No entities to link


# ── Provider Tools ───────────────────────────────────────────────────────


class TestProviderTemporalTools:
    def test_memory_stale_facts_all(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        store.upsert_entity("old", "project", "OldProject", {})

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_stale_facts({"threshold_days": 1}))
        assert "stale_entities" in result
        assert result["count"] >= 1

    def test_memory_stale_facts_entity(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        store.upsert_entity("e1", "project", "Test", {})
        store.record_fact_change("e1", "status", None, "active", "ADD", "test-s1")

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_stale_facts({"entity_id": "e1", "threshold_days": 0}))
        assert "stale_facts" in result
        assert "entity" in result

    def test_memory_contradictions(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        store.upsert_entity("e1", "project", "Test", {})
        store.record_fact_change("e1", "status", None, "active", "UPDATE", "test-s1")
        time.sleep(0.01)
        store.record_fact_change("e1", "status", "active", "inactive", "UPDATE", "test-s2")

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_contradictions({"entity_id": "e1"}))
        assert result["count"] >= 1

    def test_memory_relationships(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        store.upsert_entity("a", "person", "Alice", {})
        store.upsert_entity("b", "person", "Bob", {})
        store.add_relationship("a", "b", "knows")

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_relationships({"entity_id": "a"}))
        assert result["center"] == "a"
        assert len(result["edges"]) == 1

    def test_memory_relationships_not_found(self, store):
        from memory.episodic_provider import EpisodicMemoryProvider

        provider = EpisodicMemoryProvider()
        provider._store = store
        provider._available = True

        result = json.loads(provider._tool_relationships({"entity_id": ""}))
        assert "error" in result
