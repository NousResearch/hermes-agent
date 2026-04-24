"""Tests for the Episodic Memory Store."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Ensure hermes-agent root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.episodic_store import EpisodicStore, _merge_profiles


@pytest.fixture
def store(tmp_path):
    """Create a fresh EpisodicStore with a temp database."""
    db_path = tmp_path / "test_index.db"
    s = EpisodicStore(db_path=db_path)
    yield s
    s.close()


# ── Session operations ────────────────────────────────────────────────────


def test_ensure_session(store):
    store.ensure_session("sess-1", source="test")
    stats = store.get_stats()
    assert stats["sessions"] == 1

    # Idempotent
    store.ensure_session("sess-1", source="test")
    stats = store.get_stats()
    assert stats["sessions"] == 1


# ── Turn operations ───────────────────────────────────────────────────────


def test_append_turn(store):
    store.ensure_session("sess-1")
    tid = store.append_turn("sess-1", "user", "Hello, world!")
    assert tid is not None
    assert tid >= 1

    stats = store.get_stats()
    assert stats["turns"] == 1


def test_append_turn_with_tool_calls(store):
    store.ensure_session("sess-1")
    tool_calls = [{"name": "web_search", "args": {"query": "test"}}]
    tid = store.append_turn(
        "sess-1", "assistant", "Searching...", tool_calls=tool_calls
    )
    assert tid >= 1

    turns = store.get_turns_for_session("sess-1")
    assert len(turns) == 1
    assert json.loads(turns[0]["tool_calls"]) == tool_calls


def test_append_turn_tool_result(store):
    store.ensure_session("sess-1")
    store.append_turn("sess-1", "tool", '{"results": []}', tool_name="web_search")
    turns = store.get_turns_for_session("sess-1")
    assert turns[0]["tool_name"] == "web_search"


def test_turn_ordering(store):
    store.ensure_session("sess-1")
    store.append_turn("sess-1", "user", "First", timestamp=1000.0)
    store.append_turn("sess-1", "assistant", "Second", timestamp=1001.0)
    store.append_turn("sess-1", "user", "Third", timestamp=1002.0)

    turns = store.get_turns_for_session("sess-1")
    assert len(turns) == 3
    assert turns[0]["content"] == "First"
    assert turns[2]["content"] == "Third"


# ── Episode operations ────────────────────────────────────────────────────


def test_create_episode(store):
    store.ensure_session("sess-1")
    t1 = store.append_turn("sess-1", "user", "Let's talk about Python")
    t2 = store.append_turn("sess-1", "assistant", "Sure, what about it?")

    ep_id = store.create_episode(
        session_id="sess-1",
        topic="Python discussion",
        summary="User asked about Python basics",
        key_decisions="Use Python 3.12",
        participants="user, assistant",
        source_turns=[t1, t2],
    )
    assert ep_id >= 1

    ep = store.get_episode(ep_id)
    assert ep["topic"] == "Python discussion"
    assert ep["summary"] == "User asked about Python basics"
    assert "3.12" in ep["key_decisions"]


def test_get_episode_turns(store):
    store.ensure_session("sess-1")
    t1 = store.append_turn("sess-1", "user", "Question")
    t2 = store.append_turn("sess-1", "assistant", "Answer")

    ep_id = store.create_episode(
        session_id="sess-1",
        topic="Q&A",
        summary="Simple Q&A",
        source_turns=[t1, t2],
    )

    turns = store.get_episode_turns(ep_id)
    assert len(turns) == 2
    assert turns[0]["content"] == "Question"
    assert turns[1]["content"] == "Answer"


def test_search_episodes(store):
    store.ensure_session("sess-1")
    store.create_episode(
        session_id="sess-1",
        topic="Python decorators",
        summary="Learned about Python decorator patterns and closures",
        key_decisions="Use @wraps for all decorators",
    )
    store.create_episode(
        session_id="sess-1",
        topic="Docker setup",
        summary="Configured Docker containers for the project",
    )

    results = store.search_episodes("Python")
    assert len(results) >= 1
    assert "decorator" in results[0]["topic"].lower() or "decorator" in results[0]["summary"].lower()


def test_get_recent_episodes(store):
    store.ensure_session("sess-1")
    for i in range(5):
        store.create_episode(
            session_id="sess-1",
            topic=f"Topic {i}",
            summary=f"Summary for topic {i}",
        )

    recent = store.get_recent_episodes(limit=3)
    assert len(recent) == 3
    # Most recent first
    assert "Topic 4" in recent[0]["topic"]


# ── Entity operations ─────────────────────────────────────────────────────


def test_upsert_entity(store):
    store.upsert_entity(
        entity_id="proj-hermes",
        entity_type="project",
        name="Hermes Agent",
        profile_json={"description": "AI agent framework", "status": "active"},
    )

    entity = store.get_entity("proj-hermes")
    assert entity["name"] == "Hermes Agent"
    assert entity["type"] == "project"
    assert entity["profile_json"]["status"] == "active"


def test_upsert_entity_merge(store):
    store.upsert_entity(
        entity_id="proj-hermes",
        entity_type="project",
        name="Hermes Agent",
        profile_json={"description": "AI agent framework", "status": "active"},
    )

    # Update with new field — should merge
    store.upsert_entity(
        entity_id="proj-hermes",
        entity_type="project",
        name="Hermes Agent",
        profile_json={"language": "Python", "status": "production"},
    )

    entity = store.get_entity("proj-hermes")
    assert entity["profile_json"]["description"] == "AI agent framework"  # preserved
    assert entity["profile_json"]["status"] == "production"  # overwritten
    assert entity["profile_json"]["language"] == "Python"  # added


def test_upsert_entity_list_merge(store):
    store.upsert_entity(
        entity_id="person-jefe",
        entity_type="person",
        name="Jefe",
        profile_json={"interests": ["AI", "automation"]},
    )

    store.upsert_entity(
        entity_id="person-jefe",
        entity_type="person",
        name="Jefe",
        profile_json={"interests": ["robotics", "AI"]},  # AI is duplicate
    )

    entity = store.get_entity("person-jefe")
    assert "AI" in entity["profile_json"]["interests"]
    assert "automation" in entity["profile_json"]["interests"]
    assert "robotics" in entity["profile_json"]["interests"]
    # No duplicates
    assert entity["profile_json"]["interests"].count("AI") == 1


def test_search_entities(store):
    store.upsert_entity(
        entity_id="proj-hermes",
        entity_type="project",
        name="Hermes Agent",
        profile_json={"description": "AI agent framework"},
    )
    store.upsert_entity(
        entity_id="proj-asklepios",
        entity_type="project",
        name="Asklepios",
        profile_json={"description": "PDF pipeline for lab equipment"},
    )

    results = store.search_entities("Hermes")
    assert len(results) >= 1
    assert results[0]["name"] == "Hermes Agent"


def test_confirm_entity(store):
    store.upsert_entity(
        entity_id="proj-hermes",
        entity_type="project",
        name="Hermes Agent",
        profile_json={},
    )
    entity = store.get_entity("proj-hermes")
    assert entity["last_confirmed_at"] is None

    store.confirm_entity("proj-hermes")
    entity = store.get_entity("proj-hermes")
    assert entity["last_confirmed_at"] is not None
    assert entity["last_confirmed_at"] > 0


# ── DAG node operations ───────────────────────────────────────────────────


def test_dag_node(store):
    store.create_dag_node(
        node_id="dag-1",
        parent_ids=[],
        depth=0,
        content="Root summary of session block",
        source_range={"turn_start": 1, "turn_end": 10},
    )

    node = store.get_dag_node("dag-1")
    assert node["content"] == "Root summary of session block"
    assert node["depth"] == 0
    assert node["parent_ids"] == []
    assert node["source_range"]["turn_start"] == 1


def test_dag_node_with_parents(store):
    store.create_dag_node("dag-1", [], 0, "Root")
    store.create_dag_node("dag-2", [], 0, "Root 2")
    store.create_dag_node("dag-3", ["dag-1", "dag-2"], 1, "Condensed from dag-1 and dag-2")

    node = store.get_dag_node("dag-3")
    assert node["parent_ids"] == ["dag-1", "dag-2"]
    assert node["depth"] == 1


def test_get_dag_nodes_at_depth(store):
    store.create_dag_node("dag-1", [], 0, "Root 1")
    store.create_dag_node("dag-2", [], 0, "Root 2")
    store.create_dag_node("dag-3", ["dag-1", "dag-2"], 1, "Level 1")

    depth_0 = store.get_dag_nodes_at_depth(0)
    assert len(depth_0) == 2

    depth_1 = store.get_dag_nodes_at_depth(1)
    assert len(depth_1) == 1


# ── Health check ──────────────────────────────────────────────────────────


def test_health_check(store):
    status = store.health_check()
    assert status["db_writable"] is True
    assert status["db_readable"] is True
    assert status["round_trip"] is True
    assert status["fts_episodes"] is True
    assert status["fts_entities"] is True
    assert status["error"] is None


def test_get_health(store):
    store.health_check()
    health = store.get_health()
    assert health is not None
    assert health["status"]["round_trip"] is True
    assert health["checked_at"] > 0


# ── Stats ─────────────────────────────────────────────────────────────────


def test_get_stats(store):
    stats = store.get_stats()
    assert "turns" in stats
    assert "episodes" in stats
    assert "entities" in stats
    assert "dag_nodes" in stats
    assert "sessions" in stats
    assert "db_size_bytes" in stats
    assert "db_size_mb" in stats

    # All zeros on fresh store
    assert stats["turns"] == 0
    assert stats["episodes"] == 0
    assert stats["entities"] == 0
    assert stats["dag_nodes"] == 0


def test_stats_populated(store):
    store.ensure_session("s1")
    store.append_turn("s1", "user", "test")
    store.create_episode("s1", "topic", "summary")
    store.upsert_entity("e1", "project", "Test", {})

    stats = store.get_stats()
    assert stats["turns"] == 1
    assert stats["episodes"] == 1
    assert stats["entities"] == 1
    assert stats["sessions"] == 1


# ── Utility tests ─────────────────────────────────────────────────────────


def test_merge_profiles_basic():
    old = {"name": "test", "status": "active"}
    new = {"status": "complete", "owner": "jefe"}
    merged = _merge_profiles(old, new)
    assert merged["name"] == "test"  # preserved
    assert merged["status"] == "complete"  # overwritten
    assert merged["owner"] == "jefe"  # added


def test_merge_profiles_nested():
    old = {"settings": {"debug": True, "verbose": False}}
    new = {"settings": {"verbose": True, "timeout": 30}}
    merged = _merge_profiles(old, new)
    assert merged["settings"]["debug"] is True  # preserved
    assert merged["settings"]["verbose"] is True  # overwritten
    assert merged["settings"]["timeout"] == 30  # added


def test_merge_profiles_dedup_lists():
    old = {"tags": ["python", "ai"]}
    new = {"tags": ["ai", "ml"]}
    merged = _merge_profiles(old, new)
    assert merged["tags"] == ["python", "ai", "ml"]
    assert merged["tags"].count("ai") == 1
