"""Tests for Card Store and card management tools."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for all card store tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


# ── Card Store tests ─────────────────────────────────────────────────

class TestCardStoreInit:
    """Test CardStore initialization."""

    def test_creates_db_file(self, _isolate_hermes_home):
        """CardStore should create card_store.db in HERMES_HOME."""
        from agent.card_store import CardStore
        store = CardStore()
        assert store._db_path.exists()
        assert store._db_path.name == "card_store.db"

    def test_profile_aware(self, tmp_path, monkeypatch):
        """Each profile should get its own database."""
        home_a = tmp_path / "profile_a" / ".hermes"
        home_a.mkdir(parents=True)
        home_b = tmp_path / "profile_b" / ".hermes"
        home_b.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        from agent.card_store import CardStore
        store_a = CardStore()
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        store_b = CardStore()
        assert store_a._db_path != store_b._db_path
        assert store_a._db_path.parent == home_a
        assert store_b._db_path.parent == home_b


class TestMemoryCardCRUD:
    """Test memory card CRUD operations."""

    def test_create_and_get(self, _isolate_hermes_home):
        """Should create and retrieve a memory card."""
        from agent.card_store import CardStore
        store = CardStore()
        cid = store.create_memory_card(
            card_type="decision",
            title="Use SQLite for card store",
            body="We decided to use SQLite because it's simple and profile-aware.",
            tags=["database", "architecture"],
            project="hermes-agent",
        )
        card = store.get_memory_card(cid)
        assert card is not None
        assert card["type"] == "decision"
        assert card["title"] == "Use SQLite for card store"
        assert "database" in json.loads(card["tags"])

    def test_create_invalid_type(self, _isolate_hermes_home):
        """Should reject invalid card types."""
        from agent.card_store import CardStore
        store = CardStore()
        with pytest.raises(ValueError, match="Invalid memory card type"):
            store.create_memory_card(card_type="invalid", title="T", body="B")

    def test_update(self, _isolate_hermes_home):
        """Should update card fields."""
        from agent.card_store import CardStore
        store = CardStore()
        cid = store.create_memory_card(
            card_type="rule", title="Old title", body="Old body"
        )
        ok = store.update_memory_card(cid, title="New title", pinned=True)
        assert ok
        card = store.get_memory_card(cid)
        assert card["title"] == "New title"
        assert card["pinned"] == 1

    def test_delete(self, _isolate_hermes_home):
        """Should delete a card."""
        from agent.card_store import CardStore
        store = CardStore()
        cid = store.create_memory_card(card_type="incident", title="T", body="B")
        ok = store.delete_memory_card(cid)
        assert ok
        assert store.get_memory_card(cid) is None

    def test_list_with_pagination(self, _isolate_hermes_home):
        """Should list cards with pagination."""
        from agent.card_store import CardStore
        store = CardStore()
        for i in range(5):
            store.create_memory_card(card_type="decision", title=f"T{i}", body=f"B{i}")
        cards, total = store.list_memory_cards(limit=2, offset=0)
        assert len(cards) == 2
        assert total == 5
        cards2, total2 = store.list_memory_cards(limit=2, offset=2)
        assert len(cards2) == 2
        assert total2 == 5


class TestKnowledgeCardCRUD:
    """Test knowledge card CRUD operations."""

    def test_create_and_get(self, _isolate_hermes_home):
        """Should create and retrieve a knowledge card."""
        from agent.card_store import CardStore
        store = CardStore()
        cid = store.create_knowledge_card(
            title="Docker healthcheck pattern",
            body="Use HEALTHCHECK with curl for HTTP services.",
            source="project-a",
            domains=["devops", "backend"],
            truth_level="verified",
            origin_project="tech-tools-hermes-agent",
        )
        card = store.get_knowledge_card(cid)
        assert card is not None
        assert card["truth_level"] == "verified"
        assert "devops" in json.loads(card["domains"])

    def test_create_invalid_truth_level(self, _isolate_hermes_home):
        """Should reject invalid truth levels."""
        from agent.card_store import CardStore
        store = CardStore()
        with pytest.raises(ValueError, match="Invalid truth level"):
            store.create_knowledge_card(title="T", body="B", truth_level="fake")

    def test_update_review_status(self, _isolate_hermes_home):
        """Should update review status."""
        from agent.card_store import CardStore
        store = CardStore()
        cid = store.create_knowledge_card(title="T", body="B")
        ok = store.update_knowledge_card(cid, review_status="approved")
        assert ok
        card = store.get_knowledge_card(cid)
        assert card["review_status"] == "approved"


class TestCardSearch:
    """Test FTS5 search."""

    def test_search_memory_cards(self, _isolate_hermes_home):
        """Should find memory cards by FTS5 search."""
        from agent.card_store import CardStore
        store = CardStore()
        store.create_memory_card(
            card_type="decision",
            title="Database choice",
            body="We chose SQLite over PostgreSQL for simplicity",
        )
        results = store.search_cards("SQLite database")
        assert len(results) >= 1
        assert results[0]["card_type"] == "memory"

    def test_search_knowledge_cards(self, _isolate_hermes_home):
        """Should find knowledge cards by FTS5 search."""
        from agent.card_store import CardStore
        store = CardStore()
        store.create_knowledge_card(
            title="Docker healthcheck",
            body="Use HEALTHCHECK instruction in Dockerfile for service monitoring",
            domains=["devops"],
        )
        results = store.search_cards("Docker healthcheck monitoring")
        found = [r for r in results if r["card_type"] == "knowledge"]
        assert len(found) >= 1

    def test_search_empty_query(self, _isolate_hermes_home):
        """Empty query should return empty results."""
        from agent.card_store import CardStore
        store = CardStore()
        store.create_memory_card(card_type="rule", title="T", body="B")
        results = store.search_cards("")
        assert results == []


class TestDuplicateDetection:
    """Test duplicate card detection."""

    def test_exact_hash_match(self, _isolate_hermes_home):
        """Identical body should be detected as duplicate."""
        from agent.card_store import CardStore
        store = CardStore()
        body = "This is a specific rule about Docker volumes."
        store.create_memory_card(card_type="rule", title="Rule 1", body=body)
        candidates = store.find_duplicate_memory_cards("rule", "Rule 2", body)
        assert len(candidates) >= 1
        assert candidates[0]["similarity"] == 1.0

    def test_keyword_overlap(self, _isolate_hermes_home):
        """High keyword overlap should be detected as duplicate."""
        from agent.card_store import CardStore
        store = CardStore()
        store.create_memory_card(
            card_type="rule",
            title="Docker volume management",
            body="Always use named volumes for persistent data in Docker containers",
        )
        candidates = store.find_duplicate_memory_cards(
            "rule",
            "Volume best practice",
            "Use named volumes for Docker container persistent data management",
        )
        assert len(candidates) >= 1
        assert candidates[0]["similarity"] > 0.5

    def test_no_duplicate(self, _isolate_hermes_home):
        """Unrelated content should not be flagged as duplicate."""
        from agent.card_store import CardStore
        store = CardStore()
        store.create_memory_card(
            card_type="rule",
            title="Python type hints",
            body="Always use type hints for function signatures",
        )
        candidates = store.find_duplicate_memory_cards(
            "rule",
            "Docker networking",
            "Use bridge networks for inter-container communication",
        )
        assert len(candidates) == 0


class TestContextPackLog:
    """Test context pack selection logging."""

    def test_log_and_retrieve(self, _isolate_hermes_home):
        """Should log and retrieve context pack selections."""
        from agent.card_store import CardStore
        store = CardStore()
        eid = store.log_context_pack_selection(
            session_id="sess-1",
            card_id="card-1",
            card_type="memory",
            tier="tier-2",
            reason="Domain match",
            relevance_score=0.8,
            token_cost=150,
        )
        assert eid
        trace = store.get_context_pack_trace("sess-1")
        assert len(trace) == 1
        assert trace[0]["card_id"] == "card-1"
        assert trace[0]["reason"] == "Domain match"


class TestMetricsSnapshots:
    """Test metrics snapshot storage."""

    def test_save_and_retrieve(self, _isolate_hermes_home):
        """Should save and retrieve metrics snapshots."""
        from agent.card_store import CardStore
        store = CardStore()
        store.save_metrics_snapshot(
            snapshot_type="weekly_growth",
            period="2026-W21",
            data={"cards_created": 5, "approval_rate": 0.8},
        )
        snapshots = store.get_metrics_snapshots("weekly_growth")
        assert len(snapshots) == 1
        assert snapshots[0]["data"]["cards_created"] == 5


# ── Tool tests ───────────────────────────────────────────────────────

class TestMemoryCardTool:
    """Test manage_memory_card tool."""

    def test_create_via_tool(self, _isolate_hermes_home):
        """Should create a card via tool."""
        from tools.memory_card_tool import manage_memory_card
        result = json.loads(manage_memory_card(
            action="create",
            card_type="decision",
            title="Test decision",
            body="Test body",
        ))
        assert result["success"]
        assert "card_id" in result

    def test_get_via_tool(self, _isolate_hermes_home):
        """Should get a card via tool."""
        from tools.memory_card_tool import manage_memory_card
        create_result = json.loads(manage_memory_card(
            action="create", card_type="rule", title="T", body="B"
        ))
        cid = create_result["card_id"]
        result = json.loads(manage_memory_card(action="get", card_id=cid))
        assert result["success"]
        assert result["card"]["title"] == "T"

    def test_list_via_tool(self, _isolate_hermes_home):
        """Should list cards via tool."""
        from tools.memory_card_tool import manage_memory_card
        for i in range(3):
            manage_memory_card(action="create", card_type="incident", title=f"T{i}", body=f"B{i}")
        result = json.loads(manage_memory_card(action="list"))
        assert result["success"]
        assert result["total"] == 3

    def test_search_via_tool(self, _isolate_hermes_home):
        """Should search cards via tool."""
        from tools.memory_card_tool import manage_memory_card
        manage_memory_card(
            action="create", card_type="decision",
            title="Docker choice", body="We use Docker for containers"
        )
        result = json.loads(manage_memory_card(action="search", query="Docker"))
        assert result["success"]
        assert result["count"] >= 1

    def test_pin_unpin_via_tool(self, _isolate_hermes_home):
        """Should pin and unpin cards via tool."""
        from tools.memory_card_tool import manage_memory_card
        create_result = json.loads(manage_memory_card(
            action="create", card_type="rule", title="T", body="B"
        ))
        cid = create_result["card_id"]
        pin_result = json.loads(manage_memory_card(action="pin", card_id=cid))
        assert pin_result["success"]
        get_result = json.loads(manage_memory_card(action="get", card_id=cid))
        assert get_result["card"]["pinned"] == 1
        unpin_result = json.loads(manage_memory_card(action="unpin", card_id=cid))
        assert unpin_result["success"]

    def test_invalid_action(self):
        """Should return error for invalid action."""
        from tools.memory_card_tool import manage_memory_card
        result = json.loads(manage_memory_card(action="invalid"))
        assert not result["success"]


class TestKnowledgeCardTool:
    """Test manage_knowledge_card tool."""

    def test_create_via_tool(self, _isolate_hermes_home):
        """Should create a knowledge card via tool."""
        from tools.knowledge_card_tool import manage_knowledge_card
        result = json.loads(manage_knowledge_card(
            action="create",
            title="Test knowledge",
            body="Test body",
            truth_level="verified",
            domains='["backend"]',
        ))
        assert result["success"]
        assert "card_id" in result

    def test_review_actions(self, _isolate_hermes_home):
        """Should approve/reject/defer via tool."""
        from tools.knowledge_card_tool import manage_knowledge_card
        create_result = json.loads(manage_knowledge_card(
            action="create", title="T", body="B"
        ))
        cid = create_result["card_id"]
        approve = json.loads(manage_knowledge_card(action="approve", card_id=cid))
        assert approve["success"]
        get_result = json.loads(manage_knowledge_card(action="get", card_id=cid))
        assert get_result["card"]["review_status"] == "approved"

    def test_list_with_review_filter(self, _isolate_hermes_home):
        """Should list cards filtered by review status."""
        from tools.knowledge_card_tool import manage_knowledge_card
        manage_knowledge_card(action="create", title="T1", body="B1")
        create2 = json.loads(manage_knowledge_card(action="create", title="T2", body="B2"))
        manage_knowledge_card(action="approve", card_id=create2["card_id"])
        result = json.loads(manage_knowledge_card(
            action="list", review_status="pending_review"
        ))
        assert result["total"] == 1  # Only the unapproved one
