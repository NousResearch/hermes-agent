"""Tests for the Dream Engine — autonomous background memory consolidation."""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure plugin modules are importable
import sys
_PLUGIN_DIR = Path(__file__).resolve().parents[3] / "plugins" / "memory" / "neural"
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from dream_engine import (
    DreamEngine,
    DreamBackend,
    SQLiteDreamBackend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with memories and connections."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            content TEXT,
            embedding BLOB,
            salience REAL DEFAULT 1.0,
            created_at REAL DEFAULT 0,
            last_accessed REAL DEFAULT 0,
            access_count INTEGER DEFAULT 0
        );
        CREATE TABLE connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            target_id INTEGER,
            weight REAL DEFAULT 0.5,
            created_at REAL DEFAULT 0
        );
    """)
    yield db_path
    conn.close()
    os.unlink(db_path)


def _seed_memories(conn, count=10):
    """Insert test memories."""
    now = time.time()
    for i in range(count):
        conn.execute(
            "INSERT INTO memories (label, content, created_at) VALUES (?, ?, ?)",
            (f"mem-{i}", f"Test memory content about topic {i % 3}", now - i)
        )
    conn.commit()


def _seed_connections(conn, edges):
    """Insert test connections. edges = [(src, tgt, weight), ...]"""
    for src, tgt, w in edges:
        conn.execute(
            "INSERT INTO connections (source_id, target_id, weight, created_at) "
            "VALUES (?, ?, ?, ?)",
            (src, tgt, w, time.time())
        )
    conn.commit()


# ---------------------------------------------------------------------------
# SQLiteDreamBackend Tests
# ---------------------------------------------------------------------------

class TestSQLiteDreamBackend:
    """Test the SQLite dream backend."""

    def test_ensure_tables(self, temp_db):
        """Dream tables should be created automatically."""
        backend = SQLiteDreamBackend(temp_db)
        conn = sqlite3.connect(temp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "dream_sessions" in table_names
        assert "dream_insights" in table_names
        assert "connection_history" in table_names
        conn.close()

    def test_start_finish_session(self, temp_db):
        """Session start returns ID, finish updates stats."""
        backend = SQLiteDreamBackend(temp_db)
        sid = backend.start_session("nrem")
        assert sid > 0

        backend.finish_session(sid, {
            "processed": 5, "strengthened": 3,
            "pruned": 1, "bridges": 2, "insights": 0
        })

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT * FROM dream_sessions WHERE id=?", (sid,)).fetchone()
        assert row is not None
        conn.close()

    def test_get_recent_memories(self, temp_db):
        """Should return memories in reverse chronological order."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 15)
        conn.close()

        backend = SQLiteDreamBackend(temp_db)
        mems = backend.get_recent_memories(limit=5)
        assert len(mems) == 5
        assert all("id" in m and "content" in m for m in mems)

    def test_get_isolated_memories(self, temp_db):
        """Should find memories with few connections."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 5)
        # Connect 1-2-3, leave 4-5 isolated
        _seed_connections(conn, [(1, 2, 0.8), (2, 3, 0.7)])
        conn.close()

        backend = SQLiteDreamBackend(temp_db)
        isolated = backend.get_isolated_memories(max_connections=2, limit=10)
        # Should find memories with < 2 connections
        assert len(isolated) > 0

    def test_strengthen_connection(self, temp_db):
        """Strengthen should increase weight."""
        conn = sqlite3.connect(temp_db)
        _seed_connections(conn, [(1, 2, 0.5)])
        conn.close()

        backend = SQLiteDreamBackend(temp_db)
        backend.strengthen_connection(1, 2, 0.1)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT weight FROM connections WHERE source_id=1").fetchone()
        assert row[0] == pytest.approx(0.6, abs=0.01)
        conn.close()

    def test_weaken_connection(self, temp_db):
        """Weaken should decrease weight."""
        conn = sqlite3.connect(temp_db)
        _seed_connections(conn, [(1, 2, 0.5)])
        conn.close()

        backend = SQLiteDreamBackend(temp_db)
        backend.weaken_connection(1, 2, 0.1)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT weight FROM connections WHERE source_id=1").fetchone()
        assert row[0] == pytest.approx(0.4, abs=0.01)
        conn.close()

    def test_add_bridge(self, temp_db):
        """Adding a bridge creates a new connection."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 5)
        conn.close()

        backend = SQLiteDreamBackend(temp_db)
        backend.add_bridge(1, 3, 0.25)

        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT weight FROM connections WHERE source_id=1 AND target_id=3"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.25, abs=0.01)
        conn.close()

    def test_add_bridge_no_duplicate(self, temp_db):
        """Adding existing bridge should not duplicate."""
        conn = sqlite3.connect(temp_db)
        _seed_connections(conn, [(1, 3, 0.5)])
        conn.close()

        backend = SQLiteDreamBackend(temp_db)
        backend.add_bridge(1, 3, 0.3)

        conn = sqlite3.connect(temp_db)
        rows = conn.execute(
            "SELECT COUNT(*) FROM connections WHERE source_id=1 AND target_id=3"
        ).fetchone()
        assert rows[0] == 1
        conn.close()

    def test_prune_weak(self, temp_db):
        """Prune should delete connections below threshold."""
        conn = sqlite3.connect(temp_db)
        _seed_connections(conn, [(1, 2, 0.02), (3, 4, 0.8), (5, 6, 0.04)])
        conn.close()

        backend = SQLiteDreamBackend(temp_db)
        pruned = backend.prune_weak(0.05)
        assert pruned == 2

        remaining = backend.get_connections()
        assert len(remaining) == 1
        assert remaining[0]["source_id"] == 3

    def test_log_connection_change(self, temp_db):
        """Connection changes should be logged."""
        backend = SQLiteDreamBackend(temp_db)
        backend.log_connection_change(1, 2, 0.5, 0.55, "nrem_strengthen")

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT * FROM connection_history").fetchone()
        assert row is not None
        conn.close()

    def test_add_insight(self, temp_db):
        """Insights should be stored."""
        backend = SQLiteDreamBackend(temp_db)
        sid = backend.start_session("insight")
        backend.add_insight(sid, "cluster", 1, "Test cluster insight", 0.7)

        stats = backend.get_dream_stats()
        assert stats["insight_types"].get("cluster", 0) >= 1

    def test_get_dream_stats_empty(self, temp_db):
        """Stats should work with no sessions."""
        backend = SQLiteDreamBackend(temp_db)
        stats = backend.get_dream_stats()
        assert stats["sessions"] == 0


# ---------------------------------------------------------------------------
# DreamEngine Tests
# ---------------------------------------------------------------------------

class TestDreamEngine:
    """Test the Dream Engine core logic."""

    def test_create_sqlite(self, temp_db):
        """Factory method should create engine with SQLite backend."""
        engine = DreamEngine.sqlite(temp_db, idle_threshold=60)
        assert engine is not None
        assert isinstance(engine._backend, SQLiteDreamBackend)

    def test_touch_resets_idle(self, temp_db):
        """Touch should update last activity time."""
        engine = DreamEngine.sqlite(temp_db, idle_threshold=10)
        old_time = engine._last_activity
        time.sleep(0.1)
        engine.touch()
        assert engine._last_activity > old_time

    def test_dream_now_returns_stats(self, temp_db):
        """Forced dream should return stats dict."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 5)
        _seed_connections(conn, [(1, 2, 0.8), (2, 3, 0.6), (3, 4, 0.04)])
        conn.close()

        engine = DreamEngine.sqlite(temp_db, idle_threshold=9999)
        stats = engine.dream_now()
        assert "nrem" in stats
        assert "rem" in stats
        assert "insights" in stats
        assert "duration" in stats

    def test_nrem_phase(self, temp_db):
        """NREM should process connections."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 5)
        _seed_connections(conn, [(1, 2, 0.8), (2, 3, 0.6), (3, 4, 0.04)])
        conn.close()

        engine = DreamEngine.sqlite(temp_db, idle_threshold=9999)
        stats = engine._phase_nrem()
        assert "processed" in stats
        assert stats["processed"] > 0

    def test_rem_phase(self, temp_db):
        """REM should explore isolated memories."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 10)
        # Connect only a few, leave many isolated
        _seed_connections(conn, [(1, 2, 0.8)])
        conn.close()

        engine = DreamEngine.sqlite(temp_db, idle_threshold=9999)
        stats = engine._phase_rem()
        assert "explored" in stats

    def test_insight_phase(self, temp_db):
        """Insight phase should detect communities."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 10)
        # Two clusters
        _seed_connections(conn, [
            (1, 2, 0.8), (2, 3, 0.7), (3, 1, 0.6),  # cluster A
            (5, 6, 0.8), (6, 7, 0.7), (7, 5, 0.6),  # cluster B
        ])
        conn.close()

        engine = DreamEngine.sqlite(temp_db, idle_threshold=9999)
        stats = engine._phase_insights()
        assert stats["communities"] >= 2

    def test_start_stop_daemon(self, temp_db):
        """Engine should start and stop cleanly."""
        engine = DreamEngine.sqlite(temp_db, idle_threshold=9999)
        engine.start()
        assert engine._running is True
        assert engine._thread is not None
        engine.stop()
        assert engine._running is False

    def test_get_stats(self, temp_db):
        """get_stats should return engine and backend stats."""
        engine = DreamEngine.sqlite(temp_db, idle_threshold=9999)
        stats = engine.get_stats()
        assert "engine_running" in stats
        assert "dream_cycles" in stats
        assert stats["engine_running"] is False

    def test_empty_db_dream(self, temp_db):
        """Dreaming on empty DB should not crash."""
        engine = DreamEngine.sqlite(temp_db, idle_threshold=9999)
        stats = engine.dream_now()
        assert "nrem" in stats
        # Should complete without error


# ---------------------------------------------------------------------------
# Integration: Dream with Mock NeuralMemory
# ---------------------------------------------------------------------------

class TestDreamIntegration:
    """Test dream engine with mock neural memory."""

    def test_nrem_with_think(self, temp_db):
        """NREM should use neural_think for spreading activation."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 5)
        _seed_connections(conn, [(1, 2, 0.8), (2, 3, 0.6)])
        conn.close()

        mock_memory = MagicMock()
        mock_memory.think.return_value = [{"id": 2}, {"id": 3}]
        mock_memory.stats.return_value = {"memories": 5, "connections": 2}

        engine = DreamEngine.sqlite(temp_db, mock_memory, idle_threshold=9999)
        stats = engine._phase_nrem()

        assert mock_memory.think.called
        assert stats["processed"] > 0

    def test_rem_with_recall(self, temp_db):
        """REM should use neural_recall for bridge discovery."""
        conn = sqlite3.connect(temp_db)
        _seed_memories(conn, 10)
        _seed_connections(conn, [(1, 2, 0.8)])
        conn.close()

        mock_memory = MagicMock()
        mock_memory.recall.return_value = [
            {"id": 5, "similarity": 0.7},
            {"id": 3, "similarity": 0.5},
        ]
        mock_memory.stats.return_value = {"memories": 10, "connections": 1}

        engine = DreamEngine.sqlite(temp_db, mock_memory, idle_threshold=9999)
        stats = engine._phase_rem()

        assert stats["explored"] > 0
