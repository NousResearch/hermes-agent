"""Tests for TieredMemoryDB — SQLite + FTS5 + sqlite-vec store."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlite_vec
from sqlite_vec import serialize_float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db():
    """Create a TieredMemoryDB backed by a temp file (sqlite-vec needs file, not :memory:)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch("plugins.memory.tiered.store.scrub_content", side_effect=lambda t, **kw: t):
            with patch("plugins.memory.tiered.store.generate_embedding", return_value=None):
                from plugins.memory.tiered.store import TieredMemoryDB
                store = TieredMemoryDB(db_path)
                yield store
                store.close()


@pytest.fixture()
def db_with_vec():
    """Create a TieredMemoryDB that generates deterministic embeddings."""
    _call_count = 0

    def _fake_embedding(text: str):
        nonlocal _call_count
        _call_count += 1
        # Generate a deterministic 1024-dim vector based on the text hash
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h]
        # Pad to 1024 dims
        vec = (vec * (1024 // len(vec) + 1))[:1024]
        return vec

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_vec.db"
        with patch("plugins.memory.tiered.store.scrub_content", side_effect=lambda t, **kw: t):
            with patch("plugins.memory.tiered.store.generate_embedding", side_effect=_fake_embedding):
                from plugins.memory.tiered.store import TieredMemoryDB
                store = TieredMemoryDB(db_path)
                yield store
                store.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInsertAndRetrieve:
    def test_insert_and_retrieve(self, db):
        """Insert entry, verify it exists in memory_entries."""
        entry_id = db.insert_entry("Test Title", "Test content for memory", "project")
        assert entry_id  # non-empty string
        assert len(entry_id) == 32  # hex(randomblob(16)) = 32 chars

        row = db._conn.execute(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row is not None
        assert row["title"] == "Test Title"
        assert row["content"] == "Test content for memory"
        assert row["category"] == "project"
        assert row["archived_at"] is None

    def test_insert_sets_expires_at_for_ttl_categories(self, db):
        """Categories with TTL should set expires_at."""
        entry_id = db.insert_entry("Chat entry", "Hello world", "chat")
        row = db._conn.execute(
            "SELECT expires_at FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row["expires_at"] is not None

    def test_insert_no_expires_for_principle(self, db):
        """Principle category has no TTL."""
        entry_id = db.insert_entry("Core value", "Always be kind", "principle")
        row = db._conn.execute(
            "SELECT expires_at FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row["expires_at"] is None


class TestDedup:
    def test_dedup_by_content_hash(self, db):
        """Insert same content twice, get same id back."""
        id1 = db.insert_entry("Title A", "Duplicate content here", "project")
        id2 = db.insert_entry("Title B", "Duplicate content here", "project")
        assert id1 == id2

        count = db._conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
        assert count == 1

    def test_different_content_different_ids(self, db):
        """Different content should produce different entries."""
        id1 = db.insert_entry("Title A", "Content one", "project")
        id2 = db.insert_entry("Title B", "Content two", "project")
        assert id1 != id2


class TestFTSSearch:
    def test_fts_search(self, db):
        """Insert entries with different content, FTS search finds matching ones."""
        db.insert_entry("Python Guide", "How to use Python decorators effectively", "project")
        db.insert_entry("Rust Guide", "Memory management in Rust with ownership", "project")
        db.insert_entry("Python Tips", "Python list comprehensions are powerful", "project")

        results = db.search_fts("Python")
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert "Python Guide" in titles
        assert "Python Tips" in titles

    def test_fts_search_empty_query(self, db):
        """Empty query returns empty list."""
        db.insert_entry("Title", "Content", "project")
        results = db.search_fts("")
        assert results == []

    def test_fts_search_with_category_filter(self, db):
        """Category filter limits results."""
        db.insert_entry("Chat msg", "Python in chat context", "chat")
        db.insert_entry("Project note", "Python in project context", "project")

        results = db.search_fts("Python", category="chat")
        assert len(results) == 1
        assert results[0]["category"] == "chat"

    def test_fts_excludes_archived(self, db):
        """Archived entries should not appear in FTS search."""
        entry_id = db.insert_entry("Archived Note", "This should be hidden", "project")
        db.archive_entry(entry_id)

        results = db.search_fts("hidden")
        assert len(results) == 0


class TestVecSearch:
    def test_vec_search(self, db_with_vec):
        """Insert entries with embeddings, KNN search finds them."""
        db_with_vec.insert_entry("Memory one", "First semantic memory about cats", "project")
        db_with_vec.insert_entry("Memory two", "Second semantic memory about dogs", "project")

        # Use a known embedding for query — just use a deterministic vector
        import hashlib
        query_text = "cats and animals"
        h = hashlib.sha256(query_text.encode()).digest()
        query_vec = [float(b) / 255.0 for b in h]
        query_vec = (query_vec * (1024 // len(query_vec) + 1))[:1024]

        results = db_with_vec.search_vec(query_vec, limit=5)
        assert len(results) >= 1
        # All results should have a distance field
        for r in results:
            assert "distance" in r
            assert "id" in r
            assert "title" in r

    def test_vec_search_excludes_archived(self, db_with_vec):
        """Archived entries should not appear in vector search."""
        entry_id = db_with_vec.insert_entry("To Archive", "Semantic content here", "project")
        db_with_vec.archive_entry(entry_id)

        import hashlib
        h = hashlib.sha256(b"Semantic content here").digest()
        query_vec = [float(b) / 255.0 for b in h]
        query_vec = (query_vec * (1024 // len(query_vec) + 1))[:1024]

        results = db_with_vec.search_vec(query_vec, limit=5)
        ids = {r["id"] for r in results}
        assert entry_id not in ids


class TestArchive:
    def test_archive_entry(self, db):
        """Archive entry, verify excluded from search."""
        entry_id = db.insert_entry("Note to archive", "Archive me please", "project")
        assert db.archive_entry(entry_id) is True

        row = db._conn.execute(
            "SELECT archived_at FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row["archived_at"] is not None

    def test_archive_nonexistent(self, db):
        """Archiving non-existent id returns False."""
        assert db.archive_entry("nonexistent_id_12345678") is False

    def test_archive_already_archived(self, db):
        """Archiving already-archived entry returns False."""
        entry_id = db.insert_entry("Double archive", "Test content", "project")
        assert db.archive_entry(entry_id) is True
        assert db.archive_entry(entry_id) is False


class TestDeleteExpired:
    def test_delete_expired(self, db):
        """Insert with short TTL (manipulate expires_at directly), verify deletion."""
        entry_id = db.insert_entry("Expiring entry", "Will expire soon", "chat")

        # Manually set expires_at to the past
        db._conn.execute(
            "UPDATE memory_entries SET expires_at = datetime('now', '-1 day') WHERE id = ?",
            (entry_id,),
        )
        db._conn.commit()

        count = db.delete_expired()
        assert count == 1

        row = db._conn.execute(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row is None

    def test_delete_expired_no_expired(self, db):
        """No expired entries returns 0."""
        db.insert_entry("Future entry", "Not expired yet", "project")
        count = db.delete_expired()
        assert count == 0

    def test_delete_expired_leaves_unexpired(self, db):
        """Only expired entries are deleted; unexpired ones remain."""
        id_expired = db.insert_entry("Old entry", "Old content", "chat")
        id_fresh = db.insert_entry("New entry", "Fresh content", "project")

        db._conn.execute(
            "UPDATE memory_entries SET expires_at = datetime('now', '-1 day') WHERE id = ?",
            (id_expired,),
        )
        db._conn.commit()

        db.delete_expired()

        assert db._conn.execute(
            "SELECT id FROM memory_entries WHERE id = ?", (id_expired,)
        ).fetchone() is None
        assert db._conn.execute(
            "SELECT id FROM memory_entries WHERE id = ?", (id_fresh,)
        ).fetchone() is not None


class TestStats:
    def test_stats_updated(self, db):
        """Insert entries, check stats reflect correct counts."""
        stats_before = db.get_stats()
        assert stats_before["entry_count"] == 0
        assert stats_before["total_bytes"] == 0

        db.insert_entry("Entry 1", "Content one", "project")
        db.insert_entry("Entry 2", "Content two", "project")

        stats_after = db.get_stats()
        assert stats_after["entry_count"] == 2
        assert stats_after["total_bytes"] > 0

    def test_stats_exclude_archived(self, db):
        """Archived entries are excluded from entry_count in stats after next insert triggers update."""
        db.insert_entry("Keep", "Keep this", "project")
        entry_id = db.insert_entry("Archive", "Archive this", "project")
        db.archive_entry(entry_id)

        # Stats are updated on insert; trigger by inserting another
        db.insert_entry("New", "New content", "project")
        stats = db.get_stats()
        assert stats["entry_count"] == 2  # Keep + New, not Archive


class TestGetRecentEntries:
    def test_get_recent_entries(self, db):
        """Insert entries, verify returned."""
        db.insert_entry("Recent note", "This is recent content", "project")
        db.insert_entry("Another note", "More recent content", "chat")

        results = db.get_recent_entries(hours=48)
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert "Recent note" in titles
        assert "Another note" in titles

    def test_get_recent_entries_excludes_archived(self, db):
        """Archived entries not returned."""
        entry_id = db.insert_entry("Archived note", "Old content", "project")
        db.insert_entry("Active note", "New content", "project")
        db.archive_entry(entry_id)

        results = db.get_recent_entries(hours=48)
        assert len(results) == 1
        assert results[0]["title"] == "Active note"

    def test_get_recent_entries_excludes_category(self, db):
        """exclude_category filter works."""
        db.insert_entry("Chat msg", "Chat content", "chat")
        db.insert_entry("Project note", "Project content", "project")

        results = db.get_recent_entries(hours=48, exclude_category="chat")
        assert len(results) == 1
        assert results[0]["category"] == "project"

    def test_get_recent_entries_empty_when_old(self, db):
        """Entries older than window not returned."""
        entry_id = db.insert_entry("Old note", "Ancient content", "project")
        # Manually set created_at to 72 hours ago
        db._conn.execute(
            "UPDATE memory_entries SET created_at = datetime('now', '-72 hours') WHERE id = ?",
            (entry_id,),
        )
        db._conn.commit()

        results = db.get_recent_entries(hours=48)
        assert len(results) == 0


class TestDetectPatterns:
    def test_detect_patterns_finds_repeated(self, db):
        """Insert 3+ entries with same title prefix, verify detected."""
        for i in range(3):
            db.insert_entry("Repeated Topic Here", f"Content variation {i}", "project")

        results = db.detect_patterns(days=7, min_count=3)
        assert len(results) >= 1
        assert results[0]["count"] >= 3
        assert "Repeated Topic Here" in results[0]["topic"]

    def test_detect_patterns_ignores_below_threshold(self, db):
        """2 entries not detected when min_count=3."""
        for i in range(2):
            db.insert_entry("Only Two Times", f"Content {i}", "project")

        results = db.detect_patterns(days=7, min_count=3)
        topics = [r["topic"] for r in results]
        assert not any("Only Two Times" in t for t in topics)

    def test_detect_patterns_excludes_archived(self, db):
        """Archived entries not counted."""
        ids = []
        for i in range(3):
            eid = db.insert_entry("Archive Pattern Test", f"Content {i}", "project")
            ids.append(eid)

        # Archive all three
        for eid in ids:
            db.archive_entry(eid)

        results = db.detect_patterns(days=7, min_count=3)
        topics = [r["topic"] for r in results]
        assert not any("Archive Pattern Test" in t for t in topics)


class TestDeleteNoise:
    def test_delete_noise_removes_greetings(self, db):
        """Insert 'User Greeted' entry, verify deleted."""
        db.insert_entry("User Greeted the agent", "Hello!", "chat")
        db.insert_entry("User greeted with hi", "Hi there!", "chat")
        db.insert_entry(
            "Handled simple greeting after loading workspace context",
            "Context loaded",
            "chat",
        )

        count = db.delete_noise()
        assert count == 3

        remaining = db._conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
        assert remaining == 0

    def test_delete_noise_removes_fallback_diary(self, db):
        """Insert 'Fallback diary:' entry, verify deleted."""
        db.insert_entry("Fallback diary: session ended", "Nothing happened", "chat")

        count = db.delete_noise()
        assert count == 1

    def test_delete_noise_preserves_normal(self, db):
        """Normal entries not deleted."""
        db.insert_entry("User Greeted the agent", "Hello!", "chat")
        db.insert_entry("Important project note", "Keep this around", "project")

        count = db.delete_noise()
        assert count == 1

        remaining = db._conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
        assert remaining == 1
        row = db._conn.execute("SELECT title FROM memory_entries").fetchone()
        assert row["title"] == "Important project note"


class TestArchiveStale:
    def test_archive_stale_archives_old_briefing(self, db):
        """Briefing >7d archived."""
        entry_id = db.insert_entry("Morning briefing", "Today's agenda", "briefing")
        # Set created_at to 10 days ago
        db._conn.execute(
            "UPDATE memory_entries SET created_at = datetime('now', '-10 days') WHERE id = ?",
            (entry_id,),
        )
        db._conn.commit()

        count = db.archive_stale()
        assert count == 1

        row = db._conn.execute(
            "SELECT archived_at FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row["archived_at"] is not None

    def test_archive_stale_preserves_principle(self, db):
        """Principle never archived regardless of age."""
        entry_id = db.insert_entry("Core value", "Always be kind", "principle")
        # Set created_at to 1 year ago
        db._conn.execute(
            "UPDATE memory_entries SET created_at = datetime('now', '-365 days') WHERE id = ?",
            (entry_id,),
        )
        db._conn.commit()

        count = db.archive_stale()
        assert count == 0

        row = db._conn.execute(
            "SELECT archived_at FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        assert row["archived_at"] is None

    def test_archive_stale_preserves_recent(self, db):
        """Recent entries not archived."""
        db.insert_entry("Fresh briefing", "Just now", "briefing")
        db.insert_entry("Fresh chat", "Just chatting", "chat")
        db.insert_entry("Fresh project", "Just started", "project")

        count = db.archive_stale()
        assert count == 0


class TestUpdateStats:
    def test_update_stats_public(self, db):
        """Public update_stats recomputes correctly."""
        db.insert_entry("Entry 1", "Content one", "project")
        # Manually corrupt stats
        db._conn.execute(
            "UPDATE memory_stats SET entry_count = 999 WHERE source = 'hermes'"
        )
        db._conn.commit()

        db.update_stats()
        stats = db.get_stats()
        assert stats["entry_count"] == 1


class TestDeduplicateSimilar:
    """Tests for deduplicate_similar() — semantic near-duplicate removal."""

    @pytest.fixture()
    def db_dedup(self):
        """DB with controlled embeddings for dedup tests.

        Patches generate_embedding to return whatever the test injects via
        _next_embedding on the fixture, defaulting to None (no embedding).
        """
        _embedding_queue: list[list[float] | None] = []

        def _fake_embedding(text: str):
            if _embedding_queue:
                return _embedding_queue.pop(0)
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dedup.db"
            with patch("plugins.memory.tiered.store.scrub_content", side_effect=lambda t, **kw: t):
                with patch("plugins.memory.tiered.store.generate_embedding", side_effect=_fake_embedding):
                    from plugins.memory.tiered.store import TieredMemoryDB
                    store = TieredMemoryDB(db_path)
                    store._embedding_queue = _embedding_queue
                    yield store
                    store.close()

    @staticmethod
    def _make_vec(val: float, dim: int = 1024) -> list[float]:
        """Create a 1024-dim vector filled with a single value (then normalized)."""
        import math
        vec = [val] * dim
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

    @staticmethod
    def _make_unit_vec(index: int, dim: int = 1024) -> list[float]:
        """Create a 1024-dim unit vector with 1.0 at `index`, 0 elsewhere."""
        vec = [0.0] * dim
        vec[index % dim] = 1.0
        return vec

    def test_deduplicate_identical_embeddings(self, db_dedup):
        """3 entries with identical embeddings in same category -> only 1 remains (most recent)."""
        vec = self._make_vec(1.0)
        # Queue 3 identical embeddings
        db_dedup._embedding_queue.extend([vec, vec, vec])

        db_dedup.insert_entry("Note A", "Content alpha", "chat")
        db_dedup.insert_entry("Note B", "Content beta", "chat")
        # Make the third entry clearly the most recent
        db_dedup._conn.execute(
            "UPDATE memory_entries SET created_at = datetime('now', '+1 hour') "
            "WHERE title = 'Note C'"
        )
        db_dedup.insert_entry("Note C", "Content gamma", "chat")
        db_dedup._conn.execute(
            "UPDATE memory_entries SET created_at = datetime('now', '+1 hour') "
            "WHERE title = 'Note C'"
        )
        db_dedup._conn.commit()

        count = db_dedup.deduplicate_similar(threshold=0.08)
        assert count == 2

        remaining = db_dedup._conn.execute(
            "SELECT title FROM memory_entries WHERE archived_at IS NULL"
        ).fetchall()
        assert len(remaining) == 1
        assert remaining[0]["title"] == "Note C"

    def test_deduplicate_preserves_different_entries(self, db_dedup):
        """Entries with orthogonal embeddings should all survive."""
        vec_a = self._make_unit_vec(0)
        vec_b = self._make_unit_vec(1)
        vec_c = self._make_unit_vec(2)
        db_dedup._embedding_queue.extend([vec_a, vec_b, vec_c])

        db_dedup.insert_entry("Topic A", "Completely different A", "project")
        db_dedup.insert_entry("Topic B", "Completely different B", "project")
        db_dedup.insert_entry("Topic C", "Completely different C", "project")

        count = db_dedup.deduplicate_similar(threshold=0.08)
        assert count == 0

        remaining = db_dedup._conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE archived_at IS NULL"
        ).fetchone()[0]
        assert remaining == 3

    def test_deduplicate_category_filter(self, db_dedup):
        """Dedup with category='chat' only affects chat entries."""
        vec = self._make_vec(1.0)
        # 2 identical chat + 2 identical project
        db_dedup._embedding_queue.extend([vec, vec, vec, vec])

        db_dedup.insert_entry("Chat 1", "Chat content one", "chat")
        db_dedup.insert_entry("Chat 2", "Chat content two", "chat")
        db_dedup.insert_entry("Proj 1", "Project content one", "project")
        db_dedup.insert_entry("Proj 2", "Project content two", "project")

        count = db_dedup.deduplicate_similar(threshold=0.08, category="chat")
        assert count == 1  # one chat entry removed

        chat_count = db_dedup._conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE category = 'chat' AND archived_at IS NULL"
        ).fetchone()[0]
        assert chat_count == 1

        proj_count = db_dedup._conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE category = 'project' AND archived_at IS NULL"
        ).fetchone()[0]
        assert proj_count == 2

    def test_deduplicate_skips_principle(self, db_dedup):
        """Entries with category='principle' are never deduplicated."""
        vec = self._make_vec(1.0)
        db_dedup._embedding_queue.extend([vec, vec, vec])

        db_dedup.insert_entry("Principle A", "Core value alpha", "principle")
        db_dedup.insert_entry("Principle B", "Core value beta", "principle")
        db_dedup.insert_entry("Principle C", "Core value gamma", "principle")

        count = db_dedup.deduplicate_similar(threshold=0.08)
        assert count == 0

        remaining = db_dedup._conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE category = 'principle'"
        ).fetchone()[0]
        assert remaining == 3

    def test_deduplicate_threshold_boundary(self, db_dedup):
        """Entries clearly above the threshold distance should NOT be deduplicated."""
        import math
        # Use cosine distance = 0.10, well above threshold of 0.08.
        # cosine_distance = 1 - cos(theta), so cos(theta) = 0.90
        cos_theta = 0.90
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        vec_a = [0.0] * 1024
        vec_a[0] = 1.0

        vec_b = [0.0] * 1024
        vec_b[0] = cos_theta
        vec_b[1] = sin_theta

        db_dedup._embedding_queue.extend([vec_a, vec_b])

        db_dedup.insert_entry("Entry A", "Content at boundary A", "project")
        db_dedup.insert_entry("Entry B", "Content at boundary B", "project")

        # threshold=0.08 means distance < 0.08 triggers dedup.
        # These vectors have distance ~0.10, so should NOT be deduped.
        count = db_dedup.deduplicate_similar(threshold=0.08)
        assert count == 0

        remaining = db_dedup._conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE archived_at IS NULL"
        ).fetchone()[0]
        assert remaining == 2

    def test_deduplicate_keeps_most_recent(self, db_dedup):
        """Among duplicates, the entry with the latest created_at is kept."""
        vec = self._make_vec(1.0)
        db_dedup._embedding_queue.extend([vec, vec, vec])

        db_dedup.insert_entry("Oldest", "Content oldest", "chat")
        db_dedup.insert_entry("Middle", "Content middle", "chat")
        db_dedup.insert_entry("Newest", "Content newest", "chat")

        # Ensure ordering: oldest < middle < newest
        db_dedup._conn.execute(
            "UPDATE memory_entries SET created_at = datetime('now', '-2 hours') WHERE title = 'Oldest'"
        )
        db_dedup._conn.execute(
            "UPDATE memory_entries SET created_at = datetime('now', '-1 hour') WHERE title = 'Middle'"
        )
        db_dedup._conn.execute(
            "UPDATE memory_entries SET created_at = datetime('now', '+1 hour') WHERE title = 'Newest'"
        )
        db_dedup._conn.commit()

        count = db_dedup.deduplicate_similar(threshold=0.08)
        assert count == 2

        remaining = db_dedup._conn.execute(
            "SELECT title FROM memory_entries WHERE archived_at IS NULL"
        ).fetchall()
        assert len(remaining) == 1
        assert remaining[0]["title"] == "Newest"

    def test_deduplicate_returns_count(self, db_dedup):
        """Return value matches the number of deleted entries."""
        vec = self._make_vec(1.0)
        db_dedup._embedding_queue.extend([vec, vec, vec, vec])

        db_dedup.insert_entry("Dup 1", "Dup content one", "project")
        db_dedup.insert_entry("Dup 2", "Dup content two", "project")
        db_dedup.insert_entry("Dup 3", "Dup content three", "project")
        db_dedup.insert_entry("Dup 4", "Dup content four", "project")

        count = db_dedup.deduplicate_similar(threshold=0.08)
        assert count == 3  # 4 duplicates, keep 1, delete 3

    def test_deduplicate_empty_db(self, db_dedup):
        """Dedup on empty DB returns 0 with no errors."""
        count = db_dedup.deduplicate_similar(threshold=0.08)
        assert count == 0


class TestEscapeFtsQuery:
    def test_escape_fts_query(self):
        """Test the FTS query escaping function."""
        from plugins.memory.tiered.store import _escape_fts_query

        assert _escape_fts_query("hello world") == '"hello" "world"'
        assert _escape_fts_query("") == ""
        assert _escape_fts_query("   ") == ""
        assert _escape_fts_query("single") == '"single"'
        # Double quotes in tokens get escaped
        assert _escape_fts_query('say "hi"') == '"say" """hi"""'
