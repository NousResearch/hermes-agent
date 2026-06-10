"""Tests for FactStore.update_fact UNIQUE constraint handling.

Regression tests for issue #43389: update_fact crashes with unhandled
sqlite3.IntegrityError when content is updated to a value that already
exists in another row.
"""

import pytest

from plugins.memory.holographic.store import MemoryStore


class TestUpdateFactUniqueConstraint:
    """update_fact must handle UNIQUE(content) conflicts gracefully."""

    def _make_store(self):
        return MemoryStore(":memory:", hrr_dim=16)

    def test_update_fact_content_to_existing_returns_false(self):
        """Updating content to match another row's content returns False."""
        store = self._make_store()
        id_a = store.add_fact("Python is fast", category="lang")
        id_b = store.add_fact("Rust is safe", category="lang")

        result = store.update_fact(id_b, content="Python is fast")
        assert result is False

    def test_update_fact_content_to_existing_preserves_original(self):
        """The original row with the duplicate content is unchanged."""
        store = self._make_store()
        id_a = store.add_fact("Python is fast", category="lang")
        id_b = store.add_fact("Rust is safe", category="lang")

        store.update_fact(id_b, content="Python is fast")

        rows = list(store._conn.execute(
            "SELECT fact_id, content FROM facts ORDER BY fact_id"
        ))
        assert len(rows) == 2
        assert rows[0]["content"] == "Python is fast"
        assert rows[1]["content"] == "Rust is safe"

    def test_update_fact_content_to_unique_succeeds(self):
        """Updating content to a unique value succeeds and returns True."""
        store = self._make_store()
        id_a = store.add_fact("Python is fast", category="lang")

        result = store.update_fact(id_a, content="Python is very fast")
        assert result is True

        row = store._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (id_a,)
        ).fetchone()
        assert row["content"] == "Python is very fast"

    def test_update_fact_nonexistent_returns_false(self):
        """Updating a non-existent fact_id returns False."""
        store = self._make_store()
        result = store.update_fact(999, content="whatever")
        assert result is False

    def test_update_fact_trust_only_no_conflict(self):
        """Updating only trust_score (no content change) never hits UNIQUE."""
        store = self._make_store()
        id_a = store.add_fact("Python is fast", category="lang")
        id_b = store.add_fact("Rust is safe", category="lang")

        result = store.update_fact(id_b, trust_delta=0.3)
        assert result is True

    def test_update_fact_tags_only_no_conflict(self):
        """Updating only tags (no content change) never hits UNIQUE."""
        store = self._make_store()
        id_a = store.add_fact("Python is fast", category="lang")
        id_b = store.add_fact("Rust is safe", category="lang")

        result = store.update_fact(id_b, tags="important")
        assert result is True

    def test_update_fact_category_only_no_conflict(self):
        """Updating only category (no content change) never hits UNIQUE."""
        store = self._make_store()
        id_a = store.add_fact("Python is fast", category="lang")

        result = store.update_fact(id_a, category="programming")
        assert result is True

    def test_update_fact_content_to_own_content_succeeds(self):
        """Updating content to its own current value should not conflict."""
        store = self._make_store()
        id_a = store.add_fact("Python is fast", category="lang")

        # SQLite allows UPDATE SET content='Python is fast' WHERE fact_id=id_a
        # even with UNIQUE, because the row itself satisfies the constraint.
        result = store.update_fact(id_a, content="Python is fast")
        assert result is True
