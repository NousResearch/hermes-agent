"""Tests for HolographicMemoryProvider.on_memory_write — #55095.

Covers add, replace, and remove actions including metadata handling and
graceful fallback behavior.
"""

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


def _make_provider(tmp_path):
    db_path = str(tmp_path / "memory_store.db")
    provider = HolographicMemoryProvider(config={"db_path": db_path, "hrr_dim": 64})
    provider.initialize(session_id="test-session")
    return provider


class TestOnMemoryWriteAdd:
    def test_add_action_creates_fact(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.on_memory_write("add", "user", "User prefers dark mode")

        row = provider._store._conn.execute(
            "SELECT content, category FROM facts WHERE content = ?",
            ("User prefers dark mode",),
        ).fetchone()
        assert row is not None
        assert row["content"] == "User prefers dark mode"
        assert row["category"] == "user_pref"

    def test_add_action_general_target(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.on_memory_write("add", "project", "Uses Python 3.12")

        row = provider._store._conn.execute(
            "SELECT category FROM facts WHERE content = ?",
            ("Uses Python 3.12",),
        ).fetchone()
        assert row is not None
        assert row["category"] == "general"


class TestOnMemoryWriteReplace:
    def test_replace_finds_and_updates_existing_fact(self, tmp_path):
        provider = _make_provider(tmp_path)
        # Seed the original fact
        provider._store.add_fact("User prefers vim", category="user_pref")

        provider.on_memory_write(
            "replace", "user", "User prefers neovim",
            metadata={"old_text": "User prefers vim"},
        )

        # Old content should be gone
        old = provider._store._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?",
            ("User prefers vim",),
        ).fetchone()
        assert old is None

        # New content should exist
        new = provider._store._conn.execute(
            "SELECT content, category FROM facts WHERE content = ?",
            ("User prefers neovim",),
        ).fetchone()
        assert new is not None
        assert new["content"] == "User prefers neovim"

    def test_replace_falls_back_to_add_when_old_fact_not_found(self, tmp_path):
        provider = _make_provider(tmp_path)

        provider.on_memory_write(
            "replace", "user", "User prefers emacs",
            metadata={"old_text": "nonexistent old content"},
        )

        row = provider._store._conn.execute(
            "SELECT content, category FROM facts WHERE content = ?",
            ("User prefers emacs",),
        ).fetchone()
        assert row is not None
        assert row["category"] == "user_pref"

    def test_replace_falls_back_to_add_when_no_metadata(self, tmp_path):
        provider = _make_provider(tmp_path)

        provider.on_memory_write("replace", "user", "User prefers emacs")

        row = provider._store._conn.execute(
            "SELECT content FROM facts WHERE content = ?",
            ("User prefers emacs",),
        ).fetchone()
        assert row is not None

    def test_replace_falls_back_to_add_when_metadata_missing_old_text(self, tmp_path):
        provider = _make_provider(tmp_path)

        provider.on_memory_write(
            "replace", "user", "User prefers emacs",
            metadata={"other_key": "value"},
        )

        row = provider._store._conn.execute(
            "SELECT content FROM facts WHERE content = ?",
            ("User prefers emacs",),
        ).fetchone()
        assert row is not None


class TestOnMemoryWriteRemove:
    def test_remove_finds_and_removes_existing_fact(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Outdated preference", category="user_pref")

        provider.on_memory_write(
            "remove", "user", "",
            metadata={"old_text": "Outdated preference"},
        )

        row = provider._store._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?",
            ("Outdated preference",),
        ).fetchone()
        assert row is None

    def test_remove_is_idempotent_when_fact_not_found(self, tmp_path):
        provider = _make_provider(tmp_path)

        # Should not raise
        provider.on_memory_write(
            "remove", "user", "",
            metadata={"old_text": "never existed"},
        )

    def test_remove_no_op_when_no_metadata(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Should remain", category="general")

        # No metadata means no old_text, so nothing to look up
        provider.on_memory_write("remove", "user", "")

        row = provider._store._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?",
            ("Should remain",),
        ).fetchone()
        assert row is not None

    def test_remove_no_op_when_metadata_missing_old_text(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Should remain", category="general")

        provider.on_memory_write("remove", "user", "", metadata={})

        row = provider._store._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?",
            ("Should remain",),
        ).fetchone()
        assert row is not None


class TestMetadataHandling:
    def test_metadata_old_text_is_used_for_replace(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Original fact", category="general")

        provider.on_memory_write(
            "replace", "project", "Updated fact",
            metadata={"old_text": "Original fact"},
        )

        assert provider._find_fact_id_by_content("Original fact") is None
        assert provider._find_fact_id_by_content("Updated fact") is not None

    def test_metadata_old_text_is_used_for_remove(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Fact to remove", category="general")

        provider.on_memory_write(
            "remove", "project", "",
            metadata={"old_text": "Fact to remove"},
        )

        assert provider._find_fact_id_by_content("Fact to remove") is None

    def test_none_metadata_handled_gracefully_for_replace(self, tmp_path):
        provider = _make_provider(tmp_path)
        # Should not raise, falls back to add
        provider.on_memory_write("replace", "user", "New content", metadata=None)

        assert provider._find_fact_id_by_content("New content") is not None

    def test_none_metadata_handled_gracefully_for_remove(self, tmp_path):
        provider = _make_provider(tmp_path)
        # Should not raise, no-op
        provider.on_memory_write("remove", "user", "", metadata=None)
