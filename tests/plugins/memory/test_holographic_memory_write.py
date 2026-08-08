"""Regression tests for holographic ``on_memory_write`` mirroring.

The bridge (``MemoryManager.notify_memory_tool_write``) forwards all three
mutating actions — ``add``, ``replace``, ``remove`` — to every external
memory provider. The holographic plugin's ``on_memory_write`` previously
only handled ``add``, silently dropping ``replace`` and ``remove``. These
tests pin the contract that all three actions are mirrored correctly.

Fact lookup uses a direct SQL exact-content match on the UNIQUE ``content``
column rather than FTS5 search — avoiding ranking limits and
``retrieval_count`` side effects (per teknium1's review on PR #55098).
"""

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


def _make_provider(tmp_path):
    """Create an initialised holographic provider backed by a temp DB."""
    db_path = str(tmp_path / "memory_store.db")
    provider = HolographicMemoryProvider(config={"db_path": db_path, "hrr_dim": 64})
    provider.initialize(session_id="test-session")
    return provider


# ---------------------------------------------------------------------------
# add (regression — was already working)
# ---------------------------------------------------------------------------

class TestOnMemoryWriteAdd:
    def test_add_action_creates_fact(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.on_memory_write("add", "memory", "The project uses pytest")

        row = provider._store._conn.execute(
            "SELECT content, category FROM facts WHERE content = ?",
            ("The project uses pytest",),
        ).fetchone()
        assert row is not None
        assert row["content"] == "The project uses pytest"
        assert row["category"] == "general"

    def test_add_to_user_target_uses_user_pref_category(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.on_memory_write("add", "user", "Prefers concise responses")

        row = provider._store._conn.execute(
            "SELECT category FROM facts WHERE content = ?",
            ("Prefers concise responses",),
        ).fetchone()
        assert row is not None
        assert row["category"] == "user_pref"

    def test_add_with_empty_content_is_noop(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.on_memory_write("add", "memory", "")

        facts = provider._store.list_facts(limit=100)
        assert len(facts) == 0


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------

class TestOnMemoryWriteReplace:
    def test_replace_finds_and_updates_existing_fact(self, tmp_path):
        provider = _make_provider(tmp_path)
        # Seed with an add
        provider.on_memory_write("add", "memory", "Browser uses Camofox")
        # Replace via old_text
        provider.on_memory_write(
            "replace", "memory", "Browser uses Browser Use Cloud",
            metadata={"old_text": "Browser uses Camofox"},
        )

        # Old content should be gone
        old = provider._store._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?",
            ("Browser uses Camofox",),
        ).fetchone()
        assert old is None

        # New content should exist
        new = provider._store._conn.execute(
            "SELECT content, category FROM facts WHERE content = ?",
            ("Browser uses Browser Use Cloud",),
        ).fetchone()
        assert new is not None
        assert new["content"] == "Browser uses Browser Use Cloud"

    def test_replace_falls_back_to_add_when_fact_not_found(self, tmp_path):
        """Replace something that was never added to the holographic store."""
        provider = _make_provider(tmp_path)
        provider.on_memory_write(
            "replace", "memory", "New content",
            metadata={"old_text": "Nonexistent old text"},
        )

        row = provider._store._conn.execute(
            "SELECT content, category FROM facts WHERE content = ?",
            ("New content",),
        ).fetchone()
        assert row is not None
        assert row["category"] == "general"

    def test_replace_without_metadata_falls_back_to_add(self, tmp_path):
        """When metadata is None (older bridge), replace should still not drop
        the write — it falls back to add."""
        provider = _make_provider(tmp_path)
        provider.on_memory_write("replace", "memory", "Standalone content")

        row = provider._store._conn.execute(
            "SELECT content FROM facts WHERE content = ?",
            ("Standalone content",),
        ).fetchone()
        assert row is not None

    def test_replace_falls_back_to_add_when_metadata_missing_old_text(self, tmp_path):
        """metadata dict present but old_text key absent — graceful fallback."""
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


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

class TestOnMemoryWriteRemove:
    def test_remove_finds_and_removes_existing_fact(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Temporary config note", category="general")

        provider.on_memory_write(
            "remove", "memory", "",
            metadata={"old_text": "Temporary config note"},
        )

        row = provider._store._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?",
            ("Temporary config note",),
        ).fetchone()
        assert row is None

    def test_remove_is_idempotent_when_fact_not_found(self, tmp_path):
        """Removing a fact that doesn't exist must not raise."""
        provider = _make_provider(tmp_path)
        provider.on_memory_write(
            "remove", "memory", "",
            metadata={"old_text": "Never existed"},
        )
        assert len(provider._store.list_facts(limit=100)) == 0

    def test_remove_no_op_when_no_metadata(self, tmp_path):
        """Remove without old_text can't locate a fact — silent no-op."""
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Should remain", category="general")

        provider.on_memory_write("remove", "memory", "")

        row = provider._store._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?",
            ("Should remain",),
        ).fetchone()
        assert row is not None

    def test_remove_no_op_when_metadata_missing_old_text(self, tmp_path):
        """metadata dict present but old_text key absent — no-op."""
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Should remain", category="general")

        provider.on_memory_write("remove", "memory", "", metadata={})

        row = provider._store._conn.execute(
            "SELECT fact_id FROM facts WHERE content = ?",
            ("Should remain",),
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_on_memory_write_without_store_is_safe(self, tmp_path):
        """A provider that was never initialised must not raise on writes."""
        bare = HolographicMemoryProvider(config={"db_path": str(tmp_path / "x.db")})
        bare.on_memory_write("add", "memory", "content")
        bare.on_memory_write("replace", "memory", "content", metadata={"old_text": "old"})
        bare.on_memory_write("remove", "memory", "", metadata={"old_text": "old"})

    def test_unknown_action_is_silent_noop(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.on_memory_write("search", "memory", "content")
        assert len(provider._store.list_facts(limit=100)) == 0

    def test_none_metadata_handled_gracefully_for_replace(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.on_memory_write("replace", "user", "New content", metadata=None)

        assert provider._find_fact_id_by_content("New content") is not None

    def test_none_metadata_handled_gracefully_for_remove(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.on_memory_write("remove", "user", "", metadata=None)
        # Should not raise, no-op


# ---------------------------------------------------------------------------
# _find_fact_id_by_content (the direct-SQL lookup helper)
# ---------------------------------------------------------------------------

class TestFindFactIdByContent:
    def test_exact_match_returns_id(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Exact content here", category="general")

        fact_id = provider._find_fact_id_by_content("Exact content here")
        assert fact_id is not None
        assert isinstance(fact_id, int)

    def test_no_match_returns_none(self, tmp_path):
        provider = _make_provider(tmp_path)
        assert provider._find_fact_id_by_content("Nonexistent") is None

    def test_empty_text_returns_none(self, tmp_path):
        provider = _make_provider(tmp_path)
        assert provider._find_fact_id_by_content("") is None

    def test_whitespace_stripped_before_match(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Stripped content", category="general")

        fact_id = provider._find_fact_id_by_content("  Stripped content  ")
        assert fact_id is not None


# ---------------------------------------------------------------------------
# round-trip: add → replace → remove (the full lifecycle)
# ---------------------------------------------------------------------------

def test_full_lifecycle_add_replace_remove(tmp_path):
    """Simulate the real usage pattern: add a fact, replace it, then remove it.
    At each step the holographic store should be in sync with built-in memory."""
    provider = _make_provider(tmp_path)

    # 1. Add
    provider.on_memory_write("add", "user", "User likes vim")
    assert provider._find_fact_id_by_content("User likes vim") is not None

    # 2. Replace
    provider.on_memory_write(
        "replace", "user", "User likes neovim",
        metadata={"old_text": "User likes vim"},
    )
    assert provider._find_fact_id_by_content("User likes vim") is None
    assert provider._find_fact_id_by_content("User likes neovim") is not None

    # 3. Remove
    provider.on_memory_write(
        "remove", "user", "",
        metadata={"old_text": "User likes neovim"},
    )
    assert provider._find_fact_id_by_content("User likes neovim") is None
    assert len(provider._store.list_facts(limit=100)) == 0


# ---------------------------------------------------------------------------
# retrieval_count non-corruption (the FTS5-avoidance rationale)
# ---------------------------------------------------------------------------

class TestRetrievalCountNotCorrupted:
    """The maintenance lookup path must NOT increment retrieval_count.

    The original fix used search_facts() (which bumps retrieval_count for all
    candidates). The direct SQL approach in _find_fact_id_by_content avoids
    this. These tests pin that invariant.
    """

    def test_add_does_not_bump_retrieval_count(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Test fact A", category="general")

        # Record initial retrieval_count
        row = provider._store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content = ?",
            ("Test fact A",),
        ).fetchone()
        initial = row["retrieval_count"]

        # Add a different fact — should not touch A's count
        provider.on_memory_write("add", "memory", "Test fact B")

        row = provider._store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content = ?",
            ("Test fact A",),
        ).fetchone()
        assert row["retrieval_count"] == initial

    def test_replace_does_not_bump_retrieval_count(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Old content", category="general")

        # Record initial retrieval_count
        row = provider._store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content = ?",
            ("Old content",),
        ).fetchone()
        initial = row["retrieval_count"]

        # Replace it
        provider.on_memory_write(
            "replace", "memory", "New content",
            metadata={"old_text": "Old content"},
        )

        # The old fact is gone (replaced), but the NEW fact's retrieval_count
        # must be 0 — the maintenance lookup should never bump it.
        new_row = provider._store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content = ?",
            ("New content",),
        ).fetchone()
        assert new_row is not None
        assert new_row["retrieval_count"] == 0

    def test_remove_does_not_bump_retrieval_count(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider._store.add_fact("Fact to remove", category="general")

        # Record initial retrieval_count
        row = provider._store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content = ?",
            ("Fact to remove",),
        ).fetchone()
        assert row["retrieval_count"] == 0

        # Remove it — the lookup to find it must not bump the count
        provider.on_memory_write(
            "remove", "memory", "",
            metadata={"old_text": "Fact to remove"},
        )

        # Fact is gone — verify it was removed (and the count was never bumped)
        row = provider._store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content = ?",
            ("Fact to remove",),
        ).fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# UNIQUE-constraint collision on replace
# ---------------------------------------------------------------------------

class TestReplaceUniqueCollision:
    """When replace's new content already exists as a different fact,
    update_fact raises IntegrityError on the UNIQUE content column.

    The replace intent is "old becomes new" — if new already exists, the old
    fact is redundant and should be removed, not left stale.
    """

    def test_replace_onto_existing_content_removes_old_fact(self, tmp_path):
        provider = _make_provider(tmp_path)

        # Seed two facts
        provider._store.add_fact("Original fact", category="general")
        provider._store.add_fact("Already exists", category="general")

        # Replace "Original fact" → "Already exists" (content collision)
        provider.on_memory_write(
            "replace", "memory", "Already exists",
            metadata={"old_text": "Original fact"},
        )

        # The old fact should be removed (replace intent satisfied)
        assert provider._find_fact_id_by_content("Original fact") is None
        # The existing fact should still be there (not duplicated)
        assert provider._find_fact_id_by_content("Already exists") is not None

    def test_replace_onto_existing_content_leaves_one_fact(self, tmp_path):
        provider = _make_provider(tmp_path)

        provider._store.add_fact("Fact A", category="general")
        provider._store.add_fact("Fact B", category="general")

        provider.on_memory_write(
            "replace", "memory", "Fact B",
            metadata={"old_text": "Fact A"},
        )

        # Should have exactly one fact ("Fact B"), not two
        facts = provider._store.list_facts(limit=100)
        assert len(facts) == 1
        assert facts[0]["content"] == "Fact B"
