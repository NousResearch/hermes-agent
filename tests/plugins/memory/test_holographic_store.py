"""Tests for _resolve_entity LIKE-wildcard and case-sensitivity fixes (issue #43394)."""

from __future__ import annotations

import pytest

from plugins.memory.holographic.store import MemoryStore as FactStore


class TestResolveEntityNameMatch:
    """_resolve_entity name lookup must be exact (no LIKE wildcards)."""

    def test_underscore_not_wildcard(self):
        """Underscore in entity name must NOT match arbitrary single chars."""
        store = FactStore(":memory:")
        try:
            store.add_fact("test_entity is a framework", category="tech")
            # "testXentity" should NOT resolve to "test_entity"
            id_different = store._resolve_entity("testXentity")
            id_original = store._resolve_entity("test_entity")
            assert id_different != id_original
        finally:
            store.close()

    def test_percent_not_wildcard(self):
        """Percent in entity name must NOT match arbitrary substrings."""
        store = FactStore(":memory:")
        try:
            store.add_fact("100% complete", category="general")
            # "100X complete" should NOT resolve to "100% complete"
            id_different = store._resolve_entity("100X complete")
            id_original = store._resolve_entity("100% complete")
            assert id_different != id_original
        finally:
            store.close()

    def test_case_insensitive_match(self):
        """Entity lookup must be case-insensitive (docstring contract)."""
        store = FactStore(":memory:")
        try:
            store.add_fact("Apple released iOS 19", category="tech")
            # Different casing should resolve to the SAME entity
            id_upper = store._resolve_entity("APPLE")
            id_original = store._resolve_entity("Apple")
            assert id_upper == id_original
        finally:
            store.close()

    def test_exact_match_no_false_positive(self):
        """Similar but distinct names must resolve to different entities."""
        store = FactStore(":memory:")
        try:
            store.add_fact("OpenAI is an AI company", category="tech")
            store.add_fact("OpenAI_API is a library", category="tech")
            id1 = store._resolve_entity("OpenAI")
            id2 = store._resolve_entity("OpenAI_API")
            assert id1 != id2
        finally:
            store.close()


class TestResolveEntityAliasMatch:
    """Alias lookup must escape LIKE wildcards in search term."""

    def test_alias_underscore_not_wildcard(self):
        """Underscore in alias search must NOT match arbitrary chars."""
        store = FactStore(":memory:")
        try:
            # Manually insert entity with alias containing underscore
            store._conn.execute(
                "INSERT INTO entities (name, aliases) VALUES (?, ?)",
                ("Test Entity", "test_alias,other"),
            )
            store._conn.commit()
            # "test_alias" should match, "testXalias" should NOT
            id_match = store._resolve_entity("test_alias")
            assert id_match is not None
            # Now search for a name that doesn't exist as name or alias
            id_no_match = store._resolve_entity("testXalias")
            # Should be a NEW entity (different ID)
            assert id_no_match != id_match
        finally:
            store.close()

    def test_alias_percent_not_wildcard(self):
        """Percent in alias search must NOT match arbitrary substrings."""
        store = FactStore(":memory:")
        try:
            store._conn.execute(
                "INSERT INTO entities (name, aliases) VALUES (?, ?)",
                ("Revenue Corp", "100%"),
            )
            store._conn.commit()
            # "100%" should match via alias
            id_match = store._resolve_entity("100%")
            assert id_match is not None
            # "100X" should NOT match via alias
            id_no_match = store._resolve_entity("100X")
            assert id_no_match != id_match
        finally:
            store.close()

    def test_alias_exact_match_still_works(self):
        """Normal alias matching (no wildcards) must still work."""
        store = FactStore(":memory:")
        try:
            store._conn.execute(
                "INSERT INTO entities (name, aliases) VALUES (?, ?)",
                ("Robert", "Bob,Bobby"),
            )
            store._conn.commit()
            id_bob = store._resolve_entity("Bob")
            id_robert = store._resolve_entity("Robert")
            assert id_bob == id_robert
        finally:
            store.close()


class TestResolveEntityRegression:
    """Ensure the fix doesn't break normal entity resolution."""

    def test_new_entity_created_when_no_match(self):
        """A truly new name must create a new entity."""
        store = FactStore(":memory:")
        try:
            id1 = store._resolve_entity("UniqueEntity")
            id2 = store._resolve_entity("UniqueEntity")
            assert id1 == id2  # same entity on repeated lookup
            # Different name → different entity
            id3 = store._resolve_entity("DifferentEntity")
            assert id3 != id1
        finally:
            store.close()

    def test_add_fact_links_to_existing_entity(self):
        """add_fact must reuse existing entities (no duplicates)."""
        store = FactStore(":memory:")
        try:
            # Use a multi-word capitalized phrase (the regex extracts those)
            store.add_fact("John Doe released a new framework", category="tech")
            store.add_fact("John Doe improved the framework", category="tech")
            # Should have exactly 1 entity for "John Doe"
            rows = store._conn.execute(
                "SELECT COUNT(*) FROM entities WHERE LOWER(name) = LOWER(?)",
                ("John Doe",),
            ).fetchone()
            assert rows[0] == 1
        finally:
            store.close()
