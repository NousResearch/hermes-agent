"""Regression tests for holographic ``on_memory_write`` mirroring.

The bridge (``MemoryManager.notify_memory_tool_write``) forwards all three
mutating actions — ``add``, ``replace``, ``remove`` — to every external
memory provider.  The holographic plugin's ``on_memory_write`` previously
only handled ``add``, silently dropping ``replace`` and ``remove``.  These
tests pin the contract that all three actions are mirrored correctly.

Discovery context: the bug was found when a ``memory(operations=[replace,
replace, replace])`` call updated built-in memory but left the holographic
fact store with stale content — the replaces were forwarded by the bridge
but silently dropped by the plugin.
"""

import json

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

def test_add_mirrors_as_new_fact(tmp_path):
    provider = _make_provider(tmp_path)
    provider.on_memory_write("add", "memory", "The project uses pytest")

    results = provider._store.search_facts("pytest", min_trust=0.0, limit=10)
    assert len(results) == 1
    assert results[0]["content"] == "The project uses pytest"
    assert results[0]["category"] == "general"


def test_add_to_user_target_uses_user_pref_category(tmp_path):
    provider = _make_provider(tmp_path)
    provider.on_memory_write("add", "user", "Prefers concise responses")

    results = provider._store.search_facts("concise", min_trust=0.0, limit=10)
    assert len(results) == 1
    assert results[0]["category"] == "user_pref"


def test_add_with_empty_content_is_noop(tmp_path):
    provider = _make_provider(tmp_path)
    provider.on_memory_write("add", "memory", "")

    facts = provider._store.list_facts(limit=100)
    assert len(facts) == 0


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------

def test_replace_updates_existing_fact(tmp_path):
    provider = _make_provider(tmp_path)
    # Seed with an add
    provider.on_memory_write("add", "memory", "Browser uses Camofox")
    # Replace via old_text
    provider.on_memory_write(
        "replace", "memory", "Browser uses Browser Use Cloud",
        metadata={"old_text": "Browser uses Camofox"},
    )

    results = provider._store.search_facts("Browser", min_trust=0.0, limit=10)
    assert len(results) == 1
    assert results[0]["content"] == "Browser uses Browser Use Cloud"
    # The old content must be gone
    old_results = provider._store.search_facts("Camofox", min_trust=0.0, limit=10)
    assert all(r["content"] != "Browser uses Camofox" for r in old_results)


def test_replace_falls_back_to_add_when_fact_not_found(tmp_path):
    provider = _make_provider(tmp_path)
    # Replace something that was never added to the holographic store
    provider.on_memory_write(
        "replace", "memory", "New content",
        metadata={"old_text": "Nonexistent old text"},
    )

    results = provider._store.search_facts("New content", min_trust=0.0, limit=10)
    assert len(results) == 1
    assert results[0]["content"] == "New content"


def test_replace_without_metadata_falls_back_to_add(tmp_path):
    """When metadata is None (older bridge), replace should still not drop
    the write — it falls back to add."""
    provider = _make_provider(tmp_path)
    provider.on_memory_write("replace", "memory", "Standalone content")

    results = provider._store.search_facts("Standalone", min_trust=0.0, limit=10)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

def test_remove_deletes_existing_fact(tmp_path):
    provider = _make_provider(tmp_path)
    provider.on_memory_write("add", "memory", "Temporary config note")
    # Verify it's there
    assert len(provider._store.search_facts("Temporary", min_trust=0.0, limit=10)) == 1
    # Remove it
    provider.on_memory_write(
        "remove", "memory", "",
        metadata={"old_text": "Temporary config note"},
    )

    results = provider._store.search_facts("Temporary", min_trust=0.0, limit=10)
    assert len(results) == 0


def test_remove_is_idempotent_when_fact_not_found(tmp_path):
    """Removing a fact that doesn't exist in the holographic store must not
    raise — it's a silent no-op."""
    provider = _make_provider(tmp_path)
    provider.on_memory_write(
        "remove", "memory", "",
        metadata={"old_text": "Never existed"},
    )
    # Should not raise, and store should still be empty
    assert len(provider._store.list_facts(limit=100)) == 0


def test_remove_without_metadata_is_noop(tmp_path):
    """Remove without old_text can't locate a fact — silent no-op."""
    provider = _make_provider(tmp_path)
    provider.on_memory_write("add", "memory", "Some fact")
    provider.on_memory_write("remove", "memory", "")
    # Fact should still be there — we had no old_text to find it
    assert len(provider._store.search_facts("Some fact", min_trust=0.0, limit=10)) == 1


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_on_memory_write_without_store_is_safe(tmp_path):
    """A provider that was never initialised must not raise on writes."""
    bare = HolographicMemoryProvider(config={"db_path": str(tmp_path / "x.db")})
    bare.on_memory_write("add", "memory", "content")
    bare.on_memory_write("replace", "memory", "content", metadata={"old_text": "old"})
    bare.on_memory_write("remove", "memory", "", metadata={"old_text": "old"})


def test_unknown_action_is_silent_noop(tmp_path):
    provider = _make_provider(tmp_path)
    provider.on_memory_write("search", "memory", "content")
    assert len(provider._store.list_facts(limit=100)) == 0


# ---------------------------------------------------------------------------
# round-trip: add → replace → remove (the full lifecycle)
# ---------------------------------------------------------------------------

def test_full_lifecycle_add_replace_remove(tmp_path):
    """Simulate the real usage pattern: add a fact, replace it, then remove it.
    At each step the holographic store should be in sync with built-in memory."""
    provider = _make_provider(tmp_path)

    # 1. Add
    provider.on_memory_write("add", "user", "User likes vim")
    results = provider._store.search_facts("vim", min_trust=0.0, limit=10)
    assert len(results) == 1
    assert results[0]["content"] == "User likes vim"

    # 2. Replace
    provider.on_memory_write(
        "replace", "user", "User likes neovim",
        metadata={"old_text": "User likes vim"},
    )
    results = provider._store.search_facts("neovim", min_trust=0.0, limit=10)
    assert len(results) == 1
    assert results[0]["content"] == "User likes neovim"
    old = provider._store.search_facts("vim", min_trust=0.0, limit=10)
    assert all("neovim" in r["content"] for r in old)  # no stale "likes vim"

    # 3. Remove
    provider.on_memory_write(
        "remove", "user", "",
        metadata={"old_text": "User likes neovim"},
    )
    assert len(provider._store.search_facts("neovim", min_trust=0.0, limit=10)) == 0
    assert len(provider._store.list_facts(limit=100)) == 0
