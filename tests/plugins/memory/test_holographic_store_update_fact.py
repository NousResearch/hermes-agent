"""MemoryStore.update_fact must not crash when content collides with another row.

Regression for #43389: ``facts.content`` is ``UNIQUE``. ``add_fact`` already
catches the ``sqlite3.IntegrityError`` and returns the existing id, but
``update_fact`` ran the ``UPDATE`` unguarded, so updating one fact's content to
a value another fact already holds raised an unhandled IntegrityError and
crashed the memory operation (e.g. an LLM-generated correction that happens to
match existing content). The update is now rejected (returns False) instead.
"""

import sqlite3

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    """Each test starts and ends with an empty shared-connection registry."""
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


@pytest.fixture
def store(tmp_path):
    """A MemoryStore backed by a per-test database file.

    ``MemoryStore`` resolves ``db_path`` before ``sqlite3.connect`` and keys its
    process-wide shared-connection registry on the resolved path, so passing
    ``":memory:"`` yields a shared *on-disk* file rather than SQLite's in-memory
    sentinel — leaking rows between tests. Use ``tmp_path`` and close the store,
    matching ``test_holographic_store.py``.
    """
    store = MemoryStore(tmp_path / "memory_store.db")
    try:
        yield store
    finally:
        store.close()


def test_update_to_duplicate_content_returns_false_not_crash(store):
    store.add_fact("Python is fast", category="lang")
    id_b = store.add_fact("Rust is safe", category="lang")

    # Previously raised sqlite3.IntegrityError: UNIQUE constraint failed.
    assert store.update_fact(id_b, content="Python is fast") is False

    # The rejected update must leave the original content untouched.
    row = store._conn.execute(
        "SELECT content FROM facts WHERE fact_id = ?", (id_b,)
    ).fetchone()
    assert row["content"] == "Rust is safe"


def test_non_conflicting_content_update_still_succeeds(store):
    id_b = store.add_fact("Rust is safe", category="lang")
    assert store.update_fact(id_b, content="Rust is memory-safe") is True
    row = store._conn.execute(
        "SELECT content FROM facts WHERE fact_id = ?", (id_b,)
    ).fetchone()
    assert row["content"] == "Rust is memory-safe"


def test_update_unknown_fact_returns_false(store):
    assert store.update_fact(99999, content="anything") is False


def test_update_fact_to_its_own_content_is_not_a_conflict(store):
    # A row keeping its own UNIQUE value must not be treated as a duplicate.
    id_a = store.add_fact("Python is fast", category="lang")
    assert store.update_fact(id_a, content="Python is fast") is True


def test_non_content_update_unaffected_by_guard(store):
    # trust/tags/category updates don't touch the UNIQUE column.
    id_a = store.add_fact("Python is fast", category="lang")
    assert store.update_fact(id_a, trust_delta=0.1, tags="pl") is True
