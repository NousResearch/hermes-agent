"""Regression tests for holographic MemoryStore.update_fact content collisions.

facts.content carries a UNIQUE constraint. add_fact() dedupes that collision
on insert, but update_fact() used to issue the content UPDATE without guarding
it, so editing one fact's content to text another fact already held raised an
opaque ``UNIQUE constraint failed: facts.content`` and silently dropped the
whole update (tags/trust/category included). These tests pin the corrected
behaviour: a clear ValueError, no partial mutation, and unaffected non-colliding
paths.
"""

import os
import tempfile

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture()
def store():
    tmpdir = tempfile.mkdtemp()
    s = MemoryStore(db_path=os.path.join(tmpdir, "facts.db"))
    try:
        yield s
    finally:
        s.close()


def _content(store: MemoryStore, fact_id: int) -> str:
    row = store._conn.execute(
        "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    return row["content"]


def test_update_to_duplicate_content_raises_clear_error(store):
    a = store.add_fact("Alice likes coffee")
    b = store.add_fact("Bob likes tea")

    with pytest.raises(ValueError, match="already has this exact content"):
        store.update_fact(a, content="Bob likes tea")

    # The collision must not be reported as a raw sqlite message.
    assert _content(store, a) == "Alice likes coffee"
    assert _content(store, b) == "Bob likes tea"


def test_failed_update_leaves_batched_fields_untouched(store):
    a = store.add_fact("Alice likes coffee", tags="orig")
    store.add_fact("Bob likes tea")

    with pytest.raises(ValueError):
        # tags + content batched into one UPDATE; both must roll back together.
        store.update_fact(a, content="Bob likes tea", tags="changed")

    row = store._conn.execute(
        "SELECT content, tags FROM facts WHERE fact_id = ?", (a,)
    ).fetchone()
    assert row["content"] == "Alice likes coffee"
    assert row["tags"] == "orig"


def test_non_colliding_content_update_still_succeeds(store):
    a = store.add_fact("Alice likes coffee")
    store.add_fact("Bob likes tea")

    assert store.update_fact(a, content="Alice loves espresso") is True
    assert _content(store, a) == "Alice loves espresso"


def test_metadata_only_update_unaffected_by_guard(store):
    a = store.add_fact("Alice likes coffee", tags="orig")

    assert store.update_fact(a, tags="updated", trust_delta=0.1) is True
    row = store._conn.execute(
        "SELECT tags FROM facts WHERE fact_id = ?", (a,)
    ).fetchone()
    assert row["tags"] == "updated"
