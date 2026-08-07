"""Tests for null-guard fix in plugins.memory.holographic.store.MemoryStore.

Covers two concurrent-delete edge cases in add_fact() and update_fact() where
fetchone() returns None:
  - add_fact: IntegrityError → concurrent delete → RuntimeError, not TypeError
  - update_fact: category re-fetch after concurrent delete → "general" fallback
"""

import sqlite3
import tempfile

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture
def store():
    """Create a MemoryStore backed by a temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    store = MemoryStore(db_path=path)
    yield store
    store._conn.close()


# ---------------------------------------------------------------------------
# Connection wrapper — lets us intercept .execute() calls despite
# sqlite3.Connection being an immutable C extension type.
# ---------------------------------------------------------------------------


class _ExecuteWrapper:
    """Proxy that wraps a sqlite3.Connection and intercepts .execute()."""

    def __init__(self, real_conn):
        object.__setattr__(self, "_conn", real_conn)
        object.__setattr__(self, "_hooks", {})

    def execute(self, sql, parameters=()):
        hook = self._hooks.get("execute")
        if hook is not None:
            return hook(self._conn, sql, parameters)
        return self._conn.execute(sql, parameters)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        if name in ("_conn", "_hooks"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._conn, name, value)


# ---------------------------------------------------------------------------
# add_fact tests
# ---------------------------------------------------------------------------


def test_add_fact_duplicate_returns_existing_id(store):
    """Normal deduplication path: second add_fact with same content returns
    the original fact_id without raising."""
    fact_id_1 = store.add_fact("unique fact")
    fact_id_2 = store.add_fact("unique fact")
    assert fact_id_1 == fact_id_2
    assert isinstance(fact_id_2, int)


def test_add_fact_concurrent_delete_raises_runtime_error(store):
    """When a concurrent process deletes the fact between the failed INSERT
    and the follow-up SELECT, add_fact must raise RuntimeError with a
    descriptive message — not a bare TypeError from subscripting None."""
    store.add_fact("will be raced")

    wrapper = _ExecuteWrapper(store._conn)
    call_count = [0]

    def interceptor(real_conn, sql, parameters):
        call_count[0] += 1
        if call_count[0] == 1:
            raise sqlite3.IntegrityError("UNIQUE constraint failed: facts.content")
        # Simulate concurrent delete: delete the row from a second
        # connection so that the real SELECT returns no rows.
        conn2 = sqlite3.connect(str(store.db_path))
        conn2.execute("DELETE FROM facts WHERE content = ?", ("will be raced",))
        conn2.commit()
        conn2.close()
        return real_conn.execute(sql, parameters)

    wrapper._hooks["execute"] = interceptor
    store._conn = wrapper

    with pytest.raises(RuntimeError, match="Concurrent delete"):
        store.add_fact("will be raced")


# ---------------------------------------------------------------------------
# update_fact tests
# ---------------------------------------------------------------------------


def test_update_fact_nonexistent_returns_false(store):
    """Calling update_fact on a non-existent fact_id must return False."""
    assert store.update_fact(99999, trust_delta=0.1) is False


def test_update_fact_with_category_skips_refetch(store):
    """When category is supplied by the caller, the re-fetch SELECT must
    be skipped entirely."""
    fact_id = store.add_fact("test content", category="work")

    wrapper = _ExecuteWrapper(store._conn)
    category_select_calls = []

    def interceptor(real_conn, sql, parameters):
        if isinstance(sql, str) and "SELECT category FROM facts" in sql:
            category_select_calls.append(sql)
        return real_conn.execute(sql, parameters)

    wrapper._hooks["execute"] = interceptor
    store._conn = wrapper

    result = store.update_fact(fact_id, category="home", trust_delta=0.1)
    assert result is True
    assert len(category_select_calls) == 0, (
        "category SELECT should be skipped when category is provided"
    )


def test_update_fact_concurrent_delete_uses_general_fallback(store):
    """When category is NOT supplied and a concurrent delete removes the
    fact between the UPDATE and the category re-fetch, the code must fall
    back to 'general' instead of crashing with a TypeError."""
    fact_id = store.add_fact("test content", category="work")

    wrapper = _ExecuteWrapper(store._conn)
    select_category_triggered = [False]
    row_deleted = [False]

    def interceptor(real_conn, sql, parameters):
        if isinstance(sql, str) and "SELECT category FROM facts" in sql:
            select_category_triggered[0] = True
            # Delete the row from a second connection to simulate race,
            # then run the real SELECT which will now return no rows.
            if not row_deleted[0]:
                conn2 = sqlite3.connect(str(store.db_path))
                conn2.execute(
                    "DELETE FROM facts WHERE fact_id = ?", (fact_id,)
                )
                conn2.commit()
                conn2.close()
                row_deleted[0] = True
        return real_conn.execute(sql, parameters)

    wrapper._hooks["execute"] = interceptor
    store._conn = wrapper

    # Must not crash — should fall back to "general"
    result = store.update_fact(fact_id, trust_delta=0.1)
    assert result is True
    assert select_category_triggered[0], "category re-fetch was never triggered"
