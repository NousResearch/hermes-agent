"""The supervisor's owner enumeration seeks the exact-case owner key range."""

from contextlib import contextmanager

from persistence.sqlite.gateway.hosted_rooms import OWNER_PREFIX
from tests.fixtures.storage_holder import raw_holder


def test_owned_rooms_seeks_key_index(backend, scopes, monkeypatch):
    store = backend.open(scopes["a"])
    store.state_meta.set(OWNER_PREFIX + "room", "alice")
    holder = raw_holder(store)
    read_ctx = holder._read_ctx
    captured = []

    @contextmanager
    def traced_read(**kwargs):
        with read_ctx(**kwargs) as conn:
            conn.set_trace_callback(captured.append)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)

    monkeypatch.setattr(holder, "_read_ctx", traced_read)
    assert store.hosted_rooms.owned_rooms() == frozenset({"room"})
    sql, = [sql for sql in captured if sql.startswith("SELECT key FROM state_meta")]
    with read_ctx() as conn:
        plan = "\n".join(row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql))
    assert "SEARCH state_meta USING" in plan
    assert "SCAN state_meta" not in plan


def test_owned_rooms_ignores_owner_keys_that_exact_owner_reads_cannot_find(backend, scopes):
    store = backend.open(scopes["a"])
    store.state_meta.set(OWNER_PREFIX + "room", "alice")
    store.state_meta.set(OWNER_PREFIX.upper() + "x", "bob")
    assert store.hosted_rooms.owner("room") == "alice"
    assert store.hosted_rooms.owner("x") is None
    assert store.hosted_rooms.owned_rooms() == frozenset({"room"})
