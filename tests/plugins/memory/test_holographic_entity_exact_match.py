"""Entity resolution in the holographic store must match names *exactly*.

``_resolve_entity`` documents a case-insensitive **name or alias** match. The
name lookup used ``LIKE ?``, so ``_`` and ``%`` inside the bound name acted as
SQL wildcards: resolving ``test_entity`` also matched an unrelated
``testXentity`` row, and ``100%`` matched ``1000``. The lookup is now
``name = ? COLLATE NOCASE`` — exact, with the case-insensitive contract carried
by the collation instead of by LIKE.

Behaviour contract pinned here: a name resolves only to its own row (or is
created), while case differences keep resolving case-insensitively, and the
alias fallback still applies after the exact lookup.
"""

from plugins.memory.holographic.store import MemoryStore


def _store_with(tmp_path, names):
    store = MemoryStore(tmp_path / "memory_store.db")
    for name in names:
        store._write("INSERT INTO entities (name) VALUES (?)", (name,))
    return store


def _entity_id(store, name):
    row = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name = ?", (name,)
    ).fetchone()
    return None if row is None else int(row["entity_id"])


def test_wildcard_lookalike_does_not_satisfy_exact_lookup(tmp_path):
    """'test_entity' must not resolve to the lookalike 'testXentity'."""
    store = _store_with(tmp_path, ["testXentity", "test_entity_v2"])
    lookalike_id = _entity_id(store, "testXentity")

    resolved = store._resolve_entity("test_entity")

    assert resolved != lookalike_id
    assert _entity_id(store, "test_entity") == resolved


def test_exact_lookup_is_case_insensitive_and_not_a_prefix_match(tmp_path):
    """Case differences resolve; longer names do not."""
    store = _store_with(tmp_path, ["Test_Entity", "test_entity_v2"])
    stored_id = _entity_id(store, "Test_Entity")

    assert store._resolve_entity("test_entity") == stored_id
    assert store._resolve_entity("test_entity_v2") == _entity_id(store, "test_entity_v2")
