"""MemoryStore._resolve_entity must match entity names exactly (case-insensitively).

Regression for #43394: the lookup used SQL ``LIKE``, so ``_`` and ``%`` in an
entity name acted as wildcards — ``test_entity`` matched ``testXentity`` and the
fact was linked to the wrong (or a duplicate) entity. The exact-name match now
uses ``= ? COLLATE NOCASE`` and the alias search escapes LIKE wildcards, while
keeping the intended case-insensitive matching.
"""

from plugins.memory.holographic.store import MemoryStore, _escape_like


def _store():
    return MemoryStore(":memory:")


def _name_of(store, entity_id):
    return store._conn.execute(
        "SELECT name FROM entities WHERE entity_id = ?", (entity_id,)
    ).fetchone()["name"]


def test_underscore_is_not_a_wildcard():
    store = _store()
    store._conn.execute("INSERT INTO entities (name) VALUES ('testXentity')")
    store._conn.commit()
    xid = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name='testXentity'"
    ).fetchone()["entity_id"]

    # 'test_entity' must NOT match 'testXentity' — a new, distinct entity.
    rid = store._resolve_entity("test_entity")
    assert rid != xid
    assert _name_of(store, rid) == "test_entity"


def test_percent_is_not_a_wildcard():
    store = _store()
    store._conn.execute("INSERT INTO entities (name) VALUES ('100percent')")
    store._conn.commit()
    pid = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name='100percent'"
    ).fetchone()["entity_id"]
    # '100%' would match everything-100-prefixed under LIKE; must not here.
    assert store._resolve_entity("100%") != pid


def test_case_insensitive_dedup_preserved():
    store = _store()
    first = store._resolve_entity("Apple")
    again = store._resolve_entity("apple")
    assert first == again  # same entity, not a duplicate


def test_alias_exact_token_match():
    store = _store()
    store._conn.execute(
        "INSERT INTO entities (name, aliases) VALUES ('SomeOrg', 'beta_user')"
    )
    store._conn.commit()
    oid = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name='SomeOrg'"
    ).fetchone()["entity_id"]

    assert store._resolve_entity("beta_user") == oid       # exact alias matches
    assert store._resolve_entity("betaYuser") != oid       # '_' not a wildcard


def test_escape_like_helper():
    assert _escape_like("a_b") == "a\\_b"
    assert _escape_like("100%") == "100\\%"
    assert _escape_like("a\\b") == "a\\\\b"
    assert _escape_like("plain") == "plain"
