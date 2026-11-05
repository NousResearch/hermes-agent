"""``record_session_peer`` binds the session id as SQLite does: integers and booleans take the text column's affinity,
and a value SQLite cannot bind is refused as a store error chaining the driver's binding error, writing nothing."""

from __future__ import annotations

import sqlite3

import pytest

from persistence import PermanentStoreError


def _keys(store) -> dict[str, str]:
    return {row["id"]: row["session_key"] for row in store.holder()._read_all("SELECT id, session_key FROM sessions")}


@pytest.mark.parametrize("session_id, stored_id", [(123, "123"), (True, "1")])
def test_peer_publication_uses_sqlite_identifier_affinity(backend, scope_a, session_id, stored_id):
    store = backend.open(scope_a)
    store.routing.record_session_peer(session_id, source="telegram", session_key="peer")
    assert _keys(store) == {stored_id: "peer"}
    if session_id is True:
        assert "True" not in _keys(store)
        store.routing.record_session_peer(1, source="telegram", session_key="updated")
        assert _keys(store) == {"1": "updated"}


@pytest.mark.parametrize("session_id", [[1], {"a": 1}])
def test_peer_publication_keeps_binding_refusals(backend, scope_a, session_id):
    store = backend.open(scope_a)
    with pytest.raises(PermanentStoreError) as failure:
        store.routing.record_session_peer(session_id, source="telegram", session_key="peer")
    assert isinstance(failure.value.__cause__, sqlite3.ProgrammingError)
    expected = f"Error binding parameter 10: type '{type(session_id).__name__}' is not supported"
    assert str(failure.value.__cause__) == expected
    assert _keys(store) == {}
