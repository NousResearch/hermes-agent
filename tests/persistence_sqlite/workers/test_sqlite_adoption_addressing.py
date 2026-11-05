"""Adoption binds its addressing inputs exactly as given: a non-string session id is a foreign assignment, a non-string execution id finds no row or cannot
be bound, and nothing about the execution changes."""

import sqlite3

import pytest

from persistence import PermanentStoreError
from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.workers._support import SECRET, parity, seed_worker, store_path

__all__ = ["legacy_store"]

SESSION, EXECUTION = "s", "w"


@pytest.mark.parametrize("field,value", [("session_id", None), ("session_id", True), ("session_id", {"a": 1}),
                                         ("execution_id", None), ("execution_id", True), ("execution_id", [1])])
def test_adoption_does_not_stringify_addressing_inputs(backend, scope_a, legacy_store, field, value):
    from hermes_state_runtime import adopt_worker_execution, register_worker_execution

    path = store_path(scope_a)
    identity = {"session_id": SESSION, "execution_id": EXECUTION}
    identity[field] = str(value)
    def addressed(db, epoch):
        db.create_session(identity["session_id"], source="cli")
        register_worker_execution(db, epoch=epoch, **identity, generation=0, kind="compute", adoption_secret=SECRET)

    epoch = seed_worker(path, extra=addressed)
    store = backend.open(scope_a)
    args = dict(epoch=epoch, **{**identity, field: value}, generation=0, adoption_secret=SECRET)
    _before, after = parity(path, lambda db: adopt_worker_execution(db, **args), lambda: store.worker_executions.adopt(**args))
    if isinstance(value, list):
        # Main lets the binding error escape; the backend raises it as a permanent store error caused by it.
        assert isinstance(after[1], PermanentStoreError)
        assert isinstance(after[1].__cause__, sqlite3.ProgrammingError)
        assert str(after[1].__cause__) == "Error binding parameter 1: type 'list' is not supported"
    else:
        assert isinstance(after[1], AuthorityRefusal)
        assert after[1].reason == ("permission_denied" if field == "session_id" else "not_found")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT status,last_sequence FROM worker_executions WHERE execution_id=?",
                            (identity["execution_id"],)).fetchall() == [("registered", 0)]
