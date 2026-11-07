"""A malformed ``session.next_title`` request keeps main's precedence: the payload check is the handler's, so it answers after the receipt fences;
a stale epoch answers ``stale_epoch`` first, a live one ``invalid_params``; nothing is written either way."""

import sqlite3

import pytest

from persistence.core.errors import AuthorityRefusal, FenceStale
from tests.persistence_sqlite.workers._support import EXECUTION, ident, legacy_persist, parity, seed_worker, store_path


@pytest.mark.parametrize("stale", [False, True])
def test_worker_title_keeps_boundary_validation(backend, scope_a, stale):
    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    idn = ident(0 if stale else epoch)
    _before, after = parity(path, legacy_persist("session.next_title", {"base_title": []}, idn),
                           lambda: store.worker_executions.next_session_title(**idn, base_title=[]))
    assert isinstance(after[1], FenceStale if stale else AuthorityRefusal)
    assert after[1].reason == ("stale_epoch" if stale else "invalid_params")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT status,last_sequence FROM worker_executions WHERE execution_id=?",
                            (EXECUTION,)).fetchall() == [("registered", 0)]
