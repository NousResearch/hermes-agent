"""A refused owner write must leave the receipt sequence available for a legitimate retry."""

import pytest

from persistence import PersistenceError
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import transcript_rows
from tests.persistence_sqlite.workers._support import SESSION, ident, seed_store


def test_worker_receipt_refuses_ambiguous_session_without_consuming_sequence(backend, scope_a, scope_b):
    a, b = backend.open(scope_a), backend.open(scope_b)
    epoch = seed_store(a)
    holder = raw_holder(b)
    holder._execute_write(lambda conn: conn.execute(
        "INSERT INTO sessions(id, source, started_at, profile_name) VALUES (?, 'cli', 1, ?)",
        (SESSION, scope_b.name),
    ))
    args = dict(ident(epoch, 1), operation="transcript.append", payload={
        "messages": [{"role": "user", "content": "Write once if the owner is clear"}],
    })

    with pytest.raises(PersistenceError):
        a.worker_executions.persist_operation(**args)
    assert not transcript_rows(a, SESSION)
    assert not transcript_rows(b, SESSION)

    holder._execute_write(lambda conn: conn.execute("DELETE FROM sessions WHERE id=?", (SESSION,)))
    first = a.worker_executions.persist_operation(**args)
    assert a.worker_executions.persist_operation(**args) == first
    assert len(transcript_rows(a, SESSION)) == 1
