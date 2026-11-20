"""A worker's receipt and transcript stay in its assigned profile."""

from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_session
from tests.persistence_sqlite.workers._support import SESSION, ident, seed_store


def test_worker_receipt_only_writes_its_own_profile_when_session_ids_match(backend, scope_a, scope_b):
    a, b = backend.open(scope_a), backend.open(scope_b)
    epoch = seed_store(a)
    seed_session(b, SESSION, "cli")
    args = dict(ident(epoch, 1), operation="transcript.append", payload={
        "messages": [{"role": "user", "content": "A only"}],
    })

    first = a.worker_executions.persist_operation(**args)
    assert a.worker_executions.persist_operation(**args) == first
    assert [row["content"] for row in transcript_rows(a, SESSION)] == ["A only"]
    assert not transcript_rows(b, SESSION)
