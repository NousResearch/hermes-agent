"""Worker appends respect compression closure and publish session activity."""

import pytest

from persistence.conversation.sessions import END_REASON_COMPRESSION
from persistence.core.errors import CompressionClosed
from persistence.workers.sessions import WorkerActivity
from tests.contracts.workers._support import SESSION, assistant, ident, last_sequence, worker


def test_worker_append_leaves_the_activity_stamp_to_session_activity(backend, scope_a):
    # The worker append writes no activity column; the worker stamps activity through ``session.activity``.
    store, epoch = worker(backend, scope_a)
    store.worker_executions.append_transcript(**ident(epoch), messages=[assistant("body")])
    assert store.sessions.get(SESSION).last_activity_at is None
    stamp = 1_790_000_000.5
    store.worker_executions.touch_session_activity(**ident(epoch, sequence=2), activity=WorkerActivity(ts=stamp, description="tool: x", provenance="tool"))
    assert store.sessions.get(SESSION).last_activity_at == stamp


def test_worker_append_refuses_compressed_parent(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    store.sessions.end(SESSION, END_REASON_COMPRESSION)
    with pytest.raises(CompressionClosed):
        store.worker_executions.append_transcript(**ident(epoch), messages=[assistant("closed")])
    assert store.messages.read_transcript(SESSION) == ()
    assert last_sequence(backend, scope_a, epoch) == 0
