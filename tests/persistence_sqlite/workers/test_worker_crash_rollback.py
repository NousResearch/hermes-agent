"""A process killed inside the worker receipt transaction publishes nothing, and the exact retry then lands the whole
operation once (the transcript and usage cases are in ``test_worker_transcript.py``)."""

import sqlite3

from tests.persistence_sqlite.workers._support import (
    CRASH_OPERATIONS, SESSION, execution, ident, kill_inside_receipt, revision, seed_store, store_path,
)


def test_a_process_killed_inside_the_receipt_transaction_rolls_back_then_replays_exactly(backend, scope_a):
    path = store_path(scope_a)
    store = backend.open(scope_a)
    epoch = seed_store(store)
    before = revision(path)
    kill_inside_receipt(path, scope_a, epoch, "fallback")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT compression_fallback_streak FROM sessions WHERE id=?", (SESSION,)).fetchone()[0] in (None, 0)
        assert conn.execute("SELECT COUNT(*) FROM worker_receipts").fetchone()[0] == 0
    assert (execution(store).last_sequence, revision(path)) == (0, before)
    for _attempt in range(2):
        CRASH_OPERATIONS["fallback"](store, ident(epoch))
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT compression_fallback_streak FROM sessions WHERE id=?", (SESSION,)).fetchone()[0] == 7
            assert conn.execute("SELECT COUNT(*) FROM worker_receipts").fetchone()[0] == 1
        assert (execution(store).last_sequence, revision(path)) == (1, before + 1)
