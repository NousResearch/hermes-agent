"""Worker transcript writes that run the conversation family's append body: an explicit epoch timestamp survives the
worker client ``agent/runtime_session_store.py`` and the verb, and a process killed inside the transcript or usage
receipt transaction publishes nothing, then replays exactly."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tests.persistence_sqlite.workers._support import (
    CRASH_OPERATIONS, EXECUTION, SESSION, execution, ident, kill_inside_receipt, revision, seed_store, store_path,
)

STAMP = 1700000000.1234567


def test_m1_worker_client_keeps_the_explicit_epoch_through_the_verb(backend, scope_a, tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore

    path = store_path(scope_a)
    store = backend.open(scope_a)
    epoch = seed_store(store)

    def rpc(method: str, **params: Any) -> dict[str, Any]:
        assert (method, params["operation"]) == ("worker.persist", "transcript.append")
        result = store.worker_executions.append_transcript(
            **ident(params["epoch"], params["sequence"]), messages=params["payload"]["messages"],
            turn_lease_holder=params["payload"]["turn_lease_holder"])
        return {"count": result.count, "annotations": [a.as_wire() for a in result.annotations]}

    scope = {"epoch": epoch, "execution_id": EXECUTION, "session_id": SESSION, "generation": 0}
    client = RuntimeSessionStore(rpc, scope, tmp_path / "outbox")
    rows = [{"role": "user", "content": "exact", "timestamp": STAMP}]
    assert client.append_messages_batch(SESSION, rows) == 1
    assert rows[0]["timestamp"] == STAMP and type(rows[0]["_row_id"]) is int
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT timestamp FROM messages WHERE session_id=?", (SESSION,)).fetchone()[0] == STAMP


@pytest.mark.parametrize("operation", ["append", "usage"])
def test_a_process_killed_inside_the_receipt_transaction_rolls_back_then_replays_exactly(backend, scope_a, operation):
    path = store_path(scope_a)
    store = backend.open(scope_a)
    epoch = seed_store(store)
    before = revision(path)
    kill_inside_receipt(path, scope_a, epoch, operation)

    def landed() -> tuple[int, Any]:
        with sqlite3.connect(path) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (SESSION,)).fetchone()[0]
            tokens = conn.execute("SELECT input_tokens FROM sessions WHERE id=?", (SESSION,)).fetchone()[0]
        return rows, tokens

    assert landed() == (0, 0) or landed() == (0, None)
    assert (execution(store).last_sequence, revision(path)) == (0, before)
    for _attempt in range(2):
        CRASH_OPERATIONS[operation](store, ident(epoch))
        rows, tokens = landed()
        assert (rows, tokens) == ((1, tokens) if operation == "append" else (0, 11))
        assert (execution(store).last_sequence, revision(path)) == (1, before + 1)
