"""Worker transcript writes that run the conversation family's append body: an explicit epoch timestamp survives the
worker wire both ways (also through main's worker client ``agent/runtime_session_store.py``), main's raw request
replays the verb's receipt and the reverse, list content is accepted as main accepts it, and a process killed inside the
transcript or usage receipt transaction publishes nothing, then replays exactly."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.workers._support import (
    CRASH_OPERATIONS, EXECUTION, SESSION, execution, frozen, ident, kill_inside_receipt, revision, seed_store,
    seed_worker, store_path,
)

__all__ = ["frozen", "legacy_store"]

STAMP = 1700000000.1234567


def _legacy_db(path: Any) -> Any:
    from hermes_state import SessionDB

    return SessionDB(db_path=path)


def test_worker_transcript_keeps_exact_epoch_and_raw_receipt_replay(backend, scope_a, legacy_store):
    from hermes_state_runtime import mutate_worker_execution

    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    idn = ident(epoch)
    message = {"role": "user", "content": "text", "timestamp": STAMP}
    result = store.worker_executions.append_transcript(**idn, messages=[message], turn_lease_holder=None)
    assert result.annotations[0].timestamp == STAMP
    db = _legacy_db(path)
    try:
        assert db.get_messages(SESSION)[0]["timestamp"] == STAMP
        replay = mutate_worker_execution(db, **idn, operation="transcript.append",
                                         payload={"messages": [message], "turn_lease_holder": None})
        assert replay["annotations"][0]["timestamp"] == STAMP
        assert replay["count"] == 1
        assert len(db.get_messages(SESSION)) == 1
        # And the reverse: a receipt main committed replays through the verb.
        later = {"role": "assistant", "content": "reply", "timestamp": STAMP + 1}
        committed = mutate_worker_execution(db, **ident(epoch, 2), operation="transcript.append",
                                            payload={"messages": [later], "turn_lease_holder": None})
    finally:
        db.close()
    again = store.worker_executions.append_transcript(**ident(epoch, 2), messages=[later], turn_lease_holder=None)
    assert [a.as_wire() for a in again.annotations] == committed["annotations"]
    assert execution(store).last_sequence == 2


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


def test_list_content_is_accepted_with_the_live_epoch_like_m1(backend, scope_a, frozen):
    from tests.persistence_sqlite.workers._support import legacy_persist, parity

    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    idn = ident(epoch)
    message = {"role": "user", "content": []}
    _before, after = parity(path, legacy_persist("transcript.append", {"messages": [message], "turn_lease_holder": None}, idn),
                           lambda: store.worker_executions.append_transcript(**idn, messages=[message], turn_lease_holder=None),
                           lambda r: {"count": r.count, "annotations": [a.as_wire() for a in r.annotations]})
    assert after[1].count == 1


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
