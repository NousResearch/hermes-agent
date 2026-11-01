"""Wave 55 battery B10 — W54-F009 regression tests for gateway/delivery_ledger.py.

Race: ``record_obligation`` used ``INSERT OR REPLACE`` unconditionally. A delivery
worker (same gateway process) can own a row in state='attempting' while a new turn
for the same session answers the same inbound message with byte-identical content:
``compute_obligation_id`` yields the SAME obligation_id, and the REPLACE clobbers the
in-flight row back to state='pending', attempts=0.

Consequence: the ledger's core honesty invariant ("pending = never started,
redeliver plainly" — see delivery_ledger.py:397-401) is falsified; the marker an
ambiguous in-flight send requires is silently dropped on the next sweep.

The fix contract: guarded upsert — ``INSERT ... ON CONFLICT(obligation_id) DO UPDATE
... WHERE delivery_obligations.state <> 'attempting'`` — re-recording an in-flight
row is a no-op, while re-arming pending/failed/terminal rows keeps today's
idempotent semantics.

Test 1 forces the harmful interleaving with a threading.Barrier(2) (zero sleeps):
the worker claims the row before the producer's re-record fires, so the producer's
write provably hits a row the worker owns in 'attempting'.
"""

import os
import threading
import time

import pytest

from gateway import delivery_ledger as dl

FAKE_START = 424_242

SESSION_KEY = "agent:main:slack:channel:C1"
MESSAGE_REF = "inbound-msg-1"
CONTENT = "final answer v1"


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Per-test fresh ledger DB + stable owner-process start time."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: FAKE_START)
    yield


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            """SELECT state, attempts, owner_pid, owner_started_at, content
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0], "attempts": r[1], "owner_pid": r[2],
        "owner_started_at": r[3], "content": r[4],
    }


def _record(oid):
    dl.record_obligation(
        obligation_id=oid, session_key=SESSION_KEY, platform="slack",
        chat_id="C1", thread_id="171.001", content=CONTENT)


def _claim_as_worker(oid, pid, started):
    """The sweep-shaped guarded claim: worker grabs the pending row for its send
    (attempts+1, state='attempting', owner stamped)."""
    with dl._DB_LOCK, dl._transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state='attempting', attempts=attempts+1,
                   owner_pid=?, owner_started_at=?, updated_at=?
               WHERE obligation_id=? AND state='pending'""",
            (pid, started, time.time(), oid))
    assert cursor.rowcount == 1


def test_concurrent_record_obligation_does_not_clobber_inflight_attempt():
    """Re-recording an obligation a delivery worker already owns in 'attempting'
    must be a no-op: state/attempts/owner/marker semantics all preserved."""
    oid = dl.compute_obligation_id(SESSION_KEY, MESSAGE_REF, CONTENT)
    pid = os.getpid()

    barrier = threading.Barrier(2)
    errors = []

    def worker():
        try:
            _record(oid)
            _claim_as_worker(oid, pid, FAKE_START)  # owns the row; awaiting platform ACK
            barrier.wait(timeout=30)
        except Exception as exc:  # pragma: no cover - failure plumbing
            errors.append(("worker", exc))
            barrier.abort()

    def producer():
        try:
            barrier.wait(timeout=30)
            # Same session + same inbound message id + byte-identical content =>
            # same obligation_id; hits the row the worker currently owns.
            _record(oid)
        except Exception as exc:  # pragma: no cover - failure plumbing
            errors.append(("producer", exc))
            barrier.abort()

    worker_t = threading.Thread(target=worker, name="worker")
    producer_t = threading.Thread(target=producer, name="producer")
    worker_t.start()
    producer_t.start()
    worker_t.join(timeout=60)
    producer_t.join(timeout=60)
    assert not errors, errors

    # The in-flight row must survive untouched. state='pending' here would mean the
    # next sweep redelivers plainly WITHOUT the recovered-reply marker (marker loss).
    row = _row(oid)
    assert row is not None
    assert row["state"] == "attempting"
    assert row["attempts"] == 1
    assert row["owner_pid"] == pid
    assert row["owner_started_at"] == FAKE_START
    assert row["content"] == CONTENT


def test_record_obligation_rearms_terminal_and_pending_rows():
    """The guard must be narrow: re-recording terminal/failed or idle-pending rows
    still re-arms them to 'pending'/attempts=0 exactly as INSERT OR REPLACE did."""
    oid = dl.compute_obligation_id(SESSION_KEY, MESSAGE_REF, CONTENT)
    _record(oid)
    assert _row(oid)["state"] == "pending"

    # A terminal (failed) row is re-armed by a re-record of the same content.
    dl.mark_failed(oid)
    assert _row(oid)["state"] == "failed"
    _record(oid)
    row = _row(oid)
    assert row["state"] == "pending"
    assert row["attempts"] == 0

    # An idle pending row re-records idempotently.
    _record(oid)
    row = _row(oid)
    assert row["state"] == "pending"
    assert row["attempts"] == 0