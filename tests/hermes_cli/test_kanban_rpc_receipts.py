"""Remote-worker board RPC — the durable-receipt contract and run authority.

The property under test is the real transaction boundary, not a mocked one: a
mutation and its receipt must commit together, so that a lost reply is harmless
and a retry can never apply the effect twice.

Deliberately uses real imports and a real on-disk board. Mocking ``connect()``
here would test the mock, not the boundary.
"""

from __future__ import annotations

import pathlib
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_rpc as rpc

FUTURE = 2_000_000_000  # comfortably beyond any test clock


@pytest.fixture()
def board(tmp_path):
    """A disposable board with one running task + run, created by real domain code.

    Both rows carry an UNEXPIRED lease, because an expired one legitimately has no
    authority — tests that want that state set it explicitly.
    """
    db_path = pathlib.Path(tmp_path) / "kanban.db"
    conn = kbc.connect(db_path=db_path)
    task_id = kb.create_task(conn, title="remote fixture", assignee="masisim", body="x")
    if hasattr(task_id, "id"):
        task_id = task_id.id
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET status='running', claim_expires=? WHERE id=?",
                     (FUTURE, task_id))
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, claim_lock, claim_expires) "
            "VALUES (?,?,?,?,?)",
            (task_id, "running", 0, "owner:1", FUTURE),
        )
    run_id = conn.execute(
        "SELECT id FROM task_runs WHERE task_id=?", (task_id,)
    ).fetchone()[0]
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task_id))
    yield conn, task_id, run_id, db_path
    conn.close()


def _frame(task_id, run_id, body, request_id, operation="comment"):
    return {
        "version": rpc.PROTOCOL_VERSION,
        "board_id": "board-uuid-test",
        "actor_task_id": task_id,
        "run_id": run_id,
        "request_id": request_id,
        "operation": operation,
        "payload": {"body": body},
    }


def _send(conn, frame, author="masisim"):
    return rpc.execute_request(rpc.BoardRequest.from_wire(frame), conn=conn, author=author)


def _counts(conn, task_id):
    return (
        conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id=?", (task_id,)).fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM board_rpc_receipts").fetchone()[0],
    )


# --- the receipt contract --------------------------------------------------

def test_first_send_executes_once(board):
    conn, task_id, run_id, _ = board
    result = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    assert result["comment_id"] >= 1
    assert result["author"] == "masisim"
    assert _counts(conn, task_id) == (1, 1)


def test_replay_returns_stored_result_without_reapplying(board):
    """The lost-reply retry: same request id, same payload."""
    conn, task_id, run_id, _ = board
    first = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    second = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    assert first == second
    assert _counts(conn, task_id) == (1, 1)


def test_reused_request_id_with_different_payload_is_a_conflict(board):
    """A request id is bound to its envelope; reusing it for other content must not execute."""
    conn, task_id, run_id, _ = board
    _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    with pytest.raises(rpc.RpcError) as excinfo:
        _send(conn, _frame(task_id, run_id, "DIFFERENT", "req-1"))
    assert excinfo.value.code == "request_id_conflict"
    assert _counts(conn, task_id) == (1, 1)


def test_receipt_replay_precedes_the_liveness_rejection(board):
    """A successful completion closes its run; if the reply was lost, the retry must
    still be served from the receipt rather than rejected as a stale run."""
    conn, task_id, run_id, _ = board
    first = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done', current_run_id=NULL WHERE id=?", (task_id,))
    replayed = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    assert replayed == first
    assert _counts(conn, task_id) == (1, 1)


def test_read_only_operation_writes_no_receipt(board):
    """Polling must not turn into a write."""
    conn, task_id, run_id, _ = board
    result = _send(conn, _frame(task_id, run_id, "", "poll-1", operation=rpc.OP_CLAIM_STATUS))
    assert result["is_live"] is True
    assert result["current_run_id"] == run_id
    assert _counts(conn, task_id) == (0, 0)


# --- request integrity -----------------------------------------------------

def test_protocol_version_is_enforced(board):
    conn, task_id, run_id, _ = board
    frame = _frame(task_id, run_id, "hi", "req-1")
    frame["version"] = rpc.PROTOCOL_VERSION + 1
    with pytest.raises(rpc.RpcError) as excinfo:
        _send(conn, frame)
    assert excinfo.value.code == "unsupported_version"


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_non_integer_protocol_version_is_rejected(board, bad):
    """``True == 1`` and ``1.0 == 1`` in Python, so equality alone is not a type check."""
    conn, task_id, run_id, _ = board
    frame = _frame(task_id, run_id, "hi", "req-1")
    frame["version"] = bad
    with pytest.raises(rpc.RpcError):
        _send(conn, frame)


def test_request_hash_is_stable_across_execution(board):
    """Execution must not mutate the request — the stored hash must keep describing
    what the caller actually sent."""
    conn, task_id, run_id, _ = board
    req = rpc.BoardRequest.from_wire(_frame(task_id, run_id, "hello", "req-1"))
    before = req.request_hash()
    rpc.execute_request(req, conn=conn, author="masisim")
    assert req.request_hash() == before


# --- author is the grant's, never the model's ------------------------------

def test_payload_author_is_ignored(board):
    """The model controls the payload; it must not be able to set the author."""
    conn, task_id, run_id, _ = board
    frame = _frame(task_id, run_id, "hello", "req-1")
    frame["payload"]["_author"] = "somebody-else"
    assert _send(conn, frame)["author"] == "masisim"


def test_missing_author_is_refused_not_defaulted(board):
    """No authenticated author means the channel established none — refuse, never
    fall back to anything the payload supplies."""
    conn, task_id, run_id, _ = board
    frame = _frame(task_id, run_id, "hello", "req-1")
    frame["payload"]["_author"] = "attacker"
    with pytest.raises(rpc.RpcError) as excinfo:
        rpc.execute_request(rpc.BoardRequest.from_wire(frame), conn=conn, author=None)
    assert excinfo.value.code == "no_authenticated_author"
    assert _counts(conn, task_id) == (0, 0)


# --- authority: expired / closed / nonexistent must never read as live -----

def test_expired_lease_is_not_live_and_cannot_mutate(board):
    conn, task_id, run_id, _ = board
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (1, task_id))
    status = _send(conn, _frame(task_id, run_id, "", "p", operation=rpc.OP_CLAIM_STATUS))
    assert status["is_live"] is False
    assert any("expired" in p for p in status["problems"])
    with pytest.raises(rpc.RpcError) as excinfo:
        _send(conn, _frame(task_id, run_id, "hi", "req-x"))
    assert excinfo.value.code == "no_authority"
    assert _counts(conn, task_id) == (0, 0)


def test_closed_run_is_not_live(board):
    conn, task_id, run_id, _ = board
    with kbc.write_txn(conn):
        conn.execute("UPDATE task_runs SET status='done', ended_at=? WHERE id=?",
                     (int(time.time()), run_id))
    status = _send(conn, _frame(task_id, run_id, "", "p", operation=rpc.OP_CLAIM_STATUS))
    assert status["is_live"] is False
    assert any("closed" in p for p in status["problems"])


def test_missing_run_is_not_live(board):
    """A nonexistent run must not be reported live."""
    conn, task_id, run_id, _ = board
    status = _send(conn, _frame(task_id, run_id + 999, "", "p", operation=rpc.OP_CLAIM_STATUS))
    assert status["is_live"] is False
    assert any("does not exist" in p for p in status["problems"])


def test_stale_run_cannot_mutate(board):
    """Once the card is no longer running THIS run, a new mutation is refused."""
    conn, task_id, run_id, _ = board
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done', current_run_id=NULL WHERE id=?", (task_id,))
    with pytest.raises(rpc.RpcError) as excinfo:
        _send(conn, _frame(task_id, run_id, "late", "req-2"))
    assert excinfo.value.code == "no_authority"
    assert _counts(conn, task_id) == (0, 0)


# --- the observer boundary -------------------------------------------------

def test_comment_hook_fires_only_after_the_commit(board, monkeypatch):
    """The hook must not run inside the transaction.

    The observer opens its OWN connection and reads the comment. If the hook ran
    while our transaction was still open, that read would see zero rows (the insert
    is uncommitted) and this assertion fails.
    """
    conn, task_id, run_id, db_path = board
    seen = {}

    def observer(_conn, event, tid, **fields):
        probe = kbc.connect(db_path=db_path)
        try:
            seen["visible"] = probe.execute(
                "SELECT COUNT(*) FROM task_comments WHERE task_id=?", (tid,)
            ).fetchone()[0]
        finally:
            probe.close()
        seen["event"] = event

    monkeypatch.setattr(kb, "_fire_task_hook_by_id", observer)
    _send(conn, _frame(task_id, run_id, "hello", "req-1"))

    assert seen.get("event") == "kanban_task_commented"
    assert seen.get("visible") == 1, "observer ran before the comment was committed"


def test_hook_does_not_fire_when_the_transaction_rolls_back(board, monkeypatch):
    """An external effect must not escape a rolled-back transaction."""
    conn, task_id, run_id, _ = board
    fired = []
    monkeypatch.setattr(kb, "_fire_task_hook_by_id", lambda *a, **k: fired.append(a))

    def explode(conn_, req, result, **kw):
        raise RuntimeError("injected failure after the mutation, before the receipt")

    monkeypatch.setattr(rpc, "_store_receipt", explode)
    with pytest.raises(RuntimeError):
        _send(conn, _frame(task_id, run_id, "hello", "req-1"))

    assert fired == [], "hook fired for a transaction that rolled back"
    assert _counts(conn, task_id) == (0, 0)
