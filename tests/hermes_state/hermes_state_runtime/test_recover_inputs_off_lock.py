"""The boot recovery of lost session inputs scans the store without the state.db write lock.

``recover_session_inputs`` ran its four stale-row predicates (``worker_executions`` children, running
workers, steering admissions, started admissions) inside one write transaction. None is indexed, and a
``session_admissions`` row keeps its large ``payload_json`` before the columns the scan tests, so the
scan reads the payload overflow pages, and every gateway start held the write lock for all four. Now
the scans are reads; the write touches what they found, by key, and every UPDATE repeats its predicate.
"""
import contextlib
import re
import sqlite3

import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB

SCAN = re.compile(r"owner_epoch!=\d+")
KEYED = re.compile(r"\b(?:seq|execution_id)=")


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    handle.create_session("s", source="test")
    yield handle
    handle.close()


def _admission(conn, n, *, status, outcome=None, epoch, generation=1):
    conn.execute(
        "INSERT INTO session_admissions(admission_id,request_id,principal_id,target_session_id,lineage_json,"
        "payload_json,payload_digest,intent,status,outcome,owner_epoch,generation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (f"adm-{n}", f"req-{n}", "human", "s", "[]", "{}", f"digest-{n}", "queue", status, outcome, epoch, generation))


def _execution(conn, name, *, kind, status, epoch):
    conn.execute(
        "INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
        "VALUES(?,?,?,?,?,?,?)", (name, "s", kind, epoch, 1, status, "digest"))


def _seed_lost_owner(db):
    """What a crashed gateway of an older epoch leaves behind, plus one row of each kind already settled."""
    old = rt.begin_runtime_epoch(db, instance_id="old-owner")

    def seed(conn):
        _execution(conn, "compute", kind="compute", status="running", epoch=old)
        _execution(conn, "kid", kind="child", status="running", epoch=old)
        _admission(conn, 1, status="started", epoch=old)
        _admission(conn, 2, status="terminal", outcome=rt.STEERING, epoch=old)
        _admission(conn, 3, status="terminal", outcome="completed", epoch=old)
    db._execute_write(seed)
    return old, rt.begin_runtime_epoch(db, instance_id="new-owner")


def _rows(db):
    with db._read_ctx() as conn:
        return (dict(conn.execute("SELECT execution_id,status FROM worker_executions").fetchall()),
                dict(conn.execute("SELECT admission_id,status FROM session_admissions").fetchall()),
                dict(conn.execute("SELECT admission_id,outcome FROM session_admissions").fetchall()))


def _count_writes(db, monkeypatch):
    seen = []
    real = db._execute_write
    monkeypatch.setattr(db, "_execute_write", lambda fn, *a, **k: (seen.append(fn), real(fn, *a, **k))[1])
    return seen


def test_a_boot_with_nothing_lost_opens_no_write_transaction(db, monkeypatch):
    old = rt.begin_runtime_epoch(db, instance_id="old-owner")
    db._execute_write(lambda conn: (_execution(conn, "done", kind="compute", status="terminal", epoch=old),
                                    _admission(conn, 1, status="terminal", outcome="completed", epoch=old)))
    epoch = rt.begin_runtime_epoch(db, instance_id="new-owner")
    writes = _count_writes(db, monkeypatch)
    assert rt.recover_session_inputs(db, epoch=epoch) == 0
    assert writes == []


def test_the_scans_hold_no_write_lock_and_the_write_only_looks_rows_up_by_key(db, monkeypatch):
    _old, epoch = _seed_lost_owner(db)
    seen = []  # (where, statement, whether another connection could take the write lock right then)

    def tracer(where):
        def trace(statement):
            if not SCAN.search(statement) or statement.lstrip().upper().startswith(("BEGIN", "COMMIT")):
                return
            other = sqlite3.connect(db.db_path, timeout=0.05)
            try:
                other.execute("BEGIN IMMEDIATE")
                other.execute("ROLLBACK")
                free = True
            except sqlite3.OperationalError:
                free = False
            finally:
                other.close()
            seen.append((where, statement, free))
        return trace

    real_write, real_read = db._execute_write, db._read_ctx

    def traced_write(fn, *a, **k):
        def inner(conn):
            conn.set_trace_callback(tracer("write"))
            try:
                return fn(conn)
            finally:
                conn.set_trace_callback(None)
        return real_write(inner, *a, **k)

    @contextlib.contextmanager
    def traced_read(*a, **k):
        with real_read(*a, **k) as conn:
            conn.set_trace_callback(tracer("read"))
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)
    monkeypatch.setattr(db, "_execute_write", traced_write)
    monkeypatch.setattr(db, "_read_ctx", traced_read)

    assert rt.recover_session_inputs(db, epoch=epoch) == 1

    scans = [(where, statement) for where, statement, _free in seen if not KEYED.search(statement)]
    assert len(scans) >= 4, seen  # worker children, running workers, steering and started admissions
    assert [where for where, _ in scans] == ["read"] * len(scans), scans  # no full scan inside the write
    assert all(free for where, statement, free in seen if not KEYED.search(statement)), seen  # lock free meanwhile
    keyed_writes = [statement for where, statement, _free in seen if where == "write"]
    assert keyed_writes and all(KEYED.search(statement) for statement in keyed_writes), keyed_writes


def test_every_kind_of_lost_row_is_recovered_as_before(db):
    _old, epoch = _seed_lost_owner(db)

    def current(conn):
        _execution(conn, "mine", kind="compute", status="running", epoch=epoch)
        _admission(conn, 4, status="started", epoch=epoch)
        _admission(conn, 5, status="terminal", outcome=rt.STEERING, epoch=epoch)
    db._execute_write(current)

    assert rt.recover_session_inputs(db, epoch=epoch) == 1
    executions, statuses, outcomes = _rows(db)
    assert executions == {"compute": "unknown", "kid": "terminal", "mine": "running"}
    assert statuses == {"adm-1": "unknown", "adm-2": "terminal", "adm-3": "terminal", "adm-4": "started", "adm-5": "terminal"}
    assert outcomes == {"adm-1": None, "adm-2": "interrupted", "adm-3": "completed", "adm-4": None, "adm-5": rt.STEERING}
    assert rt.recover_session_inputs(db, epoch=epoch) == 0  # nothing left: the second boot finds nothing
    assert _rows(db) == (executions, statuses, outcomes)


def test_a_row_that_stopped_being_stale_between_the_scan_and_the_write_is_left_alone(db, monkeypatch):
    _old, epoch = _seed_lost_owner(db)
    real = db._execute_write

    def raced(fn, *a, **k):
        # The scan has run; before the write begins, the rows it found are adopted or settled elsewhere.
        def settle(conn):
            conn.execute("UPDATE worker_executions SET owner_epoch=? WHERE execution_id IN ('compute','kid')", (epoch,))
            conn.execute("UPDATE session_admissions SET owner_epoch=? WHERE admission_id IN ('adm-1','adm-2')", (epoch,))
        real(settle)
        return real(fn, *a, **k)
    monkeypatch.setattr(db, "_execute_write", raced)

    assert rt.recover_session_inputs(db, epoch=epoch) == 0
    executions, statuses, outcomes = _rows(db)
    assert executions == {"compute": "running", "kid": "running"}
    assert statuses["adm-1"] == "started" and outcomes["adm-2"] == rt.STEERING


def test_an_old_owner_is_refused_even_when_nothing_is_lost(db):
    old = rt.begin_runtime_epoch(db, instance_id="old-owner")
    rt.begin_runtime_epoch(db, instance_id="new-owner")
    with pytest.raises(rt.RuntimeStoreError, match="stale_epoch"):
        rt.recover_session_inputs(db, epoch=old)


def test_many_lost_inputs_are_all_recovered(db):
    old = rt.begin_runtime_epoch(db, instance_id="old-owner")

    def seed(conn):
        for n in range(700):
            _admission(conn, n, status="started", epoch=old)
            _execution(conn, f"w{n}", kind="compute", status="registered", epoch=old)
    db._execute_write(seed)
    epoch = rt.begin_runtime_epoch(db, instance_id="new-owner")
    assert rt.recover_session_inputs(db, epoch=epoch) == 700
    executions, statuses, _outcomes = _rows(db)
    assert set(executions.values()) == {"unknown"} and set(statuses.values()) == {"unknown"}
