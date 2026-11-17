"""The boot heal for handed-off local receipts scans the store without the state.db write lock.

``release_handed_off_receipts`` joined every local receipt to its session row INSIDE the write
transaction, so every gateway start held the write lock for as long as that scan took (seconds on a
large cold store) and every other writer waited. Now the scan is a read; a boot that finds nothing
opens no write transaction, and one that finds receipts re-checks only those inside the write.
"""
import json
import threading

import pytest

import hermes_state_local
from gateway.session_local_recovery import release_handed_off_receipts
from hermes_state import SessionDB
from hermes_state_local import POLICY_PREFIX
from hermes_state_runtime import begin_runtime_epoch


def _local_chat(db, sid, *, moved_to=None):
    """A local chat with its creation receipt; *moved_to* gives its row a messaging lane's route."""
    db.create_session(sid, "local")
    receipt = {"entry": {"session_id": sid}, "route": "local-route", "lineage": [sid]}

    def seed(conn):
        conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)", (POLICY_PREFIX + sid, json.dumps(receipt)))
        if moved_to:
            conn.execute("UPDATE sessions SET session_key=?, source='telegram' WHERE id=?", (moved_to, sid))
    db._execute_write(seed)


def _track_writes(db, monkeypatch):
    """Count write transactions and expose whether this thread is inside one right now."""
    state = threading.local()
    writes = []
    real = db._execute_write

    def tracked(fn, *a, **k):
        writes.append(fn)

        def inside(conn):
            state.inside = True
            try:
                return fn(conn)
            finally:
                state.inside = False
        return real(inside, *a, **k)
    monkeypatch.setattr(db, "_execute_write", tracked)
    return writes, lambda: getattr(state, "inside", False)


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    yield handle
    handle.close()


def test_a_boot_with_nothing_to_release_opens_no_write_transaction(db, monkeypatch):
    for n in range(20):
        _local_chat(db, f"chat{n}")  # their rows carry no lane: nothing was handed off
    epoch = begin_runtime_epoch(db, instance_id="boot")
    writes, _inside = _track_writes(db, monkeypatch)
    release_handed_off_receipts(db, epoch=epoch)
    assert writes == []


def test_the_scan_runs_outside_the_write_and_the_write_probes_only_what_it_found(db, monkeypatch):
    for n in range(5):
        _local_chat(db, f"chat{n}")
    _local_chat(db, "moved", moved_to="agent:main:telegram:dm:42")
    epoch = begin_runtime_epoch(db, instance_id="boot")
    _writes, inside = _track_writes(db, monkeypatch)
    calls = []
    real = hermes_state_local.moved_receipts

    def watching(conn, *args, **kwargs):
        calls.append((kwargs.get("owners"), inside()))
        return real(conn, *args, **kwargs)
    monkeypatch.setattr(hermes_state_local, "moved_receipts", watching)

    release_handed_off_receipts(db, epoch=epoch)

    assert calls[0] == (None, False), calls  # the whole-store scan runs outside any write transaction
    assert calls[1:] == [(["moved"], True)], calls  # the write re-checks the one receipt it found, by key
    with db._read_ctx() as conn:
        keys = {row[0] for row in conn.execute("SELECT key FROM state_meta WHERE key GLOB ?", (POLICY_PREFIX + "*",))}
    assert POLICY_PREFIX + "moved" not in keys and POLICY_PREFIX + "chat0" in keys, keys  # released, others kept


def test_a_receipt_that_stopped_being_moved_between_the_scan_and_the_write_is_kept(db, monkeypatch):
    _local_chat(db, "moved", moved_to="agent:main:telegram:dm:42")
    epoch = begin_runtime_epoch(db, instance_id="boot")
    real = hermes_state_local.moved_receipts

    def raced(conn, *args, **kwargs):
        if kwargs.get("owners") is not None:  # between the read and the write the lane let go of the row
            conn.execute("UPDATE sessions SET session_key='local-route' WHERE id='moved'")
        return real(conn, *args, **kwargs)
    monkeypatch.setattr(hermes_state_local, "moved_receipts", raced)
    release_handed_off_receipts(db, epoch=epoch)
    with db._read_ctx() as conn:
        assert conn.execute("SELECT count(*) FROM state_meta WHERE key=?", (POLICY_PREFIX + "moved",)).fetchone()[0] == 1


def test_a_point_probe_and_the_whole_scan_agree(db):
    _local_chat(db, "a", moved_to="agent:main:telegram:dm:1")
    _local_chat(db, "b", moved_to="agent:main:telegram:dm:2")
    _local_chat(db, "c")
    with db._read_ctx() as conn:
        whole = hermes_state_local.moved_receipts(conn)
        assert set(whole) == {"a", "b"}
        assert hermes_state_local.moved_receipts(conn, owners=["a", "c", "missing"]) == {"a": whole["a"]}
        assert hermes_state_local.moved_receipts(conn, owners=[]) == {}
