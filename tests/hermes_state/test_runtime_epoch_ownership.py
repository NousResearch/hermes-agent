"""An epoch bump is a takeover: it must not fence out a live owner (#142330).

The runtime epoch is the fence every write is checked against, and the owner captures it
ONCE at boot (``gateway/session_authority.py::initialize_session_authority``). A bump by a
process that does not own the profile therefore leaves the live owner permanently behind the
fence: every write on that store fails with ``stale_epoch`` until the gateway restarts, and
``/reset`` does not recover it.

``begin_runtime_epoch``'s own docstring said the caller MUST hold the profile ownership lock,
but nothing checked it - the invariant was held by convention, and an ordinary agent turn
could reach it (a script that opens the store and calls it).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def owner_process():
    """A live foreign process standing in for the gateway that owns the profile."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _write_lock_record(home, pid: int) -> None:
    """The profile's ``gateway.lock`` record naming *pid* as the holder."""
    from gateway.status import _build_pid_record, _get_process_start_time

    record = {**_build_pid_record(), "pid": pid, "start_time": _get_process_start_time(pid)}
    (home / "gateway.lock").write_text(json.dumps(record), encoding="utf-8")


def _epoch_of(db) -> int | None:
    row = db._conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()
    return row[0] if row else None


def test_a_stranger_cannot_bump_the_epoch_of_a_live_owners_store(tmp_path, owner_process):
    """The reported failure: a process that does not own the profile must not fence the owner out."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        _write_lock_record(tmp_path, owner_process)

        with pytest.raises(rt.RuntimeStoreError) as exc:
            rt.begin_runtime_epoch(db, instance_id="stranger")
        assert exc.value.reason == "ownership_conflict"

        # The owner's fence must still stand: the bump did not move.
        assert _epoch_of(db) == epoch
    finally:
        db.close()


def test_the_owner_itself_may_bump_the_epoch(tmp_path, owner_process):
    """The holder of the profile lock is the one taking the store over - it must be allowed."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        first = rt.begin_runtime_epoch(db, instance_id="owner")
        _write_lock_record(tmp_path, os.getpid())
        second = rt.begin_runtime_epoch(db, instance_id="owner-again")
        assert second > first
    finally:
        db.close()


def test_the_verdict_is_per_profile(tmp_path, owner_process):
    """A -> B: the owned profile refuses the stranger, and the other profile is untouched.

    The epoch row is a singleton of ITS store, so the refusal is about the profile that was
    asked - a second profile's own store is still free to be taken over."""
    import hermes_state_runtime as rt

    owned = tmp_path / "owned"
    free = tmp_path / "free"
    owned.mkdir()
    free.mkdir()
    _write_lock_record(owned, owner_process)

    db_owned = SessionDB(db_path=owned / "state.db")
    db_free = SessionDB(db_path=free / "state.db")
    try:
        with pytest.raises(rt.RuntimeStoreError) as exc:
            rt.begin_runtime_epoch(db_owned, instance_id="stranger")
        assert exc.value.reason == "ownership_conflict"
        assert _epoch_of(db_owned) is None, "the refused bump must not have created the row"

        # The other profile's store is a different store: free to be taken over.
        assert rt.begin_runtime_epoch(db_free, instance_id="owner") == 1
    finally:
        db_owned.close()
        db_free.close()


def test_an_unowned_store_can_be_taken_over(tmp_path):
    """No live owner: the bump IS the takeover (the gateway's own boot, and every test store)."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        epoch = rt.begin_runtime_epoch(db, instance_id="first-owner")
        assert epoch == 1
        replacement = rt.begin_runtime_epoch(db, instance_id="replacement")
        assert replacement > epoch
    finally:
        db.close()


def test_a_dead_owners_record_does_not_fence_the_bump(tmp_path):
    """A stale lock record (the gateway died) names nobody: the store can be taken over."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        first = rt.begin_runtime_epoch(db, instance_id="owner")
        _write_lock_record(tmp_path, 999_999_999)  # a pid that cannot be alive
        replacement = rt.begin_runtime_epoch(db, instance_id="replacement")
        assert replacement > first
    finally:
        db.close()
