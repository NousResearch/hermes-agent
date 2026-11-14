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
from pathlib import Path

import pytest

from hermes_state import SessionDB

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HOLDER = """
import sys, time
from pathlib import Path
from gateway.runtime_ownership import ProfileOwnership
ownership = ProfileOwnership()
ownership.reserve([Path(sys.argv[1])])
print('ready', flush=True)
time.sleep(120)
"""


@pytest.fixture()
def hold_profile():
    """Start a foreign process that really reserves a profile's ``gateway.lock`` (the gateway)."""
    procs = []

    def start(home):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(_REPO_ROOT), os.environ.get("PYTHONPATH")]))}
        proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(home)], stdout=subprocess.PIPE,
                                text=True, env=env, cwd=_REPO_ROOT)
        procs.append(proc)
        assert proc.stdout.readline().strip() == "ready"
        return proc.pid

    yield start
    for proc in procs:
        proc.kill()
        proc.wait(timeout=10)


def _epoch_of(db) -> int | None:
    row = db._conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()
    return row[0] if row else None


def test_a_stranger_cannot_bump_the_epoch_of_a_live_owners_store(tmp_path, hold_profile):
    """The reported failure: a process that does not own the profile must not fence the owner out."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        hold_profile(tmp_path)

        with pytest.raises(rt.RuntimeStoreError) as exc:
            rt.begin_runtime_epoch(db, instance_id="stranger")
        assert exc.value.reason == "ownership_conflict"

        # The owner's fence must still stand: the bump did not move.
        assert _epoch_of(db) == epoch
    finally:
        db.close()


def test_an_unpublished_holder_still_fences_the_bump(tmp_path, hold_profile):
    """The flock decides, not the record: a holder whose record is torn/unreadable still owns it."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        hold_profile(tmp_path)
        (tmp_path / "gateway.lock").write_text("", encoding="utf-8")  # same inode, record gone

        with pytest.raises(rt.RuntimeStoreError) as exc:
            rt.begin_runtime_epoch(db, instance_id="stranger")
        assert exc.value.reason == "ownership_conflict"
        assert _epoch_of(db) == epoch
    finally:
        db.close()


def test_the_owner_itself_may_bump_the_epoch(tmp_path):
    """The holder of the profile lock is the one taking the store over - it must be allowed."""
    import hermes_state_runtime as rt
    from gateway.runtime_ownership import ProfileOwnership

    db = SessionDB(db_path=tmp_path / "state.db")
    ownership = ProfileOwnership()
    try:
        first = rt.begin_runtime_epoch(db, instance_id="owner")
        ownership.reserve([tmp_path])
        second = rt.begin_runtime_epoch(db, instance_id="owner-again")
        assert second > first
    finally:
        ownership.close()
        db.close()


def test_the_verdict_is_per_profile(tmp_path, hold_profile):
    """A -> B: the owned profile refuses the stranger, and the other profile is untouched.

    The epoch row is a singleton of ITS store, so the refusal is about the profile that was
    asked - a second profile's own store is still free to be taken over."""
    import hermes_state_runtime as rt

    owned = tmp_path / "owned"
    free = tmp_path / "free"
    owned.mkdir()
    free.mkdir()
    hold_profile(owned)

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
    """A stale lock record (the gateway died, the flock went with it): the store can be taken over."""
    import hermes_state_runtime as rt
    from gateway.status import _build_pid_record

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        first = rt.begin_runtime_epoch(db, instance_id="owner")
        record = {**_build_pid_record(), "pid": 999_999_999, "start_time": None}
        (tmp_path / "gateway.lock").write_text(json.dumps(record), encoding="utf-8")
        replacement = rt.begin_runtime_epoch(db, instance_id="replacement")
        assert replacement > first
    finally:
        db.close()
