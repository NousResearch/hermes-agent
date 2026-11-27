"""An epoch bump through the backend is a takeover: it must not fence out a live owner (#142330).

The runtime epoch is the fence every write is checked against, and the owner captures it once at boot. A bump by a
process that does not own the profile therefore leaves the live owner permanently behind the fence, so
``runtime.begin_epoch`` refuses with ``ownership_conflict`` while another process holds the profile's
``gateway.lock``. The reference backend has no profile home and never refuses; these cases are the SQLite backend's.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite.conftest import profile_home, store_for

_REPO_ROOT = Path(__file__).resolve().parents[3]

_HOLDER = """
import sys
from pathlib import Path
from gateway.runtime_ownership import ProfileOwnership
ownership = ProfileOwnership()
ownership.reserve([Path(sys.argv[1])])
print('ready', flush=True)
sys.stdin.read()  # held until the fixture closes stdin (or the test process dies)
"""


@pytest.fixture()
def hold_profile():
    """Start a foreign process that really reserves a profile's ``gateway.lock`` (the gateway)."""
    procs = []

    def start(home):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(_REPO_ROOT), os.environ.get("PYTHONPATH")]))}
        proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(home)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, env=env, cwd=_REPO_ROOT)
        procs.append(proc)
        assert proc.stdout.readline().strip() == "ready"
        return proc.pid

    yield start
    for proc in procs:
        proc.stdin.close()
        proc.wait(timeout=10)


def _epoch_of(path: Path) -> int | None:
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()
    return row[0] if row else None


def _refused(store, instance_id: str) -> AuthorityRefusal:
    with pytest.raises(AuthorityRefusal) as exc:
        store.runtime.begin_epoch(instance_id=instance_id)
    assert exc.value.reason == "ownership_conflict"
    assert exc.value.field == "runtime_epoch"
    return exc.value


def test_a_stranger_cannot_bump_the_epoch_of_a_live_owners_store(backend, scope_a, hold_profile):
    store = backend.open(scope_a)
    path = store_for(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="owner")
    hold_profile(path.parent)

    _refused(store, "stranger")
    # The owner's fence still stands: the bump did not move.
    assert _epoch_of(path) == epoch


def test_an_unpublished_holder_still_fences_the_bump(backend, scope_a, hold_profile):
    """The flock decides, not the record: a holder whose record is torn or unreadable still owns it."""
    store = backend.open(scope_a)
    path = store_for(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="owner")
    hold_profile(path.parent)
    (path.parent / "gateway.lock").write_text("", encoding="utf-8")  # same inode, record gone

    _refused(store, "stranger")
    assert _epoch_of(path) == epoch


def test_a_record_naming_this_pid_in_another_pid_namespace_is_not_ours(backend, scope_a, hold_profile):
    """Same pid number, other PID namespace (a containerised holder): the flock decides, not the pid."""
    from gateway.status import _build_pid_record

    store = backend.open(scope_a)
    path = store_for(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="owner")
    hold_profile(path.parent)
    record = {**_build_pid_record(), "pidns": "another-namespace"}
    (path.parent / "gateway.lock").write_text(json.dumps(record), encoding="utf-8")

    _refused(store, "stranger")
    assert _epoch_of(path) == epoch


def test_our_released_reservations_record_does_not_grant_the_bump(backend, scope_a, hold_profile):
    """A record naming this process outlives our release: a foreign holder that took the flock since but has not
    rewritten it yet must still fence the bump."""
    from gateway.runtime_ownership import ProfileOwnership

    store = backend.open(scope_a)
    path = store_for(scope_a)
    ownership = ProfileOwnership()
    try:
        epoch = store.runtime.begin_epoch(instance_id="owner")
        ownership.reserve([path.parent])
        ours = (path.parent / "gateway.lock").read_text(encoding="utf-8")
        ownership.release(path.parent)
        hold_profile(path.parent)
        (path.parent / "gateway.lock").write_text(ours, encoding="utf-8")  # the holder has not published

        _refused(store, "stranger")
        assert _epoch_of(path) == epoch
    finally:
        ownership.close()


@pytest.mark.platforms("any")
def test_the_owner_itself_may_bump_the_epoch(backend, scope_a):
    """The holder of the profile lock is the one taking the store over: it is allowed."""
    from gateway.runtime_ownership import ProfileOwnership

    store = backend.open(scope_a)
    ownership = ProfileOwnership()
    try:
        first = store.runtime.begin_epoch(instance_id="owner")
        ownership.reserve([store_for(scope_a).parent])
        second = store.runtime.begin_epoch(instance_id="owner-again")
        assert second > first
    finally:
        ownership.close()


def test_the_verdict_is_per_profile(backend, scope_a, scope_b, hold_profile):
    """The owned profile refuses the stranger, and the other profile's own store is still free to be taken over."""
    hold_profile(store_for(scope_a).parent)
    owned, free = backend.open(scope_a), backend.open(scope_b)

    _refused(owned, "stranger")
    with sqlite3.connect(store_for(scope_a)) as conn:
        assert conn.execute("SELECT epoch FROM runtime_epoch").fetchall() == [], \
            "the refused bump must not have created the row"

    assert free.runtime.begin_epoch(instance_id="owner") == 1


def test_an_unowned_store_can_be_taken_over(backend, scope_a):
    """No live owner: the bump is the takeover (the gateway's own boot, and every test store)."""
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="first-owner")
    assert epoch == 1
    assert store.runtime.begin_epoch(instance_id="replacement") > epoch


def test_a_dead_owners_record_does_not_fence_the_bump(backend, scope_a):
    """A stale lock record (the gateway died, the flock went with it): the store can be taken over."""
    from gateway.status import _build_pid_record

    store = backend.open(scope_a)
    path = store_for(scope_a)
    first = store.runtime.begin_epoch(instance_id="owner")
    record = {**_build_pid_record(), "pid": 999_999_999, "start_time": None}
    (profile_home(scope_a.name) / "gateway.lock").write_text(json.dumps(record), encoding="utf-8")
    assert store.runtime.begin_epoch(instance_id="replacement") > first
    assert _epoch_of(path) == first + 1
