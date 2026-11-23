"""Breaking an orphaned storage flock must not take a live successor's lock with it.

``flock`` rides the open file description, which ``fork()`` duplicates: a holder that forks a child
and then dies leaves the lock held by that child (#100108). ``_acquire_db_flock`` breaks such a lock
once the recorded holder is provably dead, by unlinking the lock file and retaking it on a fresh
inode.

Two contenders that had opened the orphaned file before the break and timed out at different moments
used to end up both inside. The first breaker's new lock file was unlinked by name under the second
one, whose ``flock`` then guarded a file nothing else could see.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from persistence.sqlite._state import common as state_common

_REPO = Path(__file__).resolve().parents[3]

# A real orphan: the script takes the lock, records a holder that has already exited, forks a child
# that inherits the lock fd, and drops its own fd. The child keeps the flock until the keep-file
# goes away, so the test releases it without signalling a process outside its own subtree.
_ORPHAN_SCRIPT = """
import fcntl, json, os, sys, time
sys.path.insert(0, sys.argv[1])
from persistence.sqlite._state import common
from process_identity import pid_namespace_id
lock_path, keep = sys.argv[2], sys.argv[3]
handle = open(lock_path, "a+b")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
gone = os.fork()
if gone == 0:
    os._exit(0)
os.waitpid(gone, 0)
record = {"pid": gone, "pidns": pid_namespace_id(), "start_ticks": common._proc_start_ticks(os.getpid()),
          "acquired_at": time.time()}
common._rewrite_lock_file(handle, json.dumps(record, sort_keys=True).encode("utf-8"))
holder = os.fork()
if holder == 0:
    os.close(sys.stdout.fileno())
    sys.stderr.close()
    while os.path.exists(keep):
        time.sleep(0.05)
    os._exit(0)
handle.close()
print("orphaned", flush=True)
os.waitpid(holder, 0)
"""


@pytest.fixture()
def orphaned_lock(tmp_path):
    lock_path, keep = tmp_path / "state.db.test.lock", tmp_path / "keep"
    keep.touch()
    proc = subprocess.Popen(
        [sys.executable, "-c", _ORPHAN_SCRIPT, str(_REPO), str(lock_path), str(keep)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "orphaned"
        yield lock_path
    finally:
        keep.unlink()
        proc.wait(timeout=30)


def _contend(lock_path, timeout_seconds):
    """One contender on a handle opened now; the lock is held until the test releases it."""
    handle = open(lock_path, "a+b")
    outcome = {}

    def run():
        outcome["result"] = state_common._acquire_db_flock(
            lock_path, handle, timeout_seconds, poll_seconds=0.02, description="test lock")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


@pytest.mark.platforms("posix")
def test_a_later_contender_does_not_unlink_the_first_breakers_live_lock(orphaned_lock):
    # Both contenders open the orphaned file before either breaks it, and time out a second apart.
    first, first_outcome = _contend(orphaned_lock, timeout_seconds=0.5)
    later, later_outcome = _contend(orphaned_lock, timeout_seconds=1.5)
    first.join(timeout=60)
    later.join(timeout=60)
    results = [first_outcome["result"], later_outcome["result"]]
    try:
        acquired = [result[0] for result in results]
        assert None not in acquired, results
        # The later contender finds the lock retaken by the first and defers, as for any live holder.
        assert acquired.count(True) == 1, acquired
        # Both handles name the one live lock file: nobody guards an unlinked inode.
        inodes = {os.fstat(result[1].fileno()).st_ino for result in results}
        assert inodes == {os.stat(orphaned_lock).st_ino}
    finally:
        for result in results:
            result[1].close()
