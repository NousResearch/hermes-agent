"""Windows live cells for the checkout lock (contract C1.7): msvcrt byte lock + kill-on-close job.

Invariant under test: the checkout lock is free => no process of the update tree is alive.
On Windows a child cannot inherit an msvcrt lock, so the owner binds every update-tree child
into a kill-on-close job: killing the owner (taskkill /F) kills the child and frees the lock.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.update_lock import UpdateLock, update_in_progress

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[2]

_OWNER = r"""
import subprocess, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock, bind_child_to_update_tree
lock = UpdateLock(path=Path(sys.argv[3]), install_root=sys.argv[2])
assert lock.acquire()
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"],
                         creationflags=subprocess.CREATE_NO_WINDOW)
bind_child_to_update_tree(child)
print(child.pid, flush=True)
time.sleep(300)
"""


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], text=True, encoding="utf-8", errors="replace",
                         capture_output=True).stdout
    return str(pid) in out


def test_killed_owner_takes_its_tree_down_and_frees_the_lock(tmp_path):
    install = tmp_path / "checkout"
    install.mkdir()
    owner = subprocess.Popen([sys.executable, "-c", _OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m")],
                             stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, encoding="utf-8")
    try:
        child = int(owner.stdout.readline().strip())
        assert _alive(child)
        assert update_in_progress(install), "the owner's msvcrt lock is not visible"
        refused = UpdateLock(path=tmp_path / "other-home-marker", install_root=install)
        assert refused.acquire() is False, "a second update of one checkout was not refused"

        subprocess.run(["taskkill", "/F", "/PID", str(owner.pid)], capture_output=True, check=False)
        owner.wait(timeout=30)
        deadline = time.time() + 15
        while _alive(child) and time.time() < deadline:
            time.sleep(0.2)
        assert not _alive(child), "the update-tree child outlived its killed owner"
        assert not update_in_progress(install), "the lock stayed held after the whole tree died"
        fresh = UpdateLock(path=tmp_path / "other-home-marker", install_root=install)
        assert fresh.acquire() is True
        fresh.release()
    finally:
        if owner.poll() is None:
            owner.kill()
