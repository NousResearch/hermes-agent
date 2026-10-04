"""Windows live cells for the checkout lock (contract C1.7): msvcrt byte lock + kill-on-close job.

Invariant under test: the checkout lock is free => no process of the update tree is alive.
On Windows a child cannot inherit an msvcrt lock, so the owner binds every update-tree child
into a kill-on-close job: killing the owner (taskkill /F) kills the child and frees the lock.
"""

from __future__ import annotations

import contextlib
import os
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


# --- R2 on Windows: the updater's git and Node children join the owner's kill-on-close job -----

_GIT_OWNER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
from hermes_cli.update_cmd import _git_run
root = Path(sys.argv[2])
assert UpdateLock(path=Path(sys.argv[3]), install_root=root).acquire()
_git_run(["git"], ["stash", "push", "-m", "custody"], cwd=root)
"""

_BUILD_OWNER = r"""
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
from hermes_cli.source_build import run_source_script
root = Path(sys.argv[2])
assert UpdateLock(path=Path(sys.argv[3]), install_root=root).acquire()
env = {**os.environ, "PATH": sys.argv[4] + os.pathsep + os.environ["PATH"]}
run_source_script(root, "build.mjs", env=env, label="probe build")
"""


def _blocker(tmp_path: Path) -> str:
    """Python source that records its pid and blocks until ``go`` exists."""
    pid_file, go = (tmp_path / "blocker.pid").as_posix(), (tmp_path / "go").as_posix()
    return (f"import os, time\nopen({pid_file!r}, 'w').write(str(os.getpid()))\n"
            f"while not os.path.exists({go!r}): time.sleep(0.05)\nprint('cleaned')\n")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"})


def _tree_dies_with_its_owner(tmp_path: Path, install: Path, owner_args: list[str]) -> None:
    import psutil

    owner = subprocess.Popen([sys.executable, "-c", *owner_args], stdin=subprocess.DEVNULL)
    pid_file, tree = tmp_path / "blocker.pid", []
    try:
        deadline = time.time() + 60
        while not (pid_file.exists() and pid_file.read_text(encoding="utf-8-sig").strip()):
            assert owner.poll() is None, f"owner exited {owner.returncode} before its child blocked"
            assert time.time() < deadline, "the blocking child never started"
            time.sleep(0.1)
        blocker = psutil.Process(int(pid_file.read_text(encoding="utf-8-sig")))
        tree = [blocker, *(p for p in blocker.parents() if p.pid != owner.pid and owner.pid in
                           {q.pid for q in p.parents()})]
        assert UpdateLock(path=tmp_path / "other-marker", install_root=install).acquire() is False

        subprocess.run(["taskkill", "/F", "/PID", str(owner.pid)], capture_output=True, check=False)
        owner.wait(timeout=30)
        deadline = time.time() + 15
        while any(p.is_running() for p in tree) and time.time() < deadline:
            time.sleep(0.2)
        survivors = [f"{p.pid}:{p.name()}" for p in tree if p.is_running()]
        assert not survivors, f"update children outlived their killed owner: {survivors}"
        fresh = UpdateLock(path=tmp_path / "other-marker", install_root=install)
        assert fresh.acquire() is True, "the checkout stayed locked after the whole tree died"
        fresh.release()
    finally:
        (tmp_path / "go").touch()
        for proc in tree:
            with contextlib.suppress(Exception):
                proc.kill()
        if owner.poll() is None:
            owner.kill()


def test_killed_owner_takes_its_git_child_down(tmp_path):
    exe = Path(sys.executable).as_posix()
    if any(ch in exe for ch in " =~%#'\"&;|<>()$`*?["):
        pytest.skip("the clean filter must run without a shell: interpreter path has shell metacharacters")
    install = tmp_path / "checkout"
    install.mkdir()
    _git(install, "init", "-q")
    _git(install, "config", "user.name", "probe")
    _git(install, "config", "user.email", "probe@invalid.local")
    (install / "f.txt").write_text("before\n", encoding="utf-8")
    _git(install, "add", "f.txt")
    _git(install, "commit", "-qm", "before")
    # The clean filter is the bare interpreter: it runs the file's content (stdin) as a program.
    _git(install, "config", "filter.block.clean", exe)
    (install / ".git" / "info" / "attributes").write_text("f.txt filter=block\n", encoding="utf-8")
    (install / "f.txt").write_text(_blocker(tmp_path), encoding="utf-8")
    _tree_dies_with_its_owner(tmp_path, install, [_GIT_OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m")])


def test_killed_owner_takes_its_node_build_down(tmp_path):
    install = tmp_path / "checkout"
    install.mkdir()
    (install / "build.py").write_text(_blocker(tmp_path), encoding="utf-8")
    (install / "build.mjs").write_text("// stands in for a build script\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "node.bat").write_text(f'@"{sys.executable}" "{install / "build.py"}"\r\n', encoding="utf-8")
    _tree_dies_with_its_owner(tmp_path, install,
                              [_BUILD_OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m"), str(bin_dir)])
