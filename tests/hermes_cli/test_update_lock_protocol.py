"""Marker mutation protocol (A7) and checkout custody (R2/R11) with real processes and files.

Each test reproduces a review finding on the pre-A7 head: a stale reader deleting a claim
published after its read (R3), a hand-off owner's killed claim surviving its orphaned delegate
(A7 rule 5, the native Windows ``orphaned_update`` artifact), git / the Node build writing a
checkout a contender already owns (R2), a completion from another home entering its build
while the checkout is owned (R2), and a live pre-root-marker profile claim going unseen (R11).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hermes_cli.update_lock import UpdateLock, marker_mutex_path, process_create_time

REPO_ROOT = Path(__file__).resolve().parents[2]
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals / fd inheritance")


def _script(tmp_path: Path, code: str) -> Path:
    path = tmp_path / f"child_{abs(hash(code))}.py"
    path.write_text(textwrap.dedent(code), encoding="utf-8")
    return path


def _python(tmp_path: Path, code: str, *args, **kwargs) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(_script(tmp_path, code)), str(REPO_ROOT), *map(str, args)],
                            stdin=subprocess.DEVNULL, **kwargs)


def _wait_for(path: Path, proc: subprocess.Popen | None = None, timeout: float = 30.0) -> str:
    end = time.monotonic() + timeout
    while not path.exists():
        assert proc is None or proc.poll() is None, f"child exited {proc.returncode} before {path.name}"
        assert time.monotonic() < end, f"timed out waiting for {path}"
        time.sleep(0.02)
    return path.read_text(encoding="utf-8")


def _alive(pid: int) -> bool:
    from hermes_cli.update_lock import _pid_alive

    return _pid_alive(pid)


_STALE_READER = """
import linecache, os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_lock
src = update_lock.__file__
def trace(frame, event, arg):
    if event == "line" and frame.f_code.co_filename == src \\
            and ".unlink()" in linecache.getline(src, frame.f_lineno):
        sys.settrace(None)
        os.kill(os.getpid(), signal.SIGSTOP)   # a scheduling pause, nothing replaced
    return trace
sys.settrace(trace)
update_lock.read_live_update(path=Path(sys.argv[2]))
"""

_CLAIMANT = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
assert UpdateLock(path=Path(sys.argv[2])).acquire()
Path(sys.argv[3]).touch()
time.sleep(60)
"""


@posix_only
def test_a_stale_reclaim_never_deletes_a_claim_published_after_its_read(tmp_path):
    """R3: the reader judged a dead marker, then paused right at its unlink. A second real
    process claims meanwhile. Pre-A7 the resumed reader deleted that live claim; now the
    decision and the unlink share one hold of the marker mutex, so the claimant waits."""
    marker = tmp_path / ".hermes-update-in-progress"
    marker.write_bytes(b"0\n0\n")  # malformed: dead
    reader = _python(tmp_path, _STALE_READER, marker)
    claimant = None
    try:
        _pid, status = os.waitpid(reader.pid, os.WUNTRACED)
        assert os.WIFSTOPPED(status)
        claimant = _python(tmp_path, _CLAIMANT, marker, tmp_path / "claimed")
        time.sleep(1.0)
        os.kill(reader.pid, signal.SIGCONT)
        assert reader.wait(timeout=30) == 0
        _wait_for(tmp_path / "claimed", claimant)
        assert marker.read_text(encoding="utf-8").startswith(f"{claimant.pid}\n"), \
            "a stale reader deleted a live claim published after its read"
        assert marker_mutex_path(marker).exists(), "the sidecar is never deleted"
    finally:
        if reader.poll() is None:
            os.kill(reader.pid, signal.SIGCONT)
            reader.kill()
        reader.wait()
        if claimant is not None:
            claimant.kill()
            claimant.wait()


_MUTEX_HOLDER = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import marker_mutex
with marker_mutex(Path(sys.argv[2])):
    Path(sys.argv[3]).touch()
    time.sleep(1.5)
"""


def test_marker_mutations_wait_for_the_mutex_another_process_holds(tmp_path):
    """A7 rule 1 on every OS (POSIX flock / Windows share-none open of ``<marker>.lock``):
    while another process holds the marker mutex, a reclaim of even a dead marker waits."""
    marker = tmp_path / ".hermes-update-in-progress"
    marker.write_bytes(b"0\n0\n")
    holder = _python(tmp_path, _MUTEX_HOLDER, marker, tmp_path / "held")
    try:
        _wait_for(tmp_path / "held", holder)
        started = time.monotonic()
        lock = UpdateLock(path=marker)
        assert lock.acquire() and lock.acquired
        assert time.monotonic() - started >= 0.9, "the reclaim ran inside another process's hold"
        lock.release()
        assert marker_mutex_path(marker).exists(), "the sidecar is never deleted"
    finally:
        holder.wait(timeout=30)


_ORPHAN_UPDATE = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
base = Path(sys.argv[2])
while not (base / "delegate-written").exists(): time.sleep(0.02)
lock = UpdateLock(path=base / "marker")
assert lock.acquire(), lock.holder
(base / "ready").touch()
while not (base / "go").exists(): time.sleep(0.02)
lock.release()
"""

_HANDOFF_OWNER = """
import os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import process_create_time
base = Path(sys.argv[2])
marker = base / "marker"
marker.write_text(f"{os.getpid()}\\n{int(time.time())}\\nct:{process_create_time():.3f}\\n")
child = subprocess.Popen([sys.executable, sys.argv[3], sys.argv[1], str(base)], start_new_session=True)
with open(marker, "a") as fh:   # the hand-off scripts name their update child right after spawn
    fh.write(f"delegate:{child.pid} ct:{process_create_time(child.pid):.3f}\\n")
(base / "delegate-written").write_text(str(child.pid))
child.wait()
"""


@posix_only
def test_an_orphaned_delegate_removes_its_killed_owners_marker(tmp_path):
    """A7 rule 5 / Windows ``orphaned_update``: the hand-off script claims, names its `hermes
    update` child as delegate, and is killed alone. Pre-A7 the child adopted the claim without
    writing it, so its exit released nothing: '<dead owner> … delegate:<exited child>' survived
    (native artifact crash-12db). The delegate's release now removes a claim whose owner died."""
    owner = _python(tmp_path, _HANDOFF_OWNER, tmp_path, _script(tmp_path, _ORPHAN_UPDATE))
    try:
        child = int(_wait_for(tmp_path / "delegate-written", owner))
        _wait_for(tmp_path / "ready", owner)
        owner.kill()
        owner.wait()
        (tmp_path / "go").touch()
        end = time.monotonic() + 30
        while _alive(child) and time.monotonic() < end:
            time.sleep(0.05)
        assert not _alive(child)
        assert not (tmp_path / "marker").exists(), "the orphaned update's exit left its dead claim behind"
    finally:
        owner.kill()
        owner.wait()


@posix_only
def test_owner_release_hands_the_claim_to_a_live_delegate(tmp_path):
    """A7 rule 5, the other half: the owner's release never deletes the claim a live delegate
    still runs under; it rewrites it with that delegate as owner (and keeps the run line)."""
    marker = tmp_path / "marker"
    delegate = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL)
    try:
        lock = UpdateLock(path=marker)
        assert lock.acquire() and lock.acquired
        ct = f"{process_create_time(delegate.pid):.3f}"
        marker.write_bytes(marker.read_bytes() + f"delegate:{delegate.pid} ct:{ct}\nrun:desk-1\n".encode())
        started = marker.read_text(encoding="utf-8").splitlines()[1]
        lock.release()
        assert marker.read_text(encoding="utf-8") == f"{delegate.pid}\n{started}\nct:{ct}\nrun:desk-1\n"
    finally:
        delegate.kill()
        delegate.wait()


_GIT_OWNER = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
from hermes_cli.update_cmd import _git_run
root = Path(sys.argv[2])
assert UpdateLock(path=root / "owner-marker", install_root=root).acquire()
_git_run(["git"], ["commit", "-qm", "after"], cwd=root, check=True)
"""


def _git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, env=env).strip()


@posix_only
def test_git_keeps_checkout_custody_after_its_updater_is_killed(tmp_path):
    """R2: `_git_run`'s git (and the hook it runs) inherits the checkout lock. Pre-fix the
    killed updater's lock fell free while its git still committed, and a contender acquired."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "probe")
    _git(root, "config", "user.email", "probe@invalid.local")
    (root / "tracked").write_text("before\n")
    _git(root, "add", "tracked")
    _git(root, "commit", "-qm", "before")
    (root / "tracked").write_text("after\n")
    _git(root, "add", "tracked")
    hook = root / ".git/hooks/pre-commit"
    hook.write_text(f"#!{sys.executable}\nimport os, time\nfrom pathlib import Path\nr = Path({str(tmp_path)!r})\n"
                    "(r / 'hook').write_text(str(os.getpid()))\n"
                    "while not (r / 'go').exists(): time.sleep(0.02)\n")
    hook.chmod(0o700)
    owner = _python(tmp_path, _GIT_OWNER, root, env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
                                           "GIT_CONFIG_GLOBAL": os.devnull})
    contender = UpdateLock(path=tmp_path / "contender-marker", install_root=root)
    try:
        hook_pid = int(_wait_for(tmp_path / "hook", owner))
        owner.kill()
        owner.wait()
        assert _alive(hook_pid)
        assert contender.acquire() is False, "a contender owns the checkout while git still commits"
        (tmp_path / "go").touch()
        end = time.monotonic() + 30
        while not contender.acquire():
            assert time.monotonic() < end, "the checkout stayed locked after git exited"
            time.sleep(0.05)
    finally:
        (tmp_path / "go").touch()
        contender.release()
        owner.kill()
        owner.wait()


_BUILD_OWNER = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
from hermes_cli.source_build import run_source_script
root = Path(sys.argv[2])
assert UpdateLock(path=root / "owner-marker", install_root=root).acquire()
run_source_script(root, "build.mjs", env={**os.environ, "PATH": sys.argv[3]}, label="probe build")
"""


@posix_only
def test_the_source_build_keeps_checkout_custody_after_its_updater_is_killed(tmp_path):
    """R2: the Node build `run_source_script` starts keeps the lock fd. A real `node` stand-in
    (an executable on the build's PATH) outlives the killed updater; the checkout stays locked."""
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    node = bin_dir / "node"
    node.write_text(f"#!{sys.executable}\nimport os, time\nfrom pathlib import Path\nr = Path({str(tmp_path)!r})\n"
                    "(r / 'node').write_text(str(os.getpid()))\n"
                    "while not (r / 'go').exists(): time.sleep(0.02)\n")
    node.chmod(0o700)
    owner = _python(tmp_path, _BUILD_OWNER, root, bin_dir)
    contender = UpdateLock(path=tmp_path / "contender-marker", install_root=root)
    try:
        node_pid = int(_wait_for(tmp_path / "node", owner))
        owner.kill()
        owner.wait()
        assert _alive(node_pid)
        assert contender.acquire() is False, "a contender owns the checkout while the build still writes"
    finally:
        (tmp_path / "go").touch()
        contender.release()
        owner.kill()
        owner.wait()


_HOLDER = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
assert UpdateLock(path=Path(sys.argv[3]), install_root=sys.argv[2]).acquire()
print("held", flush=True)
time.sleep(120)
"""


def test_a_completion_from_another_home_never_builds_an_owned_checkout(tmp_path, monkeypatch):
    """R2 admission: `complete_source_checkout` ACQUIRES the checkout lock. Pre-fix it claimed
    only its own home's marker and entered the build body beside the owning update."""
    from hermes_cli import source_completion

    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-home"))
    entered = []
    monkeypatch.setattr(source_completion, "_complete_locked", lambda *a, **k: entered.append(a))
    holder = subprocess.Popen([sys.executable, str(_script(tmp_path, _HOLDER)), str(REPO_ROOT), str(root),
                               str(tmp_path / "owner-marker")],
                              stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, encoding="utf-8")
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        with pytest.raises(RuntimeError, match="an update is still running"):
            source_completion.complete_source_checkout(root, desktop=False, assume_yes=True)
        assert entered == [], "the completion body ran while another update owned the checkout"
    finally:
        holder.kill()
        holder.wait()


def test_a_live_legacy_profile_claim_is_honored(tmp_path, monkeypatch):
    """R11: before the root marker, an update under a named profile claimed
    ``<root>/profiles/<p>/.hermes-update-in-progress`` (v1) and never took the checkout lock.
    While such a claim is live, a new update from that root refuses."""
    root = tmp_path / "root"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    legacy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL)
    try:
        (profile / ".hermes-update-in-progress").write_text(f"{legacy.pid}\n{int(time.time())}\n")
        lock = UpdateLock(path=root / ".hermes-update-in-progress")
        assert lock.acquire() is False
        assert lock.holder is not None and lock.holder.pid == legacy.pid
        assert not (root / ".hermes-update-in-progress").exists()
    finally:
        legacy.kill()
        legacy.wait()
    # Dead: no longer honored (and left for the old updater that wrote it).
    lock = UpdateLock(path=root / ".hermes-update-in-progress")
    assert lock.acquire() is True
    lock.release()
