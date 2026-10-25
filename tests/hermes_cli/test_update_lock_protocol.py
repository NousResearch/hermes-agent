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
@pytest.mark.live_system_guard_bypass  # SIGCONTs only its own reader child; without psutil the guard cannot prove that
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


_RECLAIM_PAUSED_READER = """
import os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_lock
def trace(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "_reclaim_dead":
        sys.settrace(None)
        os.kill(os.getpid(), signal.SIGSTOP)   # stale snapshot taken; a scheduling pause, nothing replaced
    return None
sys.settrace(trace)
print(update_lock.read_live_update(path=Path(sys.argv[2]), install_root=Path(sys.argv[3])) is not None, flush=True)
"""

_MARKER_ONLY_CLAIMANT = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
assert UpdateLock(path=Path(sys.argv[2]), install_root=Path(sys.argv[3]), checkout_first=False).acquire()
Path(sys.argv[4]).touch()
time.sleep(60)
"""


@posix_only
@pytest.mark.live_system_guard_bypass  # SIGCONTs only its own reader child; without psutil the guard cannot prove that
def test_a_reader_reports_the_live_claim_that_replaced_its_stale_snapshot(tmp_path):
    """F2: the reader read a dead marker and paused before the locked recheck; a real marker-only
    claimant (no checkout lease, as the launch hand-off does) replaced it meanwhile. The recheck
    kept the replacement but the reader still answered "no update" — a skew-retirement vote."""
    home, install = tmp_path / "home", tmp_path / "install"
    home.mkdir()
    install.mkdir()
    marker = home / ".hermes-update-in-progress"
    _dead_marker(marker)
    reader = _python(tmp_path, _RECLAIM_PAUSED_READER, marker, install, stdout=subprocess.PIPE, text=True)
    claimant = None
    try:
        _pid, status = os.waitpid(reader.pid, os.WUNTRACED)
        assert os.WIFSTOPPED(status)
        claimant = _python(tmp_path, _MARKER_ONLY_CLAIMANT, marker, install, tmp_path / "claimed")
        _wait_for(tmp_path / "claimed", claimant)
        claim = marker.read_bytes()
        os.kill(reader.pid, signal.SIGCONT)
        out, _err = reader.communicate(timeout=30)
        assert reader.returncode == 0
        assert out.strip() == "True", "the reader reported a live replacement claim as no update"
        assert marker.read_bytes() == claim, "the live replacement claim was not preserved byte-for-byte"
        from hermes_cli.update_lock import checkout_lock_held
        # update_in_progress() is this answer OR the checkout lease, which a marker-only claim lacks.
        assert checkout_lock_held(install) is False, "the claim is marker-only: no checkout lease answers for it"
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
_git_run(["git"], ["add", "tracked"], cwd=root, check=True)
"""


def _git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, env=env).strip()


@posix_only
def test_git_keeps_checkout_custody_after_its_updater_is_killed(tmp_path):
    """R2: `_git_run`'s git (and any program it runs) inherits the checkout lock. Pre-fix the
    killed updater's lock fell free while its git still wrote the index, and a contender acquired.
    A clean filter stalls git: repository hooks never run under the updater (F1)."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "probe")
    _git(root, "config", "user.email", "probe@invalid.local")
    (root / "tracked").write_text("before\n")
    _git(root, "add", "tracked")
    _git(root, "commit", "-qm", "before")
    stall = tmp_path / "stall.py"
    stall.write_text(f"import os, sys, time\nfrom pathlib import Path\nr = Path({str(tmp_path)!r})\n"
                     "(r / 'stalled').write_text(str(os.getpid()))\n"
                     "while not (r / 'go').exists(): time.sleep(0.02)\n"
                     "sys.stdout.buffer.write(sys.stdin.buffer.read())\n")
    _git(root, "config", "filter.stall.clean", f"'{sys.executable}' '{stall}'")
    (root / ".git/info/attributes").write_text("tracked filter=stall\n")
    (root / "tracked").write_text("after\n")
    owner = _python(tmp_path, _GIT_OWNER, root, env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
                                           "GIT_CONFIG_GLOBAL": os.devnull})
    contender = UpdateLock(path=tmp_path / "contender-marker", install_root=root)
    try:
        stalled_pid = int(_wait_for(tmp_path / "stalled", owner))
        owner.kill()
        owner.wait()
        assert _alive(stalled_pid)
        assert contender.acquire() is False, "a contender owns the checkout while git still writes"
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


# --- R6: a dead marker is never reclaimed while the checkout lock is held ---------------------

_LOCK_HOLDER = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_lock
# A killed updater's completion child: holds the checkout lock, never wrote the marker.
assert update_lock._acquire_checkout(Path(sys.argv[2])) is None
Path(sys.argv[3]).write_text("held", encoding="utf-8")
time.sleep(120)
"""


def test_dead_marker_is_kept_and_reported_held_while_the_checkout_lock_is_held(tmp_path):
    from hermes_cli import update_lock

    install = tmp_path / "install"
    install.mkdir()
    marker = tmp_path / ".hermes-update-in-progress"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    marker.write_text(f"{dead.pid}\n{int(time.time())}\nct:1.5\n", encoding="utf-8")
    holder = _python(tmp_path, _LOCK_HOLDER, install, tmp_path / "ready")
    try:
        _wait_for(tmp_path / "ready", holder)
        verdict = update_lock.read_live_update(path=marker, install_root=install)
        assert marker.exists(), "a reader deleted the dead marker while the update tree still held the checkout"
        assert verdict is not None and verdict.held, f"expected the hand-off scripts' 'held', got {verdict!r}"
    finally:
        holder.kill()
        holder.wait()
    assert update_lock.read_live_update(path=marker, install_root=install) is None
    assert not marker.exists(), "with the tree gone the dead marker is reclaimed"


def _dead_marker(marker: Path) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    marker.write_text(f"{dead.pid}\n{int(time.time())}\nct:1.5\n", encoding="utf-8")


def test_a_claimer_never_reclaims_a_dead_marker_while_the_checkout_lock_is_held(tmp_path):
    """M1 (round 7): the launch path claims the marker before it takes the checkout lock
    (venv_sync). Its claim step deleted the dead marker over a held lock, then dropped its own
    claim when the checkout refused: the marker was gone while the killed update's tree ran."""
    from hermes_cli import update_lock

    install = tmp_path / "install"
    install.mkdir()
    marker = tmp_path / ".hermes-update-in-progress"
    _dead_marker(marker)
    dead_bytes = marker.read_bytes()
    holder = _python(tmp_path, _LOCK_HOLDER, install, tmp_path / "ready")
    try:
        _wait_for(tmp_path / "ready", holder)
        claim = UpdateLock(path=marker, install_root=install)
        assert claim._claim_marker() is False, "claimed over a dead marker while the checkout lock is held"
        assert marker.read_bytes() == dead_bytes, "the claim step replaced the held update's marker"
        assert claim.holder is not None and claim.holder.held
        launch = UpdateLock(path=marker, install_root=install, checkout_first=False)  # venv_sync
        assert launch.acquire() is False
        launch.release()
        assert launch.holder is not None and launch.holder.held and launch.holder.pid == 0
        assert marker.read_bytes() == dead_bytes
    finally:
        holder.kill()
        holder.wait()
    with UpdateLock(path=marker, install_root=install, checkout_first=False) as launch:
        assert launch.acquired, "with the tree gone the dead marker is reclaimed and claimed"
    assert not marker.exists()
    assert update_lock.checkout_lock_held(install) is False


_ORPHANING_HOLDER = """
import subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_lock
root = Path(sys.argv[2])
assert update_lock._acquire_checkout(root) is None   # the record names this (soon dead) updater
release = Path(sys.argv[3]).with_name("release")
wait = "import sys, time, pathlib\\nwhile not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.05)"
child = subprocess.Popen([sys.executable, "-c", wait, str(release)],
                         pass_fds=update_lock.checkout_lock_fds(root))
Path(sys.argv[3]).write_text(str(child.pid), encoding="utf-8")
"""


@posix_only
def test_a_refusal_over_a_dead_updaters_lock_uses_the_held_wording(tmp_path):
    """m6: `hermes update`, launch repair and source completion refuse through the lock's
    holder record. Its updater is dead and a child it started holds the lock: say so, never
    name the dead process."""
    from hermes_cli import update_lock

    install = tmp_path / "install"
    install.mkdir()
    updater = _python(tmp_path, _ORPHANING_HOLDER, install, tmp_path / "child")
    updater.wait(timeout=30)
    dead_pid = updater.pid
    _wait_for(tmp_path / "child")
    try:
        lock = UpdateLock(path=tmp_path / ".hermes-update-in-progress", install_root=install)
        assert lock.acquire() is False
        assert lock.holder is not None and lock.holder.held and lock.holder.pid == 0
        text = update_lock.describe_holder(lock.holder)
        assert f"process {dead_pid}" not in text
        assert "still holds the checkout" in text
    finally:
        (tmp_path / "release").touch()  # the orphaned child is not ours to signal


# --- D17: a "held?" probe never makes a concurrent update fail --------------------------------

_PROBER = """
import fcntl, os, sys, time
from pathlib import Path
path = sys.argv[2]
end = time.monotonic() + float(sys.argv[3])
Path(sys.argv[4]).write_text("go", encoding="utf-8")
n = 0
while time.monotonic() < end:
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        time.sleep(0.0001)  # held for the instant of one `flock -n` probe
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(fd)
    n += 1
    time.sleep(0.001)  # the next probe's process spawn
print(n)
"""


@posix_only
def test_held_probe_never_fails_a_concurrent_acquire(tmp_path):
    from hermes_cli import update_lock

    install = tmp_path / "install"
    install.mkdir()
    lock_path = update_lock.checkout_lock_path(install)
    lock_path.touch()
    prober = _python(tmp_path, _PROBER, lock_path, 4, tmp_path / "go", stdout=subprocess.PIPE, text=True)
    _wait_for(tmp_path / "go", prober)
    attempts = refused = 0
    try:
        while prober.poll() is None:
            attempts += 1
            if update_lock._acquire_checkout(install) is not None:
                refused += 1
            else:
                update_lock._release_checkout()
    finally:
        prober.kill()
        prober.wait()
    assert attempts > 50
    assert refused == 0, f"{refused}/{attempts} acquires failed against a probe with no real holder"


# --- oversized numeric marker fields never raise ---------------------------------------------

def test_oversized_marker_numbers_never_raise_and_are_reclaimed(tmp_path):
    from hermes_cli import update_lock

    install = tmp_path / "install"
    install.mkdir()
    marker = tmp_path / ".hermes-update-in-progress"
    for text in ("1" * 5000 + "\n1\nct:1.5\n", "4242\n" + "9" * 5000 + "\nct:1.5\n",
                 "4242\n1\nct:1.5\ndelegate:" + "7" * 5000 + " ct:1.5\n"):
        marker.write_text(text, encoding="utf-8")
        assert update_lock.read_live_update(path=marker, install_root=install) is None
        marker.write_text(text, encoding="utf-8")
        lock = UpdateLock(path=marker, install_root=install)
        assert lock.acquire() is True, lock.holder
        lock.release()
