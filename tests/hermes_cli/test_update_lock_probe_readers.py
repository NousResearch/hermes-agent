"""The update fence never reads another reader's probe of the checkout lock as a held lock.

A killed update always leaves a dead ``.hermes-update-in-progress`` marker, and while it is there
``gateway_runtime._update_fenced`` probes the checkout kernel lock on every ``ensure`` poll. The
gateway that ``ensure`` just started probes the same lock while it boots (its launch reclaims the
dead marker). A probe that took the lock exclusively turned that collision into "an update owns
the install": the first launch after the kill refused with ``Gateway draining: update_paused``.

Real processes and the real lock file. Each reader is paused mid-probe, while it holds its lock,
so the collision is certain rather than raced; a real holder still fences (control).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import gateway_runtime, update_lock

REPO_ROOT = Path(__file__).resolve().parents[2]

_PAUSED_PROBE = r"""
import sys
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_lock
paused = []

def pause():
    if not paused:
        paused.append(True)
        print("probing", flush=True)
        sys.stdin.readline()

if sys.platform == "win32":
    # The Windows probe gives its shared lock straight back: stop between LockFileEx and UnlockFileEx.
    kernel32, overlapped = update_lock._windows_lockfile_api()

    class PauseBeforeUnlock:
        def __getattr__(self, name):
            return getattr(kernel32, name)

        def UnlockFileEx(self, *args):
            pause()
            return kernel32.UnlockFileEx(*args)

    update_lock._windows_lockfile_api = lambda: (PauseBeforeUnlock(), overlapped)
else:
    # POSIX keeps the shared flock until checkout_lock_held closes the descriptor.
    shared_probe = update_lock._try_shared_lock

    def probe(fd):
        taken = shared_probe(fd)
        pause()
        return taken

    update_lock._try_shared_lock = probe
held = update_lock.checkout_lock_held(sys.argv[2])
if not paused:  # never reached the shared probe: fail the handshake instead of faking it
    print("unprobed", flush=True)
    sys.exit(3)
print(held, flush=True)
"""

_HOLDER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_lock
assert update_lock._acquire_checkout(Path(sys.argv[2])) is None
print("held", flush=True)
sys.stdin.readline()
"""


def _spawn(code: str, root: Path, *args: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code, str(REPO_ROOT), str(root), *args],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=REPO_ROOT)


@pytest.fixture
def install(tmp_path, monkeypatch):
    """A checkout whose lock file exists, and a home holding a killed update's dead marker."""
    root = tmp_path / "install"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / update_lock.GIT_CHECKOUT_LOCK_NAME).write_bytes(b"")
    home = tmp_path / "home"
    home.mkdir()
    # pid 0 is never a live process: the marker a killed update leaves behind.
    (home / update_lock.MARKER_NAME).write_text(f"0\n{int(time.time())}\nct:1.000\n", encoding="utf-8")
    # _update_fenced probes the running checkout's lock (no install_root): make that this one.
    named = update_lock.checkout_lock_path
    monkeypatch.setattr(update_lock, "checkout_lock_path", lambda install_root=None: named(install_root or root))
    return root, home


def _assert_readers_never_fence(root: Path, home: Path) -> None:
    readers = []
    try:
        # One at a time: the second probes while the first holds its lock, and both hold while the
        # launch polls.
        for _ in range(2):
            readers.append(_spawn(_PAUSED_PROBE, root))
            assert readers[-1].stdout.readline().strip() == "probing"
        assert not gateway_runtime._update_fenced({home}), "a launch read another reader's probe as an update"
        for reader in readers:
            reader.stdin.write("go\n")
            reader.stdin.flush()
        assert [reader.stdout.readline().strip() for reader in readers] == ["False", "False"], \
            "a reader read the other reader's probe as a held lock"
        assert [reader.wait(timeout=30) for reader in readers] == [0, 0]
    finally:
        for reader in readers:
            if reader.poll() is None:
                reader.kill()
                reader.wait(timeout=30)

    holder = _spawn(_HOLDER, root)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert gateway_runtime._update_fenced({home}), "a live holder of the checkout lock must fence"
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)
    assert not gateway_runtime._update_fenced({home})


@pytest.mark.platforms("posix")
def test_concurrent_lock_probes_never_fence_a_launch_posix(install) -> None:
    _assert_readers_never_fence(*install)


@pytest.mark.platforms("windows")
def test_concurrent_lock_probes_never_fence_a_launch_windows(install) -> None:
    _assert_readers_never_fence(*install)


@pytest.mark.platforms("windows")
def test_repeated_windows_probes_build_no_new_ctypes_pointer_types(install, monkeypatch) -> None:
    """A long-lived gateway probes every 30 s. Before Python 3.14 ctypes caches every pointer type
    forever, so once the first probe has built its types, later probes must not build any."""
    import ctypes

    root, _home = install
    assert update_lock.checkout_lock_held(root) is False  # the first probe may build its types once
    real_pointer = ctypes.POINTER
    built = []
    monkeypatch.setattr(ctypes, "POINTER", lambda cls: built.append(cls) or real_pointer(cls))
    for _ in range(50):
        assert update_lock.checkout_lock_held(root) is False
    assert built == []
