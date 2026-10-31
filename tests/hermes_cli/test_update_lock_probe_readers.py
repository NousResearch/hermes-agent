"""The update fence never reads another reader's probe of the checkout lock as a held lock.

A killed update always leaves a dead ``.hermes-update-in-progress`` marker, and while it is there
``gateway_runtime._update_fenced`` probes the checkout kernel lock on every ``ensure`` poll. The
gateway that ``ensure`` just started probes the same lock while it boots (its launch reclaims the
dead marker). A probe that took the lock exclusively turned that collision into "an update owns
the install": the first launch after the kill refused with ``Gateway draining: update_paused``.

Real processes and the real lock file; a real holder still fences (control).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import gateway_runtime, update_lock

REPO_ROOT = Path(__file__).resolve().parents[2]

_READER = r"""
import sys, threading
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_lock
root = sys.argv[2]
stop = threading.Event()
threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
print("ready", flush=True)
probes = 0
while not stop.is_set():
    update_lock.checkout_lock_held(root)
    probes += 1
print(probes, flush=True)
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
    readers = [_spawn(_READER, root) for _ in range(2)]
    try:
        for reader in readers:
            assert reader.stdout.readline().strip() == "ready"
        fenced = polls = 0
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            polls += 1
            fenced += gateway_runtime._update_fenced({home})
        # Both readers probed for the whole window: neither exited (or crashed) before the polls ended.
        assert [reader.poll() for reader in readers] == [None, None]
        for reader in readers:
            reader.stdin.write("stop\n")
            reader.stdin.flush()
        probes = [int(reader.stdout.readline()) for reader in readers]
        assert [reader.wait(timeout=30) for reader in readers] == [0, 0]
    finally:
        for reader in readers:
            if reader.poll() is None:
                reader.kill()
                reader.wait(timeout=30)
    assert min(probes) > 0
    assert fenced == 0, f"{fenced} of {polls} ensure polls read another reader's probe as an update"

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
