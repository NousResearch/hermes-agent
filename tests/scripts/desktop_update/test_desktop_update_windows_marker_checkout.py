"""Native checkout-owner and unopenable locks keep reclaim closed."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.scripts.desktop_update.windows_handoff_support import (
    MARKER, _dead_pid, _op, _CHECKOUT_HOLDER,
)


@pytest.mark.platforms('windows')
def test_marker_op_reclaim_reports_held_while_a_survivor_holds_the_checkout_lock(tmp_path: Path) -> None:
    """R6: the update died (dead marker) but a process it started still holds the checkout lock:
    the Desktop's reclaim must not free the marker; it answers `held` until that process exits."""
    install = tmp_path / 'hermes-agent'
    install.mkdir()
    dead = f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode()
    (tmp_path / MARKER).write_bytes(dead)
    release = tmp_path / 'release-holder'
    holder = subprocess.Popen([sys.executable, '-c', _CHECKOUT_HOLDER, str(install / '.hermes-update.lock'), str(release)])
    try:
        deadline = time.monotonic() + 30
        while not Path(str(release) + '.ready').exists():
            assert time.monotonic() < deadline and holder.poll() is None
            time.sleep(0.05)
        assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'held\n')
        assert (tmp_path / MARKER).read_bytes() == dead
    finally:
        release.touch()
        holder.wait(timeout=30)
    assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'reclaimed\n')
    assert not (tmp_path / MARKER).exists()


@pytest.mark.platforms('windows')
def test_marker_op_reclaim_without_a_marker_reports_held_while_the_checkout_lock_is_held(tmp_path: Path) -> None:
    """Review 5411223284: `absent` was answered before the checkout lock was looked at, so a
    Desktop gate opened while an update with no marker (yet, or any more) held the checkout."""
    install = tmp_path / 'hermes-agent'
    install.mkdir()
    release = tmp_path / 'release-holder'
    holder = subprocess.Popen([sys.executable, '-c', _CHECKOUT_HOLDER, str(install / '.hermes-update.lock'), str(release)])
    try:
        deadline = time.monotonic() + 30
        while not Path(str(release) + '.ready').exists():
            assert time.monotonic() < deadline and holder.poll() is None
            time.sleep(0.05)
        assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'held\n')
        assert not (tmp_path / MARKER).exists()
    finally:
        release.touch()
        holder.wait(timeout=30)
    assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'absent\n')


@pytest.mark.platforms('windows')
def test_marker_op_reclaim_reports_held_while_the_checkout_lock_cannot_be_opened(tmp_path: Path) -> None:
    """kshitij P2 (parity with marker.sh and the Desktop's probe): a lock file that exists but
    cannot be opened -- here another process holds it with no sharing -- read `not held`, so the
    reclaim deleted a dead marker the Desktop's own probe kept. It counts as held."""
    import ctypes

    install = tmp_path / 'hermes-agent'
    install.mkdir()
    lock = install / '.hermes-update.lock'
    lock.touch()
    dead = f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode()
    (tmp_path / MARKER).write_bytes(dead)
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    handle = k32.CreateFileW(str(lock), 0x80000000, 0, None, 3, 0x80, None)   # GENERIC_READ, no sharing
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
    try:
        assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'held\n')
        assert (tmp_path / MARKER).read_bytes() == dead
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))
    assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'reclaimed\n')
