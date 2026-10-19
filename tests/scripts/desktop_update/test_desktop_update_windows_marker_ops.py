"""Native Windows reclaim/busy commands and checkout-range declaration."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.scripts.desktop_update.windows_handoff_support import (
    MARKER_PS1, MARKER, _creation_time, _dead_pid, _op, _HeldLock,
)


@pytest.fixture
def sleeper():
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    yield proc
    proc.kill()
    proc.wait()


@pytest.mark.platforms('windows')
def test_marker_op_reclaim(tmp_path: Path, sleeper: subprocess.Popen) -> None:
    marker = tmp_path / MARKER
    assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'absent\n')
    marker.write_bytes(f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode())
    assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'reclaimed\n')
    assert not marker.exists()
    live = f'{sleeper.pid}\n{int(time.time())}\nct:{_creation_time(sleeper.pid)}\n'.encode()
    marker.write_bytes(live)
    assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, f'live {sleeper.pid}\n')
    assert marker.read_bytes() == live
    assert (tmp_path / (MARKER + '.lock')).exists()   # the sidecar is never deleted
    assert _op(tmp_path, '-MarkerOp', 'nonsense')[0] == 64
    assert _op(tmp_path, '-MarkerOp', 'withdraw')[0] == 64


@pytest.mark.platforms('windows')
def test_marker_op_reports_busy_while_the_lock_is_held(tmp_path: Path) -> None:
    dead = f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode()
    (tmp_path / MARKER).write_bytes(dead)
    lock = _HeldLock(tmp_path)
    try:
        assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'busy\n')
    finally:
        lock.release()
    assert (tmp_path / MARKER).read_bytes() == dead


def test_checkout_lock_probe_covers_the_owner_byte_and_every_lease_byte() -> None:
    """Runs everywhere (the live cells above need Windows): Test-CheckoutLockHeld must lock
    and unlock the same range hermes_cli/update_lock.py::_try_lock judges -- the owner byte at
    1 MiB plus the 16 R5b lease bytes after it -- or a leased survivor reads as a free checkout."""
    import re

    text = MARKER_PS1.read_text(encoding='utf-8-sig')
    body = text[text.index('function Test-CheckoutLockHeld'):]
    ranges = re.findall(r'\$fs\.(Lock|Unlock)\((\d+), (\d+)\)', body)
    assert ranges and {(int(o), int(n)) for _, o, n in ranges} == {(1 << 20, 1 + 16)}, ranges
    assert {kind for kind, _, _ in ranges} == {'Lock', 'Unlock'}
