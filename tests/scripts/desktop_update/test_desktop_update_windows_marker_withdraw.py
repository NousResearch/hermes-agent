"""Native withdrawal preserves foreign owners and removes only its own bridge."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.scripts.desktop_update import windows_marker_support
from tests.scripts.desktop_update.windows_handoff_support import MARKER, _creation_time, _dead_pid, _op

sleeper = windows_marker_support.sleeper


@pytest.mark.platforms('windows')
def test_marker_op_withdraw(tmp_path: Path, sleeper: subprocess.Popen) -> None:
    marker = tmp_path / MARKER
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    try:
        args = ('-MarkerOp', 'withdraw', '-DesktopPid', str(sleeper.pid), '-HandoffRun', 'desk-9')
        now = int(time.time())
        assert _op(tmp_path, *args)[:2] == (0, 'absent\n')
        # A late adoption: a live, non-Desktop owner carries our run -> taken.
        taken = f'{other.pid}\n{now}\nct:{_creation_time(other.pid)}\nrun:desk-9\n'.encode()
        marker.write_bytes(taken)
        assert _op(tmp_path, *args)[:2] == (0, f'taken {other.pid}\n')
        assert marker.read_bytes() == taken
        # Someone else's run (or a dead adopter) is foreign and kept.
        for body in (f'{other.pid}\n{now}\nct:{_creation_time(other.pid)}\nrun:desk-8\n',
                     f'{_dead_pid()}\n{now}\nct:5.000\nrun:desk-9\n'):
            marker.write_bytes(body.encode())
            assert _op(tmp_path, *args)[:2] == (0, 'foreign\n')
            assert marker.read_bytes() == body.encode()
        # Our own bridge is withdrawn.
        marker.write_bytes(f'{sleeper.pid}\n{now}\nct:{_creation_time(sleeper.pid)}\nrun:desk-9\n'.encode())
        code, out, err = _op(tmp_path, *args)
        assert (code, out) == (0, 'withdrawn\n'), err
        assert not marker.exists()
    finally:
        other.kill(); other.wait()
