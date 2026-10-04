"""Protocol-2 helper ops (``windows.ps1 -MarkerOp reclaim|withdraw``) and the
production release / heartbeat functions, on real processes and files.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.scripts.desktop_update.windows_handoff_support import (
    MARKER_PS1,
    MARKER,
    POWERSHELL,
    _creation_time,
    _dead_pid,
    _op,
    _HeldLock,
)


@pytest.fixture
def sleeper():
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    yield proc
    proc.kill()
    proc.wait()


# -- helper ops ---------------------------------------------------------------

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


# -- release / heartbeat (production functions in a real PowerShell process) ---

HARNESS = r"""
param([string]$MarkerPs1, [string]$Marker, [string]$Body, [string]$Action)
$MarkerPath = $Marker
$NoMarkerCleanup = $false
function Write-HandoffLog([string]$Message) { [Console]::Error.WriteLine($Message) }
. $MarkerPs1
$own = Format-Ct (Get-LiveProcessCt $PID).Ct
[System.IO.File]::WriteAllText($MarkerPath, $Body.Replace('{self}', "$PID").Replace('{selfct}', $own))
$script:MarkerClaim = 'claimed'
if ($Action -eq 'release') { Invoke-MarkerRelease }
if ($Action -eq 'heartbeat') { $script:MarkerHeartbeatSeconds = 0; Update-MarkerHeartbeat }
[Console]::Out.Write("$PID $own")
"""


def _harness(tmp_path: Path, body: str, action: str) -> tuple[str, str]:
    harness = tmp_path / 'harness.ps1'
    harness.write_text(HARNESS, encoding='utf-8')
    proc = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(harness),
                           '-MarkerPs1', str(MARKER_PS1), '-Marker', str(tmp_path / MARKER),
                           '-Body', body, '-Action', action],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pid, ct = proc.stdout.split()
    return pid, ct


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('delegate', ['live', 'dead'])
def test_release_hands_the_claim_to_a_live_delegate(
    tmp_path: Path, sleeper: subprocess.Popen, delegate: str,
) -> None:
    """A7 rule 5 (the crash-12db shape from the owner's side): the releasing owner deletes the
    marker unless its delegate still runs -- then that delegate becomes the owner."""
    dpid = sleeper.pid if delegate == 'live' else _dead_pid()
    dct = _creation_time(sleeper.pid) if delegate == 'live' else '5.000'
    started = int(time.time()) - 60
    _harness(tmp_path, f'{{self}}\n{started}\nct:{{selfct}}\ndelegate:{dpid} ct:{dct}\nrun:desk-3\n', 'release')
    if delegate == 'dead':
        assert not (tmp_path / MARKER).exists()
    else:
        assert (tmp_path / MARKER).read_bytes().decode() == f'{dpid}\n{started}\nct:{dct}\nrun:desk-3\n'


@pytest.mark.platforms('windows')
def test_heartbeat_refreshes_line_2_only_for_our_own_claim(tmp_path: Path, sleeper: subprocess.Popen) -> None:
    started = int(time.time()) - 900
    dct = _creation_time(sleeper.pid)
    pid, ct = _harness(tmp_path, f'{{self}}\n{started}\nct:{{selfct}}\ndelegate:{sleeper.pid} ct:{dct}\nrun:r\n',
                       'heartbeat')
    lines = (tmp_path / MARKER).read_bytes().decode().split('\n')
    assert lines[0] == pid and lines[2:] == [f'ct:{ct}', f'delegate:{sleeper.pid} ct:{dct}', 'run:r', '']
    assert int(time.time()) - int(lines[1]) < 60, lines
    foreign = f'{sleeper.pid}\n{started}\nct:{dct}\n'
    _harness(tmp_path, foreign, 'heartbeat')
    assert (tmp_path / MARKER).read_bytes().decode() == foreign
