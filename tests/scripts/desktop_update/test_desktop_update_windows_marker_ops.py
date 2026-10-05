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
param([string]$MarkerPs1, [string]$Marker, [string]$Body, [string]$Action, [string]$InstallRoot = '', [int]$Heartbeat = 300,
      [string]$StaleCt = '')
$MarkerPath = $Marker
$NoMarkerCleanup = $false
function Write-HandoffLog([string]$Message) { [Console]::Error.WriteLine($Message) }
. $MarkerPs1
$script:MarkerHeartbeatSeconds = $Heartbeat
if ($StaleCt) {   # "<pid>:<ct>": what the cache kept for an earlier incarnation of that pid
    $stale = $StaleCt.Split(':')
    $script:ProcessCtCache[[int]$stale[0]] = [double]::Parse($stale[1], [Globalization.CultureInfo]::InvariantCulture)
}
$own = Format-Ct (Get-LiveProcessCt $PID).Ct
[System.IO.File]::WriteAllText($MarkerPath, $Body.Replace('{self}', "$PID").Replace('{selfct}', $own))
$script:MarkerClaim = 'claimed'
if ($Action -eq 'release') { Invoke-MarkerRelease }
if ($Action -eq 'heartbeat') { $script:MarkerHeartbeatSeconds = 0; Update-MarkerHeartbeat }
[Console]::Out.Write("$PID $own")
"""


def _harness(tmp_path: Path, body: str, action: str, *extra: str) -> tuple[str, str]:
    harness = tmp_path / 'harness.ps1'
    harness.write_text(HARNESS, encoding='utf-8')
    proc = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(harness),
                           '-MarkerPs1', str(MARKER_PS1), '-Marker', str(tmp_path / MARKER),
                           '-Body', body, '-Action', action, *extra],
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
def test_release_rereads_a_cached_delegate_identity_before_handing_over(
    tmp_path: Path, sleeper: subprocess.Popen,
) -> None:
    """Get-LiveProcessCt forgets a pid only when it SEES it dead: a delegate that exited and
    whose pid was reused between two polls keeps its cached creation time. The release must
    re-read it, or the marker is handed to the unrelated process now wearing that pid."""
    started = int(time.time()) - 60
    stale = f'{{self}}\n{started}\nct:{{selfct}}\ndelegate:{sleeper.pid} ct:5.000\nrun:desk-3\n'
    _harness(tmp_path, stale, 'release', '-StaleCt', f'{sleeper.pid}:5.0')
    assert not (tmp_path / MARKER).exists(), (tmp_path / MARKER).read_bytes()


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


_CHECKOUT_HOLDER = """
import msvcrt, os, sys, time
from pathlib import Path
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_BINARY, 0o644)
# offset 1 MiB: exactly hermes_cli/update_lock.py::_try_lock; 1 MiB + 1..16: one R5b lease byte
os.lseek(fd, (1 << 20) + (int(sys.argv[3]) if len(sys.argv) > 3 else 0), os.SEEK_SET)
msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
Path(sys.argv[2] + '.ready').write_text('1', encoding='utf-8')
while not Path(sys.argv[2]).exists():
    time.sleep(0.05)
"""


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
@pytest.mark.parametrize('lease', [1, 16])
def test_marker_op_reclaim_reports_held_while_a_leased_child_outlives_its_owner(tmp_path: Path, lease: int) -> None:
    """R5b: the owner died (its byte at 1 MiB is free) but a completion child the job refused
    still holds its lease byte. The checkout is busy until it exits: reclaim answers `held`."""
    install = tmp_path / 'hermes-agent'
    install.mkdir()
    dead = f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode()
    (tmp_path / MARKER).write_bytes(dead)
    release = tmp_path / 'release-holder'
    holder = subprocess.Popen([sys.executable, '-c', _CHECKOUT_HOLDER, str(install / '.hermes-update.lock'),
                               str(release), str(lease)])
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


@pytest.mark.platforms('windows')
def test_release_wait_keeps_line_2_young_while_the_checkout_lock_is_held(tmp_path: Path) -> None:
    """R6 wait + old packaged Desktops: while the release waits on a survivor's checkout lock
    (up to hours), the heartbeat keeps line 2 young, or an old Desktop age-deletes the marker."""
    install = tmp_path / 'hermes-agent'
    install.mkdir()
    marker = tmp_path / MARKER
    release = tmp_path / 'release-holder'
    holder = subprocess.Popen([sys.executable, '-c', _CHECKOUT_HOLDER, str(install / '.hermes-update.lock'), str(release)])
    harness = tmp_path / 'harness.ps1'
    harness.write_text(HARNESS, encoding='utf-8')
    started = int(time.time()) - 900
    proc = None
    try:
        deadline = time.monotonic() + 30
        while not Path(str(release) + '.ready').exists():
            assert time.monotonic() < deadline and holder.poll() is None
            time.sleep(0.05)
        proc = subprocess.Popen([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(harness),
                                 '-MarkerPs1', str(MARKER_PS1), '-Marker', str(marker), '-Body', f'{{self}}\n{started}\nct:{{selfct}}\n',
                                 '-Action', 'release', '-InstallRoot', str(install), '-Heartbeat', '1'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 60
        while not marker.exists():
            assert time.monotonic() < deadline and proc.poll() is None, proc.communicate()
            time.sleep(0.05)
        seen = []
        for _ in range(2):
            time.sleep(4)
            assert proc.poll() is None, proc.communicate()
            lines = marker.read_bytes().decode().split('\n')
            assert time.time() - int(lines[1]) <= 3, lines
            seen.append(int(lines[1]))
        assert seen[1] > seen[0]
    finally:
        release.touch()
        holder.wait(timeout=30)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, out + err
    assert not marker.exists()
