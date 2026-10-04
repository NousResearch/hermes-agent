"""Hand-off protocol 2 / A7 on the real Windows hand-off script (real processes, real files).

* A7: every marker mutation happens inside one hold of the kernel lock on
  ``<marker>.lock``; the decision is made inside that hold.
* R4: an OLD packaged Desktop (be3fd671d70 checkout.ts) spawns a ``cmd.exe``
  wrapper and then overwrites the marker with ``<wrapper pid>\\n<startedAt>\\n``;
  the script accepts that by lineage, never by dropping the check.
* Protocol 2: ``-HandoffRun`` adopts only the Desktop's bridge for that run;
  ``-MarkerOp reclaim|withdraw`` are Electron's only way to mutate the marker.
* Release hands the claim to a live delegate; the update child is named as the
  delegate before it runs a single instruction.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / 'scripts/desktop-update/windows.ps1'
MARKER_PS1 = ROOT / 'scripts/desktop-update/marker.ps1'
MARKER = '.hermes-update-in-progress'
POWERSHELL = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'),
                          'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
HOLD_CLI = """
import os, sys, time
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    if sys.argv[1:2] == ['update']:   # the update's first act: say so, then wait to be released
        hold = Path(os.environ['HANDOFF_HOLD'])
        Path(str(hold) + '.pid').write_text(str(os.getpid()), encoding='utf-8')
        while not hold.exists():
            time.sleep(0.05)
    return 0
if __name__ == '__main__':
    sys.exit(main())
"""


@pytest.fixture
def sleeper():
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    yield proc
    proc.kill()
    proc.wait()


def _ps(command: str, timeout: int = 60) -> str:
    return subprocess.run([POWERSHELL, '-NoProfile', '-Command', command],
                          capture_output=True, text=True, timeout=timeout, check=True).stdout.strip()


def _creation_time(pid: int) -> str:
    out = _ps(f"$c = (Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CreationDate; "
              "[DateTimeOffset]::new($c.ToUniversalTime()).ToUnixTimeMilliseconds().ToString()")
    return f'{int(out) / 1000:.3f}'


def _alive(pid: int) -> bool:
    return str(pid) in subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'],
                                      capture_output=True, text=True).stdout


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def _env(home: Path, **extra: str) -> dict[str, str]:
    return {**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(home / 'empty-store'), **extra}


def _script(home: Path, *args: str, install: Path | None = None, **env: str) -> subprocess.Popen:
    return subprocess.Popen(
        [POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
         '-InstallRoot', str(install or home / 'hermes-agent'), '-NoUi', *args],
        cwd=home, env=_env(home, **env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _finish(proc: subprocess.Popen, timeout: int = 120) -> tuple[int, str]:
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
        out, _ = proc.communicate()
        pytest.fail(f'hand-off did not finish within {timeout}s: {out}')
    return proc.returncode, out


def _op(home: Path, *args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
         '-InstallRoot', str(home / 'hermes-agent'), *args],
        cwd=home, env=_env(home), capture_output=True, text=True, timeout=120)
    return proc.returncode, proc.stdout, proc.stderr


def _log(home: Path) -> str:
    path = home / 'logs/desktop-update-handoff.log'
    return path.read_text(encoding='utf-8-sig') if path.exists() else ''


class _HeldLock:
    """Hold the A7 sidecar lock from another process (this one): an open handle
    conflicts with the script's FileShare.None open."""

    def __init__(self, home: Path) -> None:
        deadline = time.monotonic() + 30
        while True:
            try:
                self._f = open(home / (MARKER + '.lock'), 'a+b')   # noqa: SIM115  # windows-footgun: ok — binary mode
                return
            except PermissionError:   # the script holds it right now
                assert time.monotonic() < deadline, 'never got the marker lock'
                time.sleep(0.02)

    def release(self) -> None:
        self._f.close()


# -- A7 ---------------------------------------------------------------------

@pytest.mark.platforms('windows')
def test_a7_claim_is_decided_inside_the_marker_lock(tmp_path: Path, sleeper: subprocess.Popen) -> None:
    """While another process holds <marker>.lock the claimant must not touch a dead marker;
    once the holder (which replaced it with a live claim meanwhile) lets go, it judges THAT."""
    dead = f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode()
    marker = tmp_path / MARKER
    marker.write_bytes(dead)
    lock = _HeldLock(tmp_path)
    try:
        claimant = _script(tmp_path, '-SelfTestMarker', '-NoMarkerCleanup')
        time.sleep(5)   # PowerShell start-up plus a few lock retries
        assert claimant.poll() is None, 'claimant did not wait for the marker lock'
        assert marker.read_bytes() == dead, 'a dead marker was mutated outside the marker lock'
        live = f'{sleeper.pid}\n{int(time.time())}\nct:{_creation_time(sleeper.pid)}\n'.encode()
        marker.write_bytes(live)   # the lock holder's own decision: a new live claim
    finally:
        lock.release()
    code, out = _finish(claimant)
    assert code == 2, out
    assert marker.read_bytes() == live


@pytest.mark.platforms('windows')
def test_a7_concurrent_claimants_over_a_dead_marker_yield_one_owner(tmp_path: Path) -> None:
    install = tmp_path / 'checkout'
    publish_fixture_launcher(install, HOLD_CLI)
    home = tmp_path / 'home'; home.mkdir()
    marker = home / MARKER
    marker.write_bytes(f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode())
    hold = tmp_path / 'release-update'
    lock = _HeldLock(home)   # line every claimant up behind the lock, then let them race
    claimants = [_script(home, install=install, HANDOFF_HOLD=str(hold)) for _ in range(4)]
    try:
        time.sleep(6)
        lock.release()
        deadline = time.monotonic() + 120
        while not Path(str(hold) + '.pid').exists():
            assert time.monotonic() < deadline, 'no claimant reached the update'
            time.sleep(0.05)
        while sum(c.poll() is not None for c in claimants) < 3:
            assert time.monotonic() < deadline, [c.poll() for c in claimants]
            time.sleep(0.1)
        refused = [c for c in claimants if c.poll() == 2]
        running = [c for c in claimants if c.poll() is None]
        assert len(running) == 1 and len(refused) == 3, [c.returncode for c in claimants]
        assert marker.read_bytes().decode().split('\n')[0] == str(running[0].pid)
    finally:
        hold.touch()
        for c in claimants:
            if c.poll() is None:
                try:
                    c.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill', '/T', '/F', '/PID', str(c.pid)], capture_output=True)
    assert running[0].returncode == 0
    assert not marker.exists()


# -- R4: an old packaged Desktop + this script ---------------------------------

OLD_DESKTOP = r"""
import os, subprocess, sys, time
from pathlib import Path
# be3fd671d70 checkout.ts: spawn the cmd.exe wrapper (non-detached), then in the
# same tick overwrite the marker with the WRAPPER's pid as a v1 claim, then
# stay up (the -SelfTestMarker script ends before it would wait us out).
home, script, mode, foreign = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
started = int(time.time())
env = dict(os.environ, HERMES_UPDATE_STARTED_AT=str(started))
ps = [os.environ['POWERSHELL'], '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script,
      '-InstallRoot', str(home / 'hermes-agent'), '-NoUi', '-SelfTestMarker', '-NoMarkerCleanup',
      '-DesktopPid', str(os.getpid())]
wrapper = ['cmd.exe', '/d', '/s', '/c'] + (['start', '', '/b'] if mode == 'start-b' else [])
child = subprocess.Popen(wrapper + ps, cwd=home, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
owner = foreign or child.pid
(home / '.hermes-update-in-progress').write_text(f'{owner}\n{started}\n', encoding='utf-8')
(home / 'desktop.json').write_text(f'{os.getpid()} {child.pid} {started}', encoding='utf-8')
time.sleep(600)
"""


def _old_desktop(home: Path, mode: str, foreign: int = 0) -> tuple[subprocess.Popen, int, int, int]:
    program = home.parent / 'old_desktop.py'
    program.write_text(OLD_DESKTOP, encoding='utf-8')
    desktop = subprocess.Popen([sys.executable, str(program), str(home), str(SCRIPT), mode, str(foreign)],
                               env={**os.environ, 'HERMES_HOME': str(home), 'POWERSHELL': POWERSHELL})
    deadline = time.monotonic() + 30
    while not (home / 'desktop.json').exists():
        assert time.monotonic() < deadline and desktop.poll() is None, 'old desktop never spawned'
        time.sleep(0.05)
    time.sleep(0.2)
    desktop_pid, wrapper, started = map(int, (home / 'desktop.json').read_text(encoding='utf-8-sig').split())
    return desktop, wrapper, started, desktop_pid


def _wait_for_log(home: Path, needle: str, timeout: int = 90) -> str:
    deadline = time.monotonic() + timeout
    while needle not in _log(home):
        assert time.monotonic() < deadline, f'{needle!r} never logged: {_log(home)}'
        time.sleep(0.1)
    return _log(home)


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('mode', ['start-b', 'waiting-wrapper'])
def test_r4_old_desktop_wrapper_claim_is_adopted_by_lineage(tmp_path: Path, mode: str) -> None:
    """`start /b` wrapper (exits at once: lineage via the startedAt it was given) and a wrapper
    still alive (lineage via its parent == the Desktop) both hand the claim to the script."""
    home = tmp_path / 'home'; home.mkdir()
    desktop, wrapper, started, desktop_pid = _old_desktop(home, mode)
    try:
        log = _wait_for_log(home, 'hand-off start:')
    finally:
        desktop.kill(); desktop.wait()
    assert 'marker=adopted' in log, log
    script_pid = log.split('hand-off start:')[1].split(' pid=')[1].split()[0]
    lines = (home / MARKER).read_bytes().decode().split('\n')
    assert lines[:2] == [script_pid, str(started)], lines
    assert lines[2].startswith('ct:'), lines
    assert f"desktop pid {desktop_pid}'s launcher pid {wrapper}" in log, log


@pytest.mark.platforms('windows')
def test_r4_old_desktop_lineage_never_adopts_an_unrelated_live_claim(
    tmp_path: Path, sleeper: subprocess.Popen,
) -> None:
    home = tmp_path / 'home'; home.mkdir()
    desktop, _, started, _ = _old_desktop(home, 'start-b', foreign=sleeper.pid)
    try:
        log = _wait_for_log(home, 'exiting without claiming')
    finally:
        desktop.kill(); desktop.wait()
    assert 'hand-off start:' not in log, log
    time.sleep(1)
    assert (home / MARKER).read_bytes().decode() == f'{sleeper.pid}\n{started}\n'


# -- protocol 2: -HandoffRun --------------------------------------------------

@pytest.mark.platforms('windows')
@pytest.mark.parametrize('bridge', ['ok', 'other-run', 'stale-ct', 'v1'])
def test_handoff_run_adopts_only_the_desktop_bridge_for_that_run(
    tmp_path: Path, sleeper: subprocess.Popen, bridge: str,
) -> None:
    started = int(time.time()) - 20
    ct = _creation_time(sleeper.pid)
    if bridge == 'stale-ct':
        ct = f'{float(ct) - 5:.3f}'
    run = 'other.run' if bridge == 'other-run' else 'desk-1-ab-12cd'
    body = f'{sleeper.pid}\n{started}\n' + ('' if bridge == 'v1' else f'ct:{ct}\n') + f'run:{run}\n'
    (tmp_path / MARKER).write_bytes(body.encode())
    code, out = _finish(_script(tmp_path, '-SelfTestMarker', '-NoMarkerCleanup',
                                '-DesktopPid', str(sleeper.pid), '-HandoffRun', 'desk-1-ab-12cd'))
    if bridge != 'ok':
        assert code == 2, out
        assert (tmp_path / MARKER).read_bytes() == body.encode()
        return
    assert code == 0, out
    lines = (tmp_path / MARKER).read_bytes().decode().split('\n')
    pid = out.split(' pid=')[1].split()[0]
    assert lines[0] == pid and lines[1] == str(started) and lines[2].startswith('ct:'), lines
    assert lines[3:] == ['run:desk-1-ab-12cd', ''], lines


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


# -- the update child is the delegate before it runs anything ------------------

FIND_UPDATE_CHILD = r"""
param([int]$Parent, [int]$Seconds)
$deadline = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $deadline) {
    $row = Get-CimInstance Win32_Process -Filter "ParentProcessId=$Parent" |
        Where-Object { $_.CommandLine -match '--yes' } | Select-Object -First 1
    if ($row) { [Console]::Out.Write($row.ProcessId); exit 0 }
    Start-Sleep -Milliseconds 100
}
exit 1
"""


@pytest.mark.platforms('windows')
def test_script_killed_before_publishing_the_delegate_runs_no_update(tmp_path: Path) -> None:
    """Kill cell at the pre-publication boundary: windows.ps1 has created its `hermes update`
    child but has not published it as the marker delegate (the marker lock is held elsewhere).
    Killed there, no update instruction may have run and the marker must read DEAD."""
    install = tmp_path / 'checkout'
    publish_fixture_launcher(install, HOLD_CLI)
    home = tmp_path / 'home'; home.mkdir()
    hold = tmp_path / 'release-update'
    ran = Path(str(hold) + '.pid')
    marker = home / MARKER
    finder = tmp_path / 'find_child.ps1'
    finder.write_text(FIND_UPDATE_CHILD, encoding='utf-8')
    script = _script(home, install=install, HANDOFF_HOLD=str(hold))
    lock = None
    try:
        deadline = time.monotonic() + 60
        while not (marker.exists() and marker.read_bytes().split(b'\n')[0] == str(script.pid).encode()):
            assert time.monotonic() < deadline and script.poll() is None, 'script never claimed'
            time.sleep(0.02)
        lock = _HeldLock(home)   # after the claim, before the delegate publication
        child = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(finder),
                                '-Parent', str(script.pid), '-Seconds', '60'],
                               capture_output=True, text=True, timeout=90)
        assert child.returncode == 0, 'the update child never appeared'
        child_pid = int(child.stdout)
        subprocess.run(['taskkill', '/F', '/PID', str(script.pid)], capture_output=True, check=True)
        script.wait(timeout=30)
        time.sleep(3)
        assert not ran.exists(), 'the update child ran before it was published as the delegate'
        assert not _alive(child_pid), 'the never-resumed update child outlived its script'
        lines = marker.read_bytes().decode().split('\n')
        assert lines[0] == str(script.pid) and len(lines) == 4 and lines[3] == '', lines   # no delegate line
    finally:
        if lock:
            lock.release()
        hold.touch()
        if script.poll() is None:
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(script.pid)], capture_output=True)
            script.wait()
    assert _op(home, '-MarkerOp', 'reclaim')[:2] == (0, 'reclaimed\n')
