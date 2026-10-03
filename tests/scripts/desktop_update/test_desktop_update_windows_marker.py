"""Update-marker contract C1 v2 on the real Windows hand-off script.

Every case runs the real ``scripts/desktop-update/windows.ps1`` under
Windows PowerShell against a marker owned by a REAL process (a sleeping
Python child) and reads the marker / result / hand-off log it leaves.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / 'scripts/desktop-update/windows.ps1'
MARKER = '.hermes-update-in-progress'
CLI = """
import sys
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
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


def _creation_time(pid: int) -> str:
    # Limited query rights (what Python/Rust readers use): works for every process.
    out = subprocess.run(
        ['powershell', '-NoProfile', '-Command',
         f"$c = (Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CreationDate; "
         "[DateTimeOffset]::new($c.ToUniversalTime()).ToUnixTimeMilliseconds().ToString()"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout.strip()
    return f'{int(out) / 1000:.3f}'


def _protected_pid() -> int:
    """A live pid whose StartTime Windows PowerShell cannot read (SYSTEM / protected)."""
    out = subprocess.run(
        ['powershell', '-NoProfile', '-Command',
         "foreach ($n in 'csrss','wininit','services','smss','lsass') { foreach ($p in @(Get-Process -Name $n "
         "-ErrorAction SilentlyContinue)) { try { $null = $p.StartTime } catch { $p.Id; exit 0 } } }"],
        capture_output=True, text=True, timeout=60,
    ).stdout.split()
    assert out, 'no process with an unreadable StartTime on this host'
    return int(out[0])


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def _run(home: Path, *args: str, install: Path | None = None, timeout: int = 120):
    command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
               '-InstallRoot', str(install or home / 'hermes-agent'), '-NoUi', *args]
    env = {**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(home / 'empty-store')}
    proc = subprocess.Popen(command, cwd=home, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
        out, _ = proc.communicate()
        pytest.fail(f'hand-off did not finish within {timeout}s: {out}')
    return proc.pid, proc.returncode, out


def _result(home: Path) -> dict:
    return json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig'))


@pytest.mark.platforms('windows')
def test_claim_is_the_first_action_and_writes_a_v2_body(tmp_path: Path) -> None:
    pid, code, out = _run(tmp_path, '-SelfTestMarker', '-NoMarkerCleanup')
    assert code == 0, out
    lines = (tmp_path / MARKER).read_bytes().decode().split('\n')
    assert lines[0] == str(pid)
    assert lines[2].startswith('ct:') and len(lines[2].split('.')[-1]) == 3, lines
    log = (tmp_path / 'logs/desktop-update-handoff.log').read_text(encoding='utf-8-sig').splitlines()
    # Nothing (UI, Add-Type'd console helpers, probes) logs before the claim.
    assert 'claimed update marker' in log[0], log


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('ct_matches', [True, False])
def test_live_foreign_owner_refuses_and_reused_pid_is_reclaimed(
    tmp_path: Path, sleeper: subprocess.Popen, ct_matches: bool,
) -> None:
    ct = _creation_time(sleeper.pid) if ct_matches else '1000.000'   # wrong ct = pid reuse
    body = f'{sleeper.pid}\n{int(time.time())}\nct:{ct}\n'.encode()
    (tmp_path / MARKER).write_bytes(body)
    pid, code, out = _run(tmp_path, '-SelfTestMarker', '-NoMarkerCleanup')
    if ct_matches:
        assert code == 2, out
        assert (tmp_path / MARKER).read_bytes() == body
        # A refused run changed nothing and owns no result (A4): the other update reports.
        assert not (tmp_path / '.hermes-update-result.json').exists()
    else:
        assert code == 0, out
        lines = (tmp_path / MARKER).read_bytes().decode().split('\n')
        assert lines[0] == str(pid) and lines[2].startswith('ct:'), lines


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('ct_matches', [True, False])
def test_owner_creation_time_is_read_for_system_processes(tmp_path: Path, ct_matches: bool) -> None:
    """A1: a marker naming a SYSTEM/protected pid (a reused pid after a reboot) is judged by its
    real creation time, not treated as live forever because StartTime is access-denied."""
    owner = _protected_pid()
    ct = _creation_time(owner) if ct_matches else '1000.000'
    body = f'{owner}\n{int(time.time())}\nct:{ct}\n'.encode()
    (tmp_path / MARKER).write_bytes(body)
    pid, code, out = _run(tmp_path, '-SelfTestMarker', '-NoMarkerCleanup')
    if ct_matches:
        assert code == 2, out
        assert (tmp_path / MARKER).read_bytes() == body
    else:
        assert code == 0, out
        assert (tmp_path / MARKER).read_bytes().decode().split('\n')[0] == str(pid)


_BODIES = {   # contract A2: identical verdicts in every reader
    'crlf-bom-v2': ('\ufeff{pid}\r\n{now}\r\nct:{ct}\r\n', 'live'),
    'missing-line-2': ('{pid}\n', 'dead'),
    'garbled-line-2': ('{pid}\nsoon\nct:{ct}\n', 'dead'),
    'fractional-started-at': ('{pid}\n{now}.5\nct:{ct}\n', 'dead'),
    'garbled-ct-is-v1-fresh': ('{pid}\n{now}\nct:garbage\n', 'live'),
    'garbled-ct-is-v1-past-20min': ('{pid}\n{old}\nct:garbage\n', 'dead'),
    'delegate-without-ct-is-ignored': ('999999\n{now}\nct:1.000\ndelegate:{pid}\n', 'dead'),
}


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('case', sorted(_BODIES))
def test_marker_bodies_are_parsed_positionally_like_every_other_reader(
    tmp_path: Path, sleeper: subprocess.Popen, case: str,
) -> None:
    template, verdict = _BODIES[case]
    now = int(time.time())
    body = template.format(pid=sleeper.pid, now=now, old=now - 1300, ct=_creation_time(sleeper.pid)).encode()
    (tmp_path / MARKER).write_bytes(body)
    pid, code, out = _run(tmp_path, '-SelfTestMarker', '-NoMarkerCleanup')
    if verdict == 'live':
        assert code == 2, out
        assert (tmp_path / MARKER).read_bytes() == body
    else:
        assert code == 0, out
        assert (tmp_path / MARKER).read_bytes().decode().split('\n')[0] == str(pid)


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('bridge', ['absent', 'someone-else'])
def test_desktop_started_handoff_only_adopts_its_bridge(
    tmp_path: Path, sleeper: subprocess.Popen, bridge: str,
) -> None:
    """A4: the Desktop gave up on a late script: it must not claim fresh, run, or leave a result."""
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    try:
        body = f'{other.pid}\n{int(time.time())}\nct:{_creation_time(other.pid)}\n'.encode()
        if bridge == 'someone-else':
            (tmp_path / MARKER).write_bytes(body)
        _, code, out = _run(tmp_path, '-DesktopPid', str(sleeper.pid))
    finally:
        other.kill(); other.wait()
    assert code == 2, out
    assert not (tmp_path / '.hermes-update-result.json').exists()
    if bridge == 'absent':
        assert not (tmp_path / MARKER).exists()
    else:
        assert (tmp_path / MARKER).read_bytes() == body


@pytest.mark.platforms('windows')
def test_live_desktop_bridge_marker_is_adopted_keeping_its_started_at(
    tmp_path: Path, sleeper: subprocess.Popen,
) -> None:
    started = int(time.time()) - 30
    (tmp_path / MARKER).write_bytes(f'{sleeper.pid}\n{started}\nct:{_creation_time(sleeper.pid)}\n'.encode())
    pid, code, out = _run(tmp_path, '-SelfTestMarker', '-NoMarkerCleanup', '-DesktopPid', str(sleeper.pid))
    assert code == 0, out
    lines = (tmp_path / MARKER).read_bytes().decode().split('\n')
    assert lines[:2] == [str(pid), str(started)], lines
    assert lines[2].startswith('ct:'), lines
    assert _result(tmp_path)['started_at'] == started


@pytest.mark.platforms('windows')
def test_dead_owner_marker_is_reclaimed_and_released(tmp_path: Path) -> None:
    (tmp_path / MARKER).write_bytes(f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode())
    pid, code, out = _run(tmp_path, '-SelfTestMarker')
    assert code == 0, out
    assert not (tmp_path / MARKER).exists()
    log = (tmp_path / 'logs/desktop-update-handoff.log').read_text(encoding='utf-8-sig')
    assert f'claimed update marker (pid {pid})' in log
    assert 'removed update marker (owned)' in log


@pytest.mark.platforms('windows')
def test_desktop_that_never_exits_is_not_relaunched_over(
    tmp_path: Path, sleeper: subprocess.Popen,
) -> None:
    install = tmp_path / 'checkout'
    publish_fixture_launcher(install, CLI)
    home = tmp_path / 'home'; home.mkdir()
    # The Desktop's bridge claim, which a -DesktopPid hand-off adopts (A4).
    (home / MARKER).write_bytes(f'{sleeper.pid}\n{int(time.time())}\nct:{_creation_time(sleeper.pid)}\n'.encode())
    relaunch = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'hostname.exe'
    _, code, out = _run(home, '-DesktopPid', str(sleeper.pid), '-RelaunchExe', str(relaunch),
                        install=install, timeout=180)
    assert code == 4, out
    log = (home / 'logs/desktop-update-handoff.log').read_text(encoding='utf-8-sig')
    assert 'relaunching desktop' not in log, log


@pytest.mark.platforms('windows')
def test_ui_profile_sweep_keeps_dirs_of_live_handoffs(tmp_path: Path, sleeper: subprocess.Popen) -> None:
    live = tmp_path / f'hermes-update-ui-{sleeper.pid}'
    dead = tmp_path / f'hermes-update-ui-{_dead_pid()}'
    live.mkdir(); dead.mkdir()
    home = tmp_path / 'home'; home.mkdir()
    result = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT), '-SelfTestUi', '-NoUi'],
        cwd=tmp_path, env={**os.environ, 'HERMES_HOME': str(home), 'TEMP': str(tmp_path),
                           'HERMES_SELFTEST_HOLD_SECONDS': '0'},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert live.exists(), 'another live hand-off lost its browser profile'
    assert not dead.exists()
