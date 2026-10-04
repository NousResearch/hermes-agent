"""Who may adopt a pre-written claim (real processes): an OLD packaged Desktop's
``cmd.exe`` wrapper v1 overwrite is accepted by lineage only (R4), and with
``-HandoffRun`` only the Desktop bridge for that run is adopted (protocol 2).
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.scripts.desktop_update.windows_handoff_support import (
    SCRIPT,
    MARKER,
    POWERSHELL,
    _creation_time,
    _script,
    _finish,
    _wait_for_log,
)


@pytest.fixture
def sleeper():
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    yield proc
    proc.kill()
    proc.wait()


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
(home / '.hermes-update-in-progress').write_bytes(f'{owner}\n{started}\n'.encode())
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
