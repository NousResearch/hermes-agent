"""Who may adopt a pre-written claim (real processes): an OLD packaged Desktop's
``cmd.exe`` wrapper v1 overwrite is accepted by lineage only (R4), and with
``-HandoffRun`` only the Desktop bridge for that run is adopted (protocol 2).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.scripts.desktop_update.lineage_rule_cases import ENV_CASES, RULE_CASES
from tests.scripts.desktop_update.windows_handoff_support import (
    SCRIPT,
    MARKER,
    MARKER_PS1,
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
other_desktop, env_skew = int(sys.argv[5]), int(sys.argv[6])  # a Desktop that is NOT the wrapper's parent
started = int(time.time())
env = dict(os.environ, HERMES_UPDATE_STARTED_AT=str(started + env_skew))
ps = [os.environ['POWERSHELL'], '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script,
      '-InstallRoot', str(home / 'hermes-agent'), '-NoUi', '-SelfTestMarker', '-NoMarkerCleanup',
      '-DesktopPid', str(other_desktop or os.getpid())]
wrapper = ['cmd.exe', '/d', '/s', '/c'] + (['start', '', '/b'] if mode == 'start-b' else [])
child = subprocess.Popen(wrapper + ps, cwd=home, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
owner = foreign or child.pid
(home / '.hermes-update-in-progress').write_bytes(f'{owner}\n{started}\n'.encode())
(home / 'desktop.json').write_text(f'{os.getpid()} {child.pid} {started}', encoding='utf-8')
time.sleep(600)
"""


def _old_desktop(home: Path, mode: str, foreign: int = 0, other_desktop: int = 0,
                 env_skew: int = 0) -> tuple[subprocess.Popen, int, int, int]:
    program = home.parent / 'old_desktop.py'
    program.write_text(OLD_DESKTOP, encoding='utf-8')
    desktop = subprocess.Popen([sys.executable, str(program), str(home), str(SCRIPT), mode, str(foreign),
                                str(other_desktop), str(env_skew)],
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


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('env_matches', [True, False], ids=['env-matches', 'env-differs'])
def test_r4_live_wrapper_not_the_desktops_child_is_adopted_only_with_the_handoff_started_at(
    tmp_path: Path, sleeper: subprocess.Popen, env_matches: bool,
) -> None:
    """The lineage table's divergent row (round 5 D11) on real processes: the cmd.exe wrapper is
    our live parent but not the Desktop's child; line 2 == HERMES_UPDATE_STARTED_AT adopts, as
    in bash. Any other startedAt refuses."""
    home = tmp_path / 'home'; home.mkdir()
    desktop, wrapper, started, _ = _old_desktop(home, 'waiting-wrapper', other_desktop=sleeper.pid,
                                                env_skew=0 if env_matches else -7)
    try:
        log = _wait_for_log(home, 'hand-off start:' if env_matches else 'exiting without claiming')
    finally:
        desktop.kill(); desktop.wait()
    if env_matches:
        assert 'marker=adopted' in log, log
        assert f"desktop pid {sleeper.pid}'s launcher pid {wrapper}" in log, log
        lines = (home / MARKER).read_bytes().decode().split('\n')
        assert lines[1] == str(started) and lines[2].startswith('ct:'), lines
    else:
        assert 'hand-off start:' not in log, log
        assert (home / MARKER).read_bytes().decode() == f'{wrapper}\n{started}\n'


# -- the one launcher-lineage rule, shared with marker.sh ---------------------

_RULE_HARNESS = r"""
param([string]$MarkerPs1, [string]$Marker, [string]$Cases)
$MarkerPath = $Marker
$NoMarkerCleanup = $true
function Write-HandoffLog([string]$Message) { [Console]::Error.WriteLine($Message) }
. $MarkerPs1
$data = [System.IO.File]::ReadAllText($Cases) | ConvertFrom-Json
foreach ($c in $data.rule) {
    $facts = @{}
    foreach ($p in $c.facts.PSObject.Properties) { $facts[$p.Name] = [bool]$p.Value }
    [Console]::Out.WriteLine("$($c.id)=$([int][bool](Test-MarkerLauncherRule $facts))")
}
foreach ($c in $data.env) {
    $env:HERMES_UPDATE_STARTED_AT = $c.env
    [Console]::Out.WriteLine("env_$($c.id)=$([int][bool](Test-MarkerEnvStartedAt $c.line2))")
}
"""

_PASCAL = {'v1': 'V1', 'names_desktop': 'NamesDesktop', 'named_alive': 'NamedAlive',
           'named_is_our_parent': 'NamedIsOurParent', 'named_parent_is_desktop': 'NamedParentIsDesktop',
           'env_started_matches': 'EnvStartedMatches'}


@pytest.mark.platforms('windows')
def test_launcher_lineage_rule_matches_the_shared_table(tmp_path: Path) -> None:
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps({
        'rule': [{'id': c['id'], 'facts': {_PASCAL[k]: v for k, v in c['facts'].items()}} for c in RULE_CASES],
        'env': [{'id': c['id'], 'env': c['env'], 'line2': c['line2']} for c in ENV_CASES],
    }), encoding='utf-8')
    harness = tmp_path / 'rule.ps1'
    harness.write_text(_RULE_HARNESS, encoding='utf-8')
    proc = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(harness),
                           '-MarkerPs1', str(MARKER_PS1), '-Marker', str(tmp_path / MARKER), '-Cases', str(cases)],
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = dict(line.split('=') for line in proc.stdout.split())
    want = {c['id']: '1' if c['expect'] else '0' for c in RULE_CASES}
    want.update({f"env_{c['id']}": '1' if c['expect'] else '0' for c in ENV_CASES})
    assert got == want, proc.stderr


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
