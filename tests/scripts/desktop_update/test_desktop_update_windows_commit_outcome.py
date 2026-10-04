"""Post-commit outcomes (contract C3) on the real Windows hand-off script.

A real ``windows.ps1`` drives a real published launcher whose ``hermes``
application is a scripted stand-in: once ``hermes update`` exits 0 the install
is on the new version, so nothing afterwards may report "still on the previous
version", and the hand-off's own supervision (watchdog, probes, marker
release) must not turn healthy work into a failure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / 'scripts/desktop-update/windows.ps1'
CLI = """
import ctypes, json, os, subprocess, sys, time
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        if os.environ.get('HANDOFF_HELP_HANG'):
            time.sleep(600)
        print('--keep-stash'); return 0
    with Path(os.environ['HANDOFF_CALLS']).open('a') as stream:
        stream.write(json.dumps(sys.argv[1:]) + '\\n')
    if sys.argv[1:2] != ['update']:
        return 0
    busy = float(os.environ.get('HANDOFF_BUSY_SECONDS', '0'))
    end = time.monotonic() + busy
    while time.monotonic() < end:   # pipe-silent, CPU-busy (download/extract shape)
        pass
    if os.environ.get('HANDOFF_DELEGATE'):
        # Leave a live process running under the claim, named on line 4.
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Path(os.environ['HANDOFF_DELEGATE']).write_text(str(child.pid))
        # update_lock's delegate line: pid + creation time (GetProcessTimes, limited rights).
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = ctypes.c_void_p(k32.OpenProcess(0x1000, False, child.pid))
        times = [ctypes.c_ulonglong() for _ in range(4)]
        k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times])
        k32.CloseHandle(handle)
        ct = times[0].value / 1e7 - 11644473600
        marker = Path(os.environ['HERMES_HOME']) / '.hermes-update-in-progress'
        head = marker.read_bytes().decode().split('\\n')[:3]
        marker.write_bytes(('\\n'.join(head) + '\\ndelegate:%d ct:%.3f\\n' % (child.pid, ct)).encode())
    if os.environ.get('HANDOFF_HANG'):
        print(os.environ['HANDOFF_HANG'], flush=True)
        time.sleep(300)
    print('Update complete!', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
"""


def _handoff(tmp_path: Path, *args: str, verify: str = 'pass\n', timeout: int = 150, **env: str):
    install = tmp_path / 'checkout'
    publish_fixture_launcher(install, CLI)
    (install / 'hermes_cli/desktop_update_verify.py').write_text(verify, encoding='utf-8')
    home = tmp_path / 'home'; home.mkdir()
    calls = tmp_path / 'calls.jsonl'
    proc = subprocess.Popen(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
         '-InstallRoot', str(install), '-NoUi', *args],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
             'HANDOFF_CALLS': str(calls), 'PYTHONIOENCODING': 'utf-8', **env},
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
        out, _ = proc.communicate()
        pytest.fail(f'hand-off did not finish within {timeout}s: {out}')
    argv = [json.loads(line) for line in calls.read_text(encoding='utf-8-sig').splitlines()] if calls.exists() else []
    result = json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig'))
    return proc.returncode, out, argv, result, home


@pytest.mark.platforms('windows')
def test_verify_failure_after_a_committed_update_is_ok_with_warnings(tmp_path: Path) -> None:
    code, out, argv, result, _ = _handoff(tmp_path, verify='raise SystemExit(3)\n')
    assert code == 0, out
    assert (result['ok'], result['exit_code'], result['manual']) == (True, 0, True), result
    assert [w.split(':')[0] for w in result['warnings']] == ['verify'], result
    assert result['message'].startswith('Hermes was updated, but'), result
    assert 'previous version' not in result['message']
    # The fleet the Desktop stopped is still brought back.
    assert argv[-1] == ['gateway', 'start', '--all'], argv


@pytest.mark.platforms('windows')
def test_result_is_atomic_json_carrying_started_at_and_warnings(tmp_path: Path) -> None:
    started = int(time.time()) - 5
    code, out, _, result, home = _handoff(tmp_path, HERMES_UPDATE_STARTED_AT=str(started))
    assert code == 0, out
    assert (result['ok'], result['started_at'], result['warnings']) == (True, started, []), result
    assert not list(home.glob('.hermes-update-result.json.*.tmp'))


@pytest.mark.platforms('windows')
def test_marker_with_a_live_delegate_is_handed_to_it_at_finish(tmp_path: Path) -> None:
    # A7 release rule: the owner deletes its claim unless a delegate still runs; then that
    # delegate becomes the owner (canonical body, started_at kept, no delegate line).
    pid_file = tmp_path / 'delegate.pid'
    try:
        code, out, _, result, home = _handoff(tmp_path, HANDOFF_DELEGATE=str(pid_file))
        assert code == 0, out
        marker = (home / '.hermes-update-in-progress').read_bytes().decode().split('\n')
        assert marker[0] == pid_file.read_text(encoding='utf-8-sig'), marker
        assert marker[2].startswith('ct:') and marker[3:] == [''], marker
    finally:
        if pid_file.exists():
            subprocess.run(['taskkill', '/F', '/PID', pid_file.read_text(encoding='utf-8-sig')], capture_output=True)


@pytest.mark.platforms('windows')
def test_watchdog_remap_after_banner_reports_interrupted_followups(tmp_path: Path) -> None:
    code, out, argv, result, _ = _handoff(
        tmp_path, '-NoGateway', HANDOFF_HANG='Update complete! (v1.0.0)',
        HERMES_UPDATE_STEP_IDLE_SECONDS='3',
    )
    assert code == 0, out
    assert (result['ok'], result['manual']) == (True, True), result
    assert any('interrupted' in w for w in result['warnings']), result


@pytest.mark.platforms('windows')
def test_cpu_busy_pipe_silent_update_is_not_killed_by_the_idle_watchdog(tmp_path: Path) -> None:
    code, out, argv, result, _ = _handoff(
        tmp_path, '-NoGateway', HANDOFF_BUSY_SECONDS='15', HERMES_UPDATE_STEP_IDLE_SECONDS='4',
    )
    assert code == 0, out
    assert (result['ok'], result['warnings']) == (True, []), result


@pytest.mark.platforms('windows')
def test_hanging_update_help_probe_is_bounded(tmp_path: Path) -> None:
    code, out, argv, result, _ = _handoff(
        tmp_path, '-NoGateway', '-ProbeTimeoutSeconds', '10', timeout=120, HANDOFF_HELP_HANG='1',
    )
    assert code == 0, out
    assert argv[0][:2] == ['update', '--yes'] and '--keep-stash' not in argv[0], argv
