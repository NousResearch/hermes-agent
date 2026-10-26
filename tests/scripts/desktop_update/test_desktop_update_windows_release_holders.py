"""#101789 on the real Windows hand-off script: a Desktop swap blocked by a program outside Hermes.

A real ``windows.ps1`` drives a published launcher whose ``hermes update`` is a scripted stand-in
printing exactly what ``hermes_cli.desktop_release_holders`` prints when Windows refused the
promotion rename. The result the relaunched Desktop reads must name that program (message and a
structured ``release_holders`` list), so the Desktop can offer to close exactly it, and the retry
that click starts must run under the run id its consent is bound to.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from hermes_cli.desktop_release_holders import ReleaseHolder, take_consent
from tests.installation_launcher_fixture import publish_fixture_launcher
from tests.scripts.desktop_update.windows_handoff_support import MARKER, POWERSHELL, _creation_time

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / 'scripts/desktop-update/windows.ps1'
EXE = r'D:\Program Files (x86)\SogouInput\16.7.0.4912\SogouCloud.exe'
CLI = """
import os, sys
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    if sys.argv[1:2] != ['update']:
        return 0
    if os.environ.get('CORRELATION_OUT'):   # the id update_receipt and take_consent read
        Path(os.environ['CORRELATION_OUT']).write_text(os.environ.get('HERMES_UPDATE_CORRELATION_ID', ''))
    print('  Desktop app folder held by: SogouCloud.exe (pid 4242, started 1761470000.125, EXE_PATH)', flush=True)
    print('  Desktop app folder held by: AVHelper.exe (pid 5151, started 1761470001.500, path unreadable)', flush=True)
    print('  Desktop app build owed: desktop app build failed', flush=True)
    print('Update complete!', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
""".replace('EXE_PATH', EXE.replace('\\', '\\\\'))


@pytest.mark.platforms('windows')
def test_blocked_swap_result_names_the_holder_for_the_desktop(tmp_path: Path) -> None:
    install = tmp_path / 'checkout'
    publish_fixture_launcher(install, CLI)
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n', encoding='utf-8')
    home = tmp_path / 'home'
    home.mkdir()
    # The consent the Desktop recorded for this retry; its swap is never reached (the build failed).
    consent = install / 'apps/desktop/release/.close-holders-consent.json'
    consent.parent.mkdir(parents=True, exist_ok=True)
    consent.write_text('{"run_id": "r", "holders": []}', encoding='utf-8')
    proc = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
         '-InstallRoot', str(install), '-NoUi', '-NoGateway'],
        check=False, cwd=tmp_path, capture_output=True, text=True, timeout=150,
        env={**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
             'PYTHONIOENCODING': 'utf-8'},
    )
    result_path = home / '.hermes-update-result.json'
    log = home / 'logs' / 'desktop-update-handoff.log'
    assert result_path.is_file(), (proc.stdout, log.read_text(encoding='utf-8', errors='replace') if log.is_file() else '')
    result = json.loads(result_path.read_text(encoding='utf-8-sig'))

    assert (result['ok'], result['manual']) == (True, True), (proc.stdout, result)
    assert result['release_holders'] == [
        {'name': 'SogouCloud.exe', 'pid': 4242, 'start_time': '1761470000.125', 'exe': EXE},
        {'name': 'AVHelper.exe', 'pid': 5151, 'start_time': '1761470001.500', 'exe': ''},
    ], result
    assert not consent.exists(), 'a consent the run never used ends with the run'
    assert 'SogouCloud.exe (pid 4242)' in result['message'], result
    assert 'not part of Hermes' in result['message'], result


# A relaunched Desktop: records the correlation id it inherited, but only when it was started by a
# rung that hands on the hand-off's own environment (the tethered Start-Process fallback). The WMI
# and explorer rungs start it without RELAUNCH_PROBE_OUT, so it exits at once and they fall through.
RELAUNCHED_DESKTOP = """
using System; using System.IO; using System.Threading;
public static class RelaunchedDesktop {
    public static int Main() {
        string probe = Environment.GetEnvironmentVariable("RELAUNCH_PROBE_OUT");
        if (String.IsNullOrEmpty(probe)) { return 0; }
        string corr = Environment.GetEnvironmentVariable("HERMES_UPDATE_CORRELATION_ID");
        File.WriteAllText(probe, System.Diagnostics.Process.GetCurrentProcess().Id + " " + (corr ?? "<unset>"));
        Thread.Sleep(90000);
        return 0;
    }
}
"""


@pytest.mark.platforms('windows')
def test_close_and_retry_runs_under_its_own_run_even_from_a_relaunched_desktop(tmp_path: Path) -> None:
    # The Desktop clicking "Close X and retry" was itself relaunched by the previous hand-off's
    # Start-Process rung, so it inherited that run's HERMES_UPDATE_CORRELATION_ID. Its retry binds
    # the consent to the NEW run (-HandoffRun); the update must look the consent up by that run, and
    # this hand-off's id must not reach the Desktop it relaunches in turn.
    install = tmp_path / 'checkout'
    publish_fixture_launcher(install, CLI)
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n', encoding='utf-8')
    home = tmp_path / 'home'
    home.mkdir()
    source = tmp_path / 'desktop.cs'
    source.write_text(RELAUNCHED_DESKTOP, encoding='utf-8')
    relaunch = tmp_path / 'release' / 'win-unpacked' / 'Hermes.exe'
    relaunch.parent.mkdir(parents=True)
    subprocess.run([POWERSHELL, '-NoProfile', '-Command',
                    f"Add-Type -Path '{source}' -OutputAssembly '{relaunch}' -OutputType ConsoleApplication"],
                   check=True, capture_output=True, timeout=120)
    # The quitting Desktop's protocol-2 bridge for this run (checkout.ts claimBridgeMarker), and the
    # program the user agreed to close in that run.
    run = 'desk-2-retry-ab12'
    desktop = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'])
    holder = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    corr_out, relaunch_out = tmp_path / 'corr.txt', tmp_path / 'relaunched.txt'
    relaunched = None
    try:
        (home / MARKER).write_bytes(
            f'{desktop.pid}\n{int(time.time())}\nct:{_creation_time(desktop.pid)}\nrun:{run}\n'.encode())
        proc = subprocess.run(
            [POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
             '-InstallRoot', str(install), '-NoUi', '-NoGateway', '-DesktopPid', str(desktop.pid),
             '-HandoffRun', run, '-RelaunchExe', str(relaunch)],
            check=False, cwd=tmp_path, capture_output=True, text=True, timeout=240,
            env={**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
                 'PYTHONIOENCODING': 'utf-8', 'HERMES_UPDATE_CORRELATION_ID': 'desk-1-previous-run',
                 'CORRELATION_OUT': str(corr_out), 'RELAUNCH_PROBE_OUT': str(relaunch_out)},
        )
        log = home / 'logs' / 'desktop-update-handoff.log'
        detail = (proc.returncode, proc.stdout[-2000:], log.read_text(encoding='utf-8', errors='replace')[-4000:]
                  if log.is_file() else '')
        assert corr_out.is_file() and relaunch_out.is_file(), detail
        relaunched_pid, relaunched_corr = relaunch_out.read_text(encoding='utf-8').split(' ', 1)
        relaunched = int(relaunched_pid)
        corr = corr_out.read_text(encoding='utf-8')

        # The real consumer of that id: the consent the Desktop bound to this run.
        live = psutil.Process(holder.pid)
        consent_dir = tmp_path / 'consent'
        consent_dir.mkdir()
        (consent_dir / '.close-holders-consent.json').write_text(json.dumps(
            {'run_id': run, 'holders': [{'pid': live.pid, 'start_time': live.create_time(), 'exe': live.exe()}]}),
            encoding='utf-8')
        covered = take_consent(consent_dir, corr).covers(
            ReleaseHolder(pid=live.pid, name=live.name(), exe=live.exe(), create_time=live.create_time()))
    finally:
        desktop.kill()
        holder.kill()
        if relaunched:
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(relaunched)], capture_output=True, check=False)
    assert (corr, covered) == (run, True), detail
    assert relaunched_corr == '<unset>', detail
