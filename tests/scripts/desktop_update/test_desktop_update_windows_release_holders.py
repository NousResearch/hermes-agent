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


# The Desktop that clicked "Close X and retry": it quits once the hand-off has adopted its bridge.
# The hand-off holds the marker open while it rewrites it, so a read can be refused mid-adoption.
QUITTING_DESKTOP = """
import sys, time
from pathlib import Path
marker, bridge = Path(sys.argv[1]), None
end = time.monotonic() + 180
while time.monotonic() < end:
    try:
        owner = marker.read_bytes().split(b'\\n')[0] if marker.exists() else None
    except OSError:
        owner = bridge
    if bridge is None:
        bridge = owner
    elif owner != bridge:
        break
    time.sleep(0.05)
"""

# A relaunched Desktop, whichever relaunch rung starts it: records the correlation id it inherited
# beside its own exe, then stays up without a window. It never exits early, so the first rung wins
# on every runner, and that rung spends its focus wait polling for a window while the test looks.
RELAUNCHED_DESKTOP = """
using System; using System.IO; using System.Threading;
public static class RelaunchedDesktop {
    public static int Main() {
        string corr = Environment.GetEnvironmentVariable("HERMES_UPDATE_CORRELATION_ID");
        File.AppendAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "launches.txt"),
            System.Diagnostics.Process.GetCurrentProcess().Id + " " + (corr ?? "<unset>") + Environment.NewLine);
        Thread.Sleep(120000);
        return 0;
    }
}
"""


def _env_when_relaunching(proc: subprocess.Popen, launches: Path) -> dict[str, str] | None:
    """The hand-off's own environment once it has started the Desktop: what an inheriting rung hands on."""
    deadline = time.monotonic() + 240
    while proc.poll() is None and time.monotonic() < deadline:
        if launches.is_file() and launches.read_text(encoding='utf-8').strip():
            try:
                return psutil.Process(proc.pid).environ()
            except psutil.NoSuchProcess:
                return None
        time.sleep(0.05)
    return None


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
    # Named apart from other tests' stand-in Hermes.exe: the explorer rung takes any fresh process of
    # this name as its launch. A GUI-subsystem exe, so no console window satisfies the focus wait.
    relaunch = tmp_path / 'release' / 'win-unpacked' / 'HermesRetryDesktop.exe'
    relaunch.parent.mkdir(parents=True)
    subprocess.run([POWERSHELL, '-NoProfile', '-Command',
                    f"Add-Type -Path '{source}' -OutputAssembly '{relaunch}' -OutputType WindowsApplication"],
                   check=True, capture_output=True, timeout=120)
    launches = relaunch.parent / 'launches.txt'
    # The quitting Desktop's protocol-2 bridge for this run (checkout.ts claimBridgeMarker), and the
    # program the user agreed to close in that run.
    run = 'desk-2-retry-ab12'
    (tmp_path / 'desktop.py').write_text(QUITTING_DESKTOP, encoding='utf-8')
    desktop = subprocess.Popen([sys.executable, str(tmp_path / 'desktop.py'), str(home / MARKER)])
    holder = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    corr_out, out = tmp_path / 'corr.txt', tmp_path / 'handoff.out'
    try:
        (home / MARKER).write_bytes(
            f'{desktop.pid}\n{int(time.time())}\nct:{_creation_time(desktop.pid)}\nrun:{run}\n'.encode())
        with out.open('w', encoding='utf-8') as sink:
            proc = subprocess.Popen(
                [POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
                 '-InstallRoot', str(install), '-NoUi', '-NoGateway', '-DesktopPid', str(desktop.pid),
                 '-HandoffRun', run, '-RelaunchExe', str(relaunch)],
                cwd=tmp_path, stdout=sink, stderr=subprocess.STDOUT, text=True,
                env={**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
                     'PYTHONIOENCODING': 'utf-8', 'HERMES_UPDATE_CORRELATION_ID': 'desk-1-previous-run',
                     'CORRELATION_OUT': str(corr_out)},
            )
            handoff_env = _env_when_relaunching(proc, launches)
            proc.wait(timeout=240)
        log = home / 'logs' / 'desktop-update-handoff.log'
        detail = (proc.returncode, desktop.poll(), out.read_text(encoding='utf-8', errors='replace')[-2000:],
                  log.read_text(encoding='utf-8', errors='replace')[-4000:] if log.is_file() else '')
        assert corr_out.is_file() and launches.is_file() and handoff_env is not None, detail
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
        for line in launches.read_text(encoding='utf-8').splitlines() if launches.is_file() else ():
            subprocess.run(['taskkill', '/T', '/F', '/PID', line.split(' ', 1)[0]], capture_output=True, check=False)
    assert (corr, covered) == (run, True), detail
    # Only the tethered Start-Process rung inherits the hand-off's environment, so check both ends:
    # what the hand-off itself carried while it relaunched, and what the Desktop it started saw.
    leaked = {k: v for k, v in handoff_env.items() if k.upper() == 'HERMES_UPDATE_CORRELATION_ID'}
    assert leaked == {}, detail
    assert [line.split(' ', 1)[1] for line in launches.read_text(encoding='utf-8').splitlines()] == ['<unset>'], detail
