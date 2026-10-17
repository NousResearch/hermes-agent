"""Native launch/result acceptance: real publisher, no checkout-local venv."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CLI = """
import json, os, sys
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    with Path(os.environ['HANDOFF_CALLS']).open('a') as stream:
        stream.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')
    print('Desktop build failed')  # no warning-driven second build on PM
    if sys.argv[1:] == ['gateway', 'start']:
        return int(os.environ.get('GATEWAY_EXIT', '0'))
    if os.environ.get('HANDOFF_HANG'):
        # Print, then park silently: the shape the hand-off's idle watchdog kills.
        sys.stdout.reconfigure(encoding='utf-8')  # as the real CLI does
        print(os.environ['HANDOFF_HANG'].replace('|', '\\n'), flush=True)
        import time; time.sleep(300)
    return int(os.environ['HANDOFF_EXIT'])
if __name__ == '__main__':
    sys.exit(main())
"""

@pytest.mark.platforms('windows')
@pytest.mark.parametrize(
    ('code', 'no_gateway', 'gateway_code'),
    [(0, False, 0), (1, False, 0), (2, False, 0), (0, True, 0), (0, False, 1)],
)
def test_pm_handoff_reports_update_and_gateway_results(
    tmp_path: Path, code: int, no_gateway: bool, gateway_code: int,
) -> None:
    install = tmp_path / 'checkout with spaces'
    publish_fixture_launcher(install, CLI)
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n', encoding='utf-8')
    home = tmp_path / 'profile'; home.mkdir()
    calls = tmp_path / 'calls.jsonl'
    command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
               str(ROOT / 'scripts/desktop-update/windows.ps1'), '-InstallRoot', str(install), '-NoUi']
    if no_gateway:
        command.append('-NoGateway')
    result = subprocess.run(
        command,
        cwd=tmp_path, env={**os.environ, 'HERMES_HOME': str(home),
                          'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
                          'HANDOFF_CALLS': str(calls), 'HANDOFF_EXIT': str(code),
                          'GATEWAY_EXIT': str(gateway_code)},
        capture_output=True, text=True, timeout=120,
        check=False,
    )
    assert result.returncode == code, result.stdout + result.stderr
    expected = [{'argv': ['update', '--yes'] + ([] if no_gateway else ['--gateway'])
                 + ['--branch', 'main', '--keep-stash'], 'cwd': str(install)}]
    if code == 0 and not no_gateway:
        expected.append({'argv': ['gateway', 'start'], 'cwd': str(install)})
    assert [json.loads(line) for line in calls.read_text(encoding='utf-8-sig').splitlines()] == expected
    receipt = json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig'))
    assert receipt['ok'] == (code == 0)
    # The fixture publishes no live state: even a zero-exit launch is not readiness.
    assert receipt.get('manual', False) == (code == 0 and not no_gateway)
    assert not (home / '.hermes-update-in-progress').exists()


@pytest.mark.platforms('windows')
@pytest.mark.parametrize(('output', 'code'), [
    ('✓ Update complete! (v1.0.0)', 0),
    ('✓ Update complete! (v1.0.0)|  ✗ hermes-gateway failed to come back after restart.', 124),
    ('✓ Update complete! (v1.0.0)|Update incomplete — some units were not restarted', 124),
    ('Cloning into the checkout...', 124),
])
def test_update_killed_by_idle_watchdog_after_completing_is_a_success(
    tmp_path: Path, output: str, code: int,
) -> None:
    """#96205: a finished update parked silently in its restart phase is still finished,
    unless anything after its banner already reported a failure."""
    install = tmp_path / 'checkout'
    publish_fixture_launcher(install, CLI)
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n', encoding='utf-8')
    home = tmp_path / 'profile'; home.mkdir()
    calls = tmp_path / 'calls.jsonl'
    result = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
         str(ROOT / 'scripts/desktop-update/windows.ps1'), '-InstallRoot', str(install), '-NoUi'],
        cwd=tmp_path, env={**os.environ, 'HERMES_HOME': str(home),
                          'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
                          'HANDOFF_CALLS': str(calls), 'HANDOFF_EXIT': '0', 'HANDOFF_HANG': output,
                          'HERMES_UPDATE_STEP_IDLE_SECONDS': '3', 'PYTHONIOENCODING': 'utf-8'},
        capture_output=True, text=True, timeout=180,
        check=False,
    )
    assert result.returncode == code, result.stdout + result.stderr
    argv = [json.loads(line)['argv'][:2] for line in calls.read_text(encoding='utf-8-sig').splitlines()]
    assert argv == ([['update', '--yes'], ['gateway', 'start']] if code == 0 else [['update', '--yes']])
    receipt = json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig'))
    assert receipt['ok'] == (code == 0)


@pytest.mark.platforms('windows')
def test_earlier_pm_userbin_launcher_is_identity_checked(tmp_path: Path) -> None:
    home = tmp_path / 'profile'
    userbin = home / 'bin'; userbin.mkdir(parents=True)
    root = tmp_path / 'source'
    launcher = publish_fixture_launcher(root, CLI)
    external = userbin / launcher.name
    launcher.rename(external)
    wrong = tmp_path / 'other'
    (wrong / 'pm').mkdir(parents=True)
    (wrong / 'hermes_cli').mkdir()
    (wrong / 'hermes_cli/_launchers.py').touch()
    helper = str(ROOT / 'scripts/desktop-update/runtime.ps1').replace("'", "''")
    for target, expected_code in [(root, 0), (wrong, 1)]:
        script = f". '{helper}'; try {{ @(Get-HermesRuntimeCommand -InstallRoot '{target}') | ConvertTo-Json -Compress }} catch {{ exit 1 }}"
        result = subprocess.run(['powershell', '-NoProfile', '-Command', script],
                                env={**os.environ, 'HERMES_HOME': str(home)},
                                capture_output=True, text=True, timeout=45, check=False)
        assert result.returncode == expected_code, result.stdout + result.stderr
        if expected_code == 0:
            assert json.loads(result.stdout) == str(external)


@pytest.mark.platforms('windows')
def test_command_file_launcher_probe_runs_through_cmd_exe_explicitly(tmp_path: Path) -> None:
    """A .cmd launcher probe goes through `cmd.exe /d /s /c` exactly like the update step,
    never through CreateProcess's implicit batch handling; unquotable arguments are refused."""
    bin_dir = tmp_path / 'launcher dir'; bin_dir.mkdir()
    launcher = bin_dir / 'hermes.cmd'
    launcher.write_text('@echo off\r\necho CMDLINE=%CMDCMDLINE%\r\necho ARGS=%*\r\n', encoding='ascii')
    helper = str(ROOT / 'scripts/desktop-update/runtime.ps1').replace("'", "''")
    target = str(launcher).replace("'", "''")

    def probe(argument: str) -> subprocess.CompletedProcess:
        script = (f". '{helper}'; try {{ $r = Invoke-HermesProbe '{target}' @('--version', '{argument}'); "
                  "[Console]::Out.Write(\"code=$($r.Code)`n$($r.Output)\") } catch { [Console]::Out.Write('refused'); exit 3 }")
        return subprocess.run(['powershell', '-NoProfile', '-Command', script],
                              capture_output=True, text=True, timeout=60, check=False)

    ok = probe('two words')
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert 'code=0' in ok.stdout, ok.stdout
    cmdline = next(line for line in ok.stdout.splitlines() if line.startswith('CMDLINE='))
    assert ' /d /s /c ' in cmdline.lower(), cmdline
    assert 'ARGS="--version" "two words"' in ok.stdout, ok.stdout
    bad = probe('100%PATH%')
    assert (bad.returncode, bad.stdout) == (3, 'refused'), bad.stdout + bad.stderr
