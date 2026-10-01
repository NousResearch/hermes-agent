"""Native launch/result acceptance: real publisher, no checkout-local venv."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

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
    if sys.argv[1:] == ['gateway', 'start', '--all']:
        return int(os.environ.get('GATEWAY_EXIT', '0'))
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
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n')
    home = tmp_path / 'profile'; home.mkdir()
    calls = tmp_path / 'calls.jsonl'
    command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
               str(ROOT / 'scripts/desktop-update/windows.ps1'), '-InstallRoot', str(install), '-NoUi']
    if no_gateway:
        command.append('-NoGateway')
    result = subprocess.run(
        command,
        cwd=tmp_path, env={**os.environ, 'TEMP': str(tmp_path), 'TMP': str(tmp_path),
                          'HERMES_HOME': str(home),
                          'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
                          'HANDOFF_CALLS': str(calls), 'HANDOFF_EXIT': str(code),
                          'GATEWAY_EXIT': str(gateway_code)},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == code, result.stdout + result.stderr
    expected = [{'argv': ['update', '--yes'] + ([] if no_gateway else ['--gateway'])
                 + ['--branch', 'main', '--keep-stash'], 'cwd': str(install)}]
    if code == 0 and not no_gateway:
        expected.append({'argv': ['gateway', 'start', '--all'], 'cwd': str(install)})
    if not calls.exists():
        pytest.fail(f"handoff did not invoke fixture (exit {result.returncode}):\n"
                    f"stdout:\n{result.stdout[-2500:]}\nstderr:\n{result.stderr[-3500:]}",
                    pytrace=False)
    assert [json.loads(line) for line in calls.read_text().splitlines()] == expected
    receipt = json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig'))
    assert receipt['ok'] == (code == 0)
    assert receipt.get('manual', False) == (code == 0 and not no_gateway and gateway_code != 0)
    assert not (home / '.hermes-update-in-progress').exists()


def _live_fleet_cli() -> str:
    return CLI.replace(
        "    print('Desktop build failed')",
        "    if sys.argv[1] == 'update':\n"
        "        from datetime import datetime, timezone\n"
        "        state = {'pid': int(os.environ['HANDOFF_GATEWAY_PID']), "
        "'gateway_state': 'running', 'code_sha': os.environ['HANDOFF_STATE_SHA'], "
        "'updated_at': datetime.now(timezone.utc).isoformat(), "
        "'served_profiles': ['default']}\n"
        "        (Path(os.environ['HERMES_HOME']) / 'gateway_state.json').write_text(json.dumps(state))\n"
        "        receipts = Path(os.environ['HERMES_HOME']) / 'logs' / 'update_receipts'\n"
        "        receipts.mkdir(parents=True, exist_ok=True)\n"
        "        (receipts / 'update_fixture.json').write_text(json.dumps({"
        "'outcome': 'success', 'post_update': {'sha': os.environ['HANDOFF_SHA']}}))\n"
        "    print('Desktop build failed')",
    )


@pytest.mark.platforms('windows')
def test_handoff_skips_gateway_start_when_updated_fleet_is_live(tmp_path: Path) -> None:
    install = tmp_path / 'checkout with spaces'
    source = _live_fleet_cli()
    publish_fixture_launcher(install, source)
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n')
    sha = 'a' * 40
    home = tmp_path / 'profile'; home.mkdir()
    calls = tmp_path / 'calls.jsonl'
    script_path = str(ROOT / 'scripts/desktop-update/windows.ps1').replace("'", "''")
    install_path = str(install).replace("'", "''")
    # An ISO October timestamp must remain October on pt-BR Windows. The
    # culture-dependent parse used to turn it into January and restart Gateway.
    command = (
        "[Threading.Thread]::CurrentThread.CurrentCulture = "
        "[cultureinfo]::GetCultureInfo('pt-BR'); "
        f"& '{script_path}' -InstallRoot '{install_path}' -NoUi"
    )
    result = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command],
        cwd=tmp_path,
        env={**os.environ, 'TEMP': str(tmp_path), 'TMP': str(tmp_path),
             'HERMES_HOME': str(home),
             'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
             'HANDOFF_CALLS': str(calls), 'HANDOFF_EXIT': '0', 'GATEWAY_EXIT': '1',
             'HANDOFF_GATEWAY_PID': str(os.getpid()), 'HANDOFF_SHA': sha,
             'HANDOFF_STATE_SHA': sha},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert [json.loads(line)['argv'][0] for line in calls.read_text().splitlines()] == ['update']
    receipt = json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig'))
    assert receipt['ok'] and not receipt.get('manual', False)


@pytest.mark.platforms('windows')
def test_handoff_waits_for_new_gateway_state_before_fallback(tmp_path: Path) -> None:
    install = tmp_path / 'checkout with spaces'
    publish_fixture_launcher(install, _live_fleet_cli())
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n')
    sha = 'a' * 40
    home = tmp_path / 'profile'; home.mkdir()
    calls = tmp_path / 'calls.jsonl'
    command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
               str(ROOT / 'scripts/desktop-update/windows.ps1'),
               '-InstallRoot', str(install), '-NoUi']
    env = {**os.environ, 'TEMP': str(tmp_path), 'TMP': str(tmp_path),
           'HERMES_HOME': str(home),
           'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
           'HANDOFF_CALLS': str(calls), 'HANDOFF_EXIT': '0', 'GATEWAY_EXIT': '1',
           'HANDOFF_GATEWAY_PID': str(os.getpid()), 'HANDOFF_SHA': sha,
           'HANDOFF_STATE_SHA': '0' * 40}
    proc = subprocess.Popen(command, cwd=tmp_path, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        handoff_log = home / 'logs/desktop-update-handoff.log'
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if handoff_log.exists() and 'hermes update exit code: 0' in handoff_log.read_text(errors='replace'):
                break
            if proc.poll() is not None:
                pytest.fail('handoff exited before the gateway state could be published')
            time.sleep(.05)
        else:
            pytest.fail('handoff never completed its update step')
        time.sleep(1)
        state_path = home / 'gateway_state.json'
        state = json.loads(state_path.read_text())
        state['code_sha'] = sha
        state_path.write_text(json.dumps(state))
        stdout, stderr = proc.communicate(timeout=120)
        assert proc.returncode == 0, stdout + stderr
        assert [json.loads(line)['argv'][0] for line in calls.read_text().splitlines()] == ['update']
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


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
                                capture_output=True, text=True, timeout=45)
        assert result.returncode == expected_code, result.stdout + result.stderr
        if expected_code == 0:
            assert json.loads(result.stdout) == str(external)
