"""Native launch/result acceptance: real publisher, no checkout-local venv."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher

ROOT = Path(__file__).resolve().parents[1]
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
    return int(os.environ['HANDOFF_EXIT'])
if __name__ == '__main__':
    sys.exit(main())
"""

@pytest.mark.platforms('windows')
@pytest.mark.parametrize('code', [0, 1, 2])
def test_pm_handoff_runs_once_and_reports_the_actual_result(tmp_path: Path, code: int) -> None:
    install = tmp_path / 'checkout with spaces'
    publish_fixture_launcher(install, CLI)
    (install / 'hermes_cli/desktop_update_verify.py').write_text('pass\n')
    home = tmp_path / 'profile'; home.mkdir()
    calls = tmp_path / 'calls.jsonl'
    result = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
         str(ROOT / 'scripts/desktop-update/windows.ps1'), '-InstallRoot', str(install), '-NoUi'],
        cwd=tmp_path, env={**os.environ, 'HERMES_HOME': str(home),
                          'HERMES_RUNTIME_DIR': str(tmp_path / 'empty-store'),
                          'HANDOFF_CALLS': str(calls), 'HANDOFF_EXIT': str(code)},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == code, result.stdout + result.stderr
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [
        {'argv': ['update', '--yes', '--gateway', '--branch', 'main', '--keep-stash'], 'cwd': str(install)}
    ]
    receipt = json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig'))
    assert receipt['ok'] == (code == 0)
    assert not (home / '.hermes-update-in-progress').exists()


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
