"""Native whole-script publication; POSIX setup/install lives in the fresh E2E."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from pm.environments import install_state_dir, site_packages
from pm.lock import Lockfile
from tests.hermes_cli.test_source_launcher_publication import fixture_tree

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("shell", ["powershell", "pwsh"])
def test_powershell_stage_publishes_without_a_checkout_venv(tmp_path, monkeypatch, shell):
    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    # Exercise the real pin reader, including the nested artifact objects.
    # fixture_tree already populated pm/ with the boot modules.
    shutil.copy2(ROOT / 'pm/lock.json', repo / 'pm/lock.json')
    pin = Lockfile(repo / 'pm/lock.json').version('python')
    assert pin
    py_version = '.'.join(pin.split('+')[0].split('.')[:2])
    calls = home / 'uv-calls'
    selected = install_state_dir(repo) / 'environments/ready/venv'
    site = site_packages(selected)
    site.mkdir(parents=True)
    (selected / 'pyvenv.cfg').write_text('home = fixture\n', encoding='utf-8')
    (site / 'selected_probe.py').write_text('VALUE = 11\n', encoding='utf-8')
    (install_state_dir(repo) / 'facts.json').write_text(
        json.dumps({'packages': {'venv': {'environment': str(selected)}}}), encoding='utf-8')
    wrapper = tmp_path / 'stage.ps1'
    wrapper.write_text('''$ErrorActionPreference = 'Stop'
. $env:PROBE_INSTALLER -InstallDir $env:PROBE_REPO -HermesHome $env:PROBE_HOME
Initialize-ResolvedPaths
# Replace acquisition only; Get-BootstrapPython and Publish-UserCommand stay real.
function Get-Uv { return 'Invoke-FixtureUv' }
function Invoke-FixtureUv {
    $call = $args -join ' '
    Add-Content -LiteralPath $env:PROBE_UV_CALLS -Encoding UTF8 -Value $call
    switch -Exact ($call) {
        "python install --no-bin $env:PROBE_PY_VERSION" {
            if (-not (Test-Path -LiteralPath $env:PROBE_PYTHON -PathType Leaf)) { throw 'missing fixture Python' }
        }
        "python find --managed-python --no-project $env:PROBE_PY_VERSION" {
            Write-Output $env:PROBE_PYTHON
        }
        default { throw "unexpected bootstrap uv call: $call" }
    }
    $global:LASTEXITCODE = 0
}
function Invoke-WebRequest { throw 'unexpected bootstrap download' }
# Replace only the registry publication edge, never mutate the actual user PATH.
function Set-LauncherUserPath([string]$binDir) {
    if ($binDir -ne (Join-Path $env:PROBE_HOME 'bin')) { throw 'wrong user PATH target' }
    $script:publishedPath = $binDir
    Write-Output 'REACHED_PATH_PUBLICATION'
}
Publish-UserCommand
if (-not $script:publishedPath) { throw 'registry-publication seam was bypassed' }
exit 0
''', encoding='utf-8-sig')
    powershell = (Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
                  if shell == 'powershell' else Path(shutil.which('pwsh') or pytest.fail('native lane requires PowerShell 7')))
    env = dict(os.environ, PROBE_INSTALLER=str(ROOT / 'scripts/install.ps1'),
               PROBE_REPO=str(repo), PROBE_HOME=str(home), PROBE_PYTHON=str(interpreter),
               HERMES_HOME=str(tmp_path / 'other-home'),
               PROBE_PY_VERSION=py_version, PROBE_UV_CALLS=str(calls),
               UV_OFFLINE='1', UV_PYTHON_DOWNLOADS='never')
    env['PATH'] = os.pathsep.join([str(powershell.parent), str(Path(os.environ['SystemRoot']) / 'System32')])
    env['PATHEXT'] = '.COM;.EXE;.BAT;.CMD'
    env.pop('UV_PYTHON_INSTALL_DIR', None)
    result = subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(wrapper)],
                            cwd=tmp_path, env=env, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert b'REACHED_PATH_PUBLICATION' in result.stdout
    assert calls.read_text(encoding='utf-8-sig').splitlines() == [
        f'python install --no-bin {py_version}',
        f'python find --managed-python --no-project {py_version}',
    ]
    for name in ('hermes', 'hermes-acp'):
        command = home / 'bin' / (name + ('.exe' if (home / 'bin' / (name + '.exe')).is_file() else '.cmd'))
        child_env = dict(env)
        child_env.pop('HERMES_HOME', None)
        child = subprocess.run([str(command), 'from-powershell'], cwd=tmp_path, env=child_env,
                               capture_output=True, text=True, encoding='utf-8', timeout=30)
        assert child.returncode == 7, child.stdout + child.stderr
        witness = json.loads(child.stdout)
        assert witness['value'] == 11 and witness['argv'] == ['from-powershell']
        assert Path(witness['home']) == home and Path(witness['exe']).samefile(interpreter)
    assert not (repo / 'venv').exists()
