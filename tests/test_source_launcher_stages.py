"""Bootstrap stages publish launchers through the shared writer after PM setup."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hermes_cli.runtime_paths import install_state_dir, site_packages
from pm.lock import Lockfile
from pm.store import current_target
from tests.hermes_cli.test_source_launcher_publication import fixture_tree

ROOT = Path(__file__).resolve().parents[1]


def selected_environment(repo, value=11):
    selected = install_state_dir(repo) / 'environments/ready/venv'
    site = site_packages(selected)
    site.mkdir(parents=True)
    (selected / 'pyvenv.cfg').write_text('home = fixture\n', encoding='utf-8')
    (site / 'selected_probe.py').write_text(f'VALUE = {value}\n', encoding='utf-8')
    (install_state_dir(repo) / 'facts.json').write_text(
        json.dumps({'packages': {'venv': {'environment': str(selected)}}}), encoding='utf-8')


@pytest.mark.platforms("windows", "posix")
def test_developer_setup_publishes_without_a_checkout_venv(tmp_path, monkeypatch):
    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    shutil.copy2(ROOT / 'setup-hermes.sh', repo / 'setup-hermes.sh')
    store = home / 'tools'
    uv = Path(shutil.which('uv') or pytest.fail('canonical test environment needs uv'))
    version = Lockfile(ROOT / 'pm/lock.json').version('uv')
    entry = store / f'uv-{version}-{current_target()}'
    entry.mkdir()
    shutil.copy2(uv, entry / ('uv.exe' if os.name == 'nt' else 'uv'))
    (repo / 'pm').mkdir()
    (repo / 'pm/__init__.py').write_text('', encoding='utf-8')
    # Stop at PM's install boundary; the launcher writer and boot selection stay real.
    (repo / 'pm/cli.py').write_text('', encoding='utf-8')
    lock = Lockfile(repo / 'pm/lock.json')
    lock.set_pin('uv', version, {})
    lock.set_pin('python', '.'.join(map(str, sys.version_info[:2])), {})
    lock.save()
    selected_environment(repo)
    env = dict(os.environ, HOME=str(tmp_path / 'shell-home'), HERMES_HOME=str(home),
               HERMES_RUNTIME_DIR=str(store), UV_OFFLINE='1', UV_PYTHON_DOWNLOADS='never')
    env.pop('UV_PYTHON_INSTALL_DIR', None)
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    Path(env['HOME']).mkdir()
    result = subprocess.run(['bash', str(repo / 'setup-hermes.sh')], cwd=tmp_path, env=env,
                            capture_output=True, timeout=90)
    if result.returncode:
        print(result.stdout.decode('utf-8', errors='replace'))
        print(result.stderr.decode('utf-8', errors='backslashreplace'))
    assert result.returncode == 0, result.stdout + result.stderr
    out = home / 'bin' if os.name == 'nt' else Path(env['HOME']) / '.local/bin'
    launchers = list(out.glob('hermes*')) if out.exists() else []
    assert launchers, result.stdout + result.stderr
    command = out / ('hermes.exe' if (out / 'hermes.exe').is_file() else 'hermes.cmd' if os.name == 'nt' else 'hermes')
    child_env = dict(env)
    child_env.pop('HERMES_HOME')
    child_env.pop('HERMES_RUNTIME_DIR')
    child = subprocess.run([str(command), 'literal input'], cwd=tmp_path, env=child_env,
                           capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert child.returncode == 7, child.stdout + child.stderr
    witness = json.loads(child.stdout)
    assert witness['value'] == 11 and witness['argv'] == ['literal input']
    assert Path(witness['home']) == home and Path(witness['exe']).samefile(interpreter)
    assert not (repo / 'venv').exists()


@pytest.mark.platforms("windows", "posix")
def test_shell_installer_path_stage_uses_shared_publication(tmp_path, monkeypatch):
    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    (repo / 'pm').mkdir()
    lock = Lockfile(repo / 'pm/lock.json')
    lock.set_pin('python', '.'.join(map(str, sys.version_info[:2])), {})
    lock.save()
    selected_environment(repo)
    shell_home = tmp_path / 'shell-home'
    shell_home.mkdir()
    env = dict(os.environ, PROBE_SCRIPT=(ROOT / 'scripts/install.sh').as_posix(),
               PROBE_REPO=repo.as_posix(), PROBE_HOME=home.as_posix(),
               PROBE_SHELL_HOME=shell_home.as_posix(), UV_OFFLINE='1', UV_PYTHON_DOWNLOADS='never')
    env.pop('UV_PYTHON_INSTALL_DIR', None)
    script = '''source "$PROBE_SCRIPT" --manifest --dir "$PROBE_REPO" --hermes-home "$PROBE_HOME"
HOME="$PROBE_SHELL_HOME"
JSON=true
run_stage path
'''
    # Source the stage to test its transport on Windows without pretending it is POSIX.
    result = subprocess.run(['bash', '-c', script], cwd=tmp_path, env=env,
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = [json.loads(line) for line in result.stdout.splitlines() if line.startswith(b'{')]
    assert frames == [{'ok': True, 'stage': 'path', 'skipped': False}]
    out = shell_home / '.local/bin'
    for name in ('hermes', 'hermes-acp'):
        command = out / (name + '.exe' if (out / (name + '.exe')).is_file()
                         else name + '.cmd' if os.name == 'nt' else name)
        child_env = dict(env)
        child_env.pop('HERMES_HOME', None)
        child = subprocess.run([str(command), 'from-stage'], cwd=tmp_path, env=child_env,
                               capture_output=True, text=True, encoding='utf-8', timeout=30)
        assert child.returncode == 7, child.stdout + child.stderr
        witness = json.loads(child.stdout)
        assert witness['value'] == 11 and witness['argv'] == ['from-stage']
        assert Path(witness['home']) == home and Path(witness['exe']).samefile(interpreter)
    assert not (repo / 'venv').exists()


@pytest.mark.platforms("windows")
def test_powershell_stage_publishes_without_a_checkout_venv(tmp_path, monkeypatch):
    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    (repo / 'pm').mkdir()
    lock = Lockfile(repo / 'pm/lock.json')
    lock.set_pin('python', '.'.join(map(str, sys.version_info[:2])), {})
    lock.save()
    selected_environment(repo)
    wrapper = tmp_path / 'stage.ps1'
    wrapper.write_text('''$ErrorActionPreference = 'Stop'
. $env:PROBE_INSTALLER -InstallDir $env:PROBE_REPO -HermesHome $env:PROBE_HOME
Initialize-ResolvedPaths
# Replace only the registry publication edge, never mutate the actual user PATH.
function Set-LauncherUserPath([string]$binDir) {
    if ($binDir -ne (Join-Path $env:PROBE_HOME 'bin')) { throw 'wrong user PATH target' }
    $script:publishedPath = $binDir
    Write-Output 'REACHED_PATH_PUBLICATION'
}
Stage-Path
if (-not $script:publishedPath) { throw 'registry-publication seam was bypassed' }
exit 0
''', encoding='utf-8-sig')
    powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    env = dict(os.environ, PROBE_INSTALLER=str(ROOT / 'scripts/install.ps1'),
               PROBE_REPO=str(repo), PROBE_HOME=str(home), HERMES_HOME=str(tmp_path / 'other-home'),
               UV_OFFLINE='1', UV_PYTHON_DOWNLOADS='never')
    uv = Path(shutil.which('uv') or pytest.fail('canonical test environment needs uv'))
    env['PATH'] = os.pathsep.join([str(uv.parent), str(powershell.parent), str(Path(os.environ['SystemRoot']) / 'System32')])
    env['PATHEXT'] = '.COM;.EXE;.BAT;.CMD'
    env.pop('UV_PYTHON_INSTALL_DIR', None)
    result = subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(wrapper)],
                            cwd=tmp_path, env=env, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert b'REACHED_PATH_PUBLICATION' in result.stdout
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
