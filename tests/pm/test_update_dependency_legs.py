"""Optional dependency refreshes run in PM's repository with its installed tools."""
from argparse import Namespace
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

import pytest

from pm import cli, paths, registry
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.packages import Nodejs, Npm, Python, Uv
from pm.store import current_target


class ManualFixture(Package):
    name = 'manual-fixture'


def project(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'pyproject.toml').write_text(
        '[project]\nname="update-proof"\nversion="1"\nrequires-python=">=3.14"\n'
        '[tool.uv]\npackage=false\n', encoding='utf-8')
    caller = tmp_path / 'unrelated-project'
    caller.mkdir()
    (caller / 'pyproject.toml').write_text('not a project', encoding='utf-8')
    lock = Lockfile(repo / 'pm/lock.json')
    lock.set_pin('manual-fixture', '1', {})
    lock.save()
    monkeypatch.setitem(registry._packages, 'manual-fixture', ManualFixture())
    monkeypatch.setattr(paths, 'repo_root', lambda: repo)
    monkeypatch.setattr(cli, 'repo_root', lambda: repo)
    monkeypatch.setattr(paths, 'lockfile_path', lambda: lock.path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'store'))
    monkeypatch.chdir(caller)

    run_live = cli._run_live

    def run_owned(cmd, *, cwd, env, **kwargs):
        assert Path(cwd).resolve() == repo.resolve(), 'refuse a dependency command outside the temporary project'
        assert Path(cmd[0]).resolve().is_relative_to(tmp_path.resolve()), 'refuse a tool outside the temporary store'
        return run_live(cmd, cwd=cwd, env=env, **kwargs)

    monkeypatch.setattr(cli, '_run_live', run_owned)
    return repo, caller, lock


@pytest.mark.platforms('windows', 'posix')
def test_uv_refresh_uses_real_installed_tool_and_only_the_owned_project(tmp_path, monkeypatch, capsys):
    repo, caller, lock = project(tmp_path, monkeypatch)
    uv = Path(shutil.which('uv') or pytest.fail('canonical test environment requires uv'))
    runtime = tmp_path / 'store'
    managed = runtime / 'uv-fixture' / uv.name
    managed.parent.mkdir(parents=True)
    shutil.copy2(uv, managed)
    python = Path(sys._base_executable).resolve()
    python_root = python.parent if os.name == 'nt' else python.parents[1]
    target = current_target()
    facts = Facts(runtime / 'facts.json')
    for name, package, entry in [('uv', Uv(), managed.parent), ('python', Python(), python_root)]:
        monkeypatch.setitem(registry._packages, name, package)
        lock.set_pin(name, 'fixture', {target: {'url': f'https://example.invalid/{name}', 'sha256': '1' * 64}})
        facts.record(name, 'fixture', str(entry), package.env(entry, target), runtime,
                     target=target, artifacts=['1' * 64])
    lock.save()
    before_lock = lock.path.read_bytes()
    parent_env = dict(os.environ)
    syncs = []
    ensure = importlib.import_module('pm.ensure')
    monkeypatch.setattr(ensure, 'sync_venv', lambda **kw: syncs.append(kw))
    args = Namespace(names=['manual-fixture'], target=None, check=True, uv=True, npm=False)
    assert cli.cmd_update(args) == 0
    assert not (repo / 'uv.lock').exists()
    assert not syncs
    check_output = capsys.readouterr().out
    args.check = False
    assert cli.cmd_update(args) == 0
    assert 'uv lock --upgrade' in check_output
    assert (repo / 'uv.lock').is_file()
    assert not (caller / 'uv.lock').exists()
    assert syncs == [{'explicit': True}]
    assert lock.path.read_bytes() == before_lock
    assert dict(os.environ) == parent_env

    original = (repo / 'uv.lock').read_bytes()
    syncs.clear()
    (repo / 'pyproject.toml').write_text('invalid project [', encoding='utf-8')
    assert cli.cmd_update(args) == 1
    assert (repo / 'uv.lock').read_bytes() == original and not syncs
    assert 'uv lock --upgrade failed' in capsys.readouterr().out

    managed.unlink()
    assert cli.cmd_update(args) == 1
    assert (repo / 'uv.lock').read_bytes() == original and not syncs
    assert 'not installed' in capsys.readouterr().out


@pytest.mark.platforms('windows', 'posix')
def test_npm_refresh_uses_its_installed_entry_and_owned_project(monkeypatch, capsys):
    node_source = Path(shutil.which('node') or pytest.fail('node is required'))
    npm_source = Path(shutil.which('npm.cmd' if os.name == 'nt' else 'npm') or pytest.fail('npm is required')).resolve()
    npm_source = npm_source.parent / 'node_modules/npm' if os.name == 'nt' else npm_source.parents[1]
    assert (npm_source / 'bin/npm-cli.js').is_file()
    # Keep the real tool's argv clear of the harness's hermes-update guard.
    temp_root = Path(os.environ['LOCALAPPDATA']) / 'Temp' if os.name == 'nt' else Path('/tmp')
    with tempfile.TemporaryDirectory(prefix='pm-deps-', dir=temp_root) as temporary, monkeypatch.context() as scoped:
        monkeypatch = scoped
        root = Path(temporary)
        repo, caller, lock = project(root, monkeypatch)
        user = root / 'user'
        user.mkdir()
        for key in ('HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA'):
            monkeypatch.setenv(key, str(user))
        target = current_target()
        runtime = root / 'store'
        node_entry = runtime / 'node-fixture'
        node_binary = node_entry / ('node.exe' if os.name == 'nt' else 'bin/node')
        node_binary.parent.mkdir(parents=True)
        shutil.copy2(node_source, node_binary)
        bundled_npm = node_entry / ('node_modules/npm' if os.name == 'nt' else 'lib/node_modules/npm')
        shutil.copytree(npm_source, bundled_npm)
        facts = Facts(runtime / 'facts.json')
        node, npm = Nodejs(), Npm()
        monkeypatch.setitem(registry._packages, 'node', node)
        monkeypatch.setitem(registry._packages, 'npm', npm)
        lock.set_pin('node', 'fixture', {target: {'url': 'https://example.invalid/node', 'sha256': '1' * 64}})
        facts.record('node', 'fixture', str(node_entry), node.env(node_entry, target), runtime,
                     target=target, artifacts=['1' * 64])
        lock.save()
        archive = root / 'npm.tgz'
        with tarfile.open(archive, 'w:gz') as output:
            output.add(npm_source, arcname='package')
        npm_entry = runtime / 'npm-fixture'
        npm.unpack(archive, npm_entry, target)
        version = json.loads((npm_source / 'package.json').read_text(encoding='utf-8'))['version']
        lock.set_pin('npm', version, {target: {'url': 'https://example.invalid/npm', 'sha256': '2' * 64}})
        facts.record('npm', version, str(npm_entry), npm.env(npm_entry, target), runtime,
                     target=target, artifacts=['2' * 64])
        lock.save()
        # After install, Node's bundled npm is no longer available as a fallback.
        shutil.rmtree(bundled_npm)
        (repo / 'package.json').write_text(json.dumps({'name': 'pm-leg-proof', 'version': '1.0.0', 'private': True}), encoding='utf-8')
        (repo / '.npmrc').write_text('offline=true\naudit=false\nfund=false\nignore-scripts=true\n', encoding='utf-8')
        monkeypatch.setenv('PATH', str(Path(os.environ.get('SYSTEMROOT', '/')) / 'System32') if os.name == 'nt' else '/usr/bin:/bin')
        monkeypatch.setenv('NODE_OPTIONS', '--invalid-option-to-be-scrubbed')
        monkeypatch.setenv('npm_config_cache', str(root / 'ambient-cache'))
        before = dict(os.environ)
        before_lock = lock.path.read_bytes()
        args = Namespace(names=['manual-fixture'], target=None, check=True, uv=False, npm=True)
        assert cli.cmd_update(args) == 0
        assert not (repo / 'package-lock.json').exists()
        args.check = False
        assert cli.cmd_update(args) == 0, capsys.readouterr().out
        assert json.loads((repo / 'package-lock.json').read_text(encoding='utf-8'))['name'] == 'pm-leg-proof'
        assert not (caller / 'package-lock.json').exists()
        assert dict(os.environ) == before
        assert lock.path.read_bytes() == before_lock
        assert not (root / 'ambient-cache').exists()
        lock_bytes = (repo / 'package-lock.json').read_bytes()
        (repo / 'package.json').write_text('not json', encoding='utf-8')
        assert cli.cmd_update(args) == 1
        assert (repo / 'package-lock.json').read_bytes() == lock_bytes
        assert 'npm update failed' in capsys.readouterr().out

        node_binary.rename(node_binary.with_suffix('.held'))
        assert cli.cmd_update(args) == 1
        assert (repo / 'package-lock.json').read_bytes() == lock_bytes
        assert 'not installed' in capsys.readouterr().out
        node_binary.with_suffix('.held').rename(node_binary)
        npm.binary(npm_entry, target).unlink()
        assert cli.cmd_update(args) == 1
        assert (repo / 'package-lock.json').read_bytes() == lock_bytes
        assert 'not installed' in capsys.readouterr().out
        assert dict(os.environ) == before and lock.path.read_bytes() == before_lock
