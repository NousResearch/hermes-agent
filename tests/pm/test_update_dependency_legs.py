"""Optional dependency refreshes run in PM's repository with its installed tools."""
from argparse import Namespace
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace

import pytest

from pm import cli, paths, registry
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.packages import Nodejs, Pnpm, Python, Uv
from pm.store import current_target, tree_digest


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
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'home')
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


def _shared_prefix(root: Path, *, limit: int = 50_000) -> str:
    """Why ``root`` cannot be digested as a standalone Python package entry, or ''.

    The fixture records the host interpreter's prefix as PM's Python entry,
    which means reading every file under it. A distro prefix (/usr, /usr/local,
    Homebrew) holds unrelated, sometimes unreadable, trees; a standalone or
    self-contained build holds one interpreter.
    """
    count = 0
    for directory, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(directory) / name
            count += 1
            if count > limit:
                return f'more than {limit} files'
            if not path.is_symlink() and not os.access(path, os.R_OK):
                return f'{path} is unreadable'
    return ''


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
    shared = _shared_prefix(python_root)
    if shared:
        pytest.skip(f"host Python prefix {python_root} is not self-contained: {shared}")
    target = current_target()
    facts = Facts(runtime / 'facts.json')
    for name, package, entry in [('uv', Uv(), managed.parent), ('python', Python(), python_root)]:
        monkeypatch.setitem(registry._packages, name, package)
        lock.set_pin(name, 'fixture', {target: {'url': f'https://example.invalid/{name}', 'sha256': '1' * 64}})
        facts.record(name, 'fixture', str(entry), package.env(entry, target), runtime,
                     target=target, artifacts=['1' * 64], digest=tree_digest(entry))
    lock.save()
    before_lock = lock.path.read_bytes()
    parent_env = dict(os.environ)
    syncs = []
    ensure = importlib.import_module('pm.install')
    monkeypatch.setattr(ensure, 'sync_venv', lambda **kw: syncs.append(kw))
    args = Namespace(names=['manual-fixture'], target=None, skip_target=[], check=True, uv=True, pnpm=False, termux=False)
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
    assert 'Python lock refresh failed' in capsys.readouterr().out

    managed.unlink()
    assert cli.cmd_update(args) == 1
    assert (repo / 'uv.lock').read_bytes() == original and not syncs
    assert 'Python lock refresh failed' in capsys.readouterr().out


@contextmanager
def pnpm_project(monkeypatch):
    """A temporary project whose PM-installed pnpm is the real tool; yields what the tests check."""
    pnpm_tool = Path(shutil.which('pnpm') or pytest.fail('canonical test environment requires pnpm'))
    # Keep the real tool's argv clear of the harness's hermes-update guard.
    temp_root = Path(os.environ['LOCALAPPDATA']) / 'Temp' if os.name == 'nt' else Path('/tmp')
    with tempfile.TemporaryDirectory(prefix='pm-deps-', dir=temp_root) as temporary, monkeypatch.context() as scoped:
        monkeypatch = scoped
        root = Path(temporary)
        repo, caller, lock = project(root, monkeypatch)
        user = root / 'user'
        user.mkdir()
        for key in ('HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'XDG_STATE_HOME'):
            monkeypatch.setenv(key, str(user))
        target = current_target()
        runtime = root / 'store'
        node_entry = runtime / 'node-fixture'
        node_binary = node_entry / ('node.exe' if os.name == 'nt' else 'bin/node')
        node_binary.parent.mkdir(parents=True)
        node_binary.write_bytes(b'MZ' if os.name == 'nt' else b'#!/bin/sh\nexit 99\n')
        if os.name != 'nt':
            node_binary.chmod(0o755)
        facts = Facts(runtime / 'facts.json')
        node, pnpm = Nodejs(), Pnpm()
        monkeypatch.setitem(registry._packages, 'node', node)
        monkeypatch.setitem(registry._packages, 'pnpm', pnpm)
        lock.set_pin('node', 'fixture', {target: {'url': 'https://example.invalid/node', 'sha256': '1' * 64}})
        facts.record('node', 'fixture', str(node_entry), node.env(node_entry, target), runtime,
                     target=target, artifacts=['1' * 64])
        # The real executable: a native binary that needs no node to refresh a lockfile.
        pnpm_entry = runtime / 'pnpm-fixture'
        pnpm_binary = pnpm.binary(pnpm_entry, target)
        pnpm_binary.parent.mkdir(parents=True)
        # A developer run finds PM's own pnpm under the real Hermes home, which a test may not read
        # (copy) or resolve (symlink): on POSIX the fixture execs it, so only the child touches it.
        if os.name == 'nt':
            shutil.copy2(pnpm_tool.resolve(), pnpm_binary)
        else:
            pnpm_binary.write_text(f'#!/bin/sh\nexec {shlex.quote(str(pnpm_tool))} "$@"\n', encoding='utf-8')
            pnpm_binary.chmod(0o755)
        lock.set_pin('pnpm', 'fixture', {target: {'url': 'https://example.invalid/pnpm', 'sha256': '2' * 64}})
        facts.record('pnpm', 'fixture', str(pnpm_entry), pnpm.env(pnpm_entry, target), runtime,
                     target=target, artifacts=['2' * 64])
        lock.save()
        yield SimpleNamespace(monkeypatch=monkeypatch, root=root, repo=repo, caller=caller, lock=lock, user=user, pnpm_binary=pnpm_binary)


@pytest.mark.platforms('windows', 'posix')
def test_pnpm_refresh_uses_its_installed_entry_and_owned_project(monkeypatch, capsys):
    with pnpm_project(monkeypatch) as ctx:
        root, repo, caller, lock, user, pnpm_binary = ctx.root, ctx.repo, ctx.caller, ctx.lock, ctx.user, ctx.pnpm_binary
        monkeypatch = ctx.monkeypatch
        (repo / 'package.json').write_text(json.dumps({'name': 'pm-leg-proof', 'version': '1.0.0', 'private': True}), encoding='utf-8')
        # An exclude for a version the refreshed lock does not hold.
        (repo / 'pnpm-workspace.yaml').write_text(
            'minimumReleaseAge: 20160\nminimumReleaseAgeExclude:\n  - unlocked-fixture@1.0.0  # why\n', encoding='utf-8')
        monkeypatch.setenv('PATH', str(Path(os.environ.get('SYSTEMROOT', '/')) / 'System32') if os.name == 'nt' else '/usr/bin:/bin')
        monkeypatch.setenv('pnpm_config_store_dir', str(root / 'ambient-store'))
        before = dict(os.environ)
        before_lock = lock.path.read_bytes()
        args = Namespace(names=['manual-fixture'], target=None, skip_target=[], check=True, uv=False, pnpm=True, termux=False)
        assert cli.cmd_update(args) == 0
        assert not (repo / 'pnpm-lock.yaml').exists()
        args.check = False
        assert cli.cmd_update(args) == 0, capsys.readouterr().out
        assert "lockfileVersion: '9.0'" in (repo / 'pnpm-lock.yaml').read_text(encoding='utf-8')
        # The refresh leaves no stale exclude for the next install to remove.
        assert 'unlocked-fixture' not in (repo / 'pnpm-workspace.yaml').read_text(encoding='utf-8')
        assert not (caller / 'pnpm-lock.yaml').exists()
        assert dict(os.environ) == before
        assert lock.path.read_bytes() == before_lock
        # pnpm's store is Hermes-owned under the Hermes root: neither the ambient setting nor the user's dirs.
        assert (root / 'home' / 'cache' / 'pnpm' / 'store').is_dir()
        assert not (root / 'ambient-store').exists()
        # pnpm's per-user lock dirs follow the LocalAppData known folder on Windows, which
        # no pnpm setting moves (on POSIX they go to $XDG_RUNTIME_DIR or /tmp).
        allowed = r'AppData/Local/pnpm-[a-z-]+-locks/[^/]+' if os.name == 'nt' else r'(?!)'
        files = [path.relative_to(user).as_posix() for path in user.rglob('*') if path.is_file()]
        assert [path for path in files if not re.fullmatch(allowed, path)] == []
        assert os.name == 'nt' or list(user.iterdir()) == []
        lock_bytes = (repo / 'pnpm-lock.yaml').read_bytes()
        (repo / 'package.json').write_text('not json', encoding='utf-8')
        assert cli.cmd_update(args) == 1
        assert (repo / 'pnpm-lock.yaml').read_bytes() == lock_bytes
        assert 'pnpm update failed' in capsys.readouterr().out

        pnpm_binary.unlink()
        assert cli.cmd_update(args) == 1
        assert (repo / 'pnpm-lock.yaml').read_bytes() == lock_bytes
        assert 'not installed' in capsys.readouterr().out
        assert dict(os.environ) == before and lock.path.read_bytes() == before_lock


@contextmanager
def young_release_registry():
    """A registry on loopback serving `aging-fixture` 1.0.0 (2020) and 1.1.0 (published now)."""
    published = {'1.0.0': '2020-01-01T00:00:00.000Z',
                 '1.1.0': datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S.000Z')}

    class Registry(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.strip('/') != 'aging-fixture':
                self.send_response(404)
                self.end_headers()
                return
            url = f'http://127.0.0.1:{self.server.server_port}/aging-fixture/-/aging-fixture'
            body = json.dumps({
                'name': 'aging-fixture', 'dist-tags': {'latest': '1.1.0'},
                'versions': {v: {'name': 'aging-fixture', 'version': v,
                                 'dist': {'tarball': f'{url}-{v}.tgz', 'integrity': 'sha512-' + 'A' * 86 + '=='}}
                             for v in published},
                'time': {'created': published['1.0.0'], 'modified': published['1.1.0'], **published},
            }).encode()
            self.send_response(200)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(('127.0.0.1', 0), Registry) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/'
        finally:
            server.shutdown()


@pytest.mark.platforms('windows', 'posix')
def test_pnpm_refresh_prunes_unlocked_excludes_and_names_too_young_picks(monkeypatch, capsys):
    with pnpm_project(monkeypatch) as ctx, young_release_registry() as registry_url:
        repo, monkeypatch = ctx.repo, ctx.monkeypatch
        monkeypatch.setenv('PATH', str(Path(os.environ.get('SYSTEMROOT', '/')) / 'System32') if os.name == 'nt' else '/usr/bin:/bin')
        (repo / '.npmrc').write_text(f'registry={registry_url}\n', encoding='utf-8')
        manifest = {'name': 'pm-age-proof', 'version': '1.0.0', 'private': True, 'dependencies': {'aging-fixture': '^1.0.0'}}
        (repo / 'package.json').write_text(json.dumps(manifest), encoding='utf-8')
        (repo / 'pnpm-workspace.yaml').write_text(
            'minimumReleaseAge: 20160\nminimumReleaseAgeExclude:\n'
            '  - gone-fixture@2.0.0  # reason for gone\n'
            '  - aging-fixture@1.0.0  # reason for aging\n', encoding='utf-8')
        args = Namespace(names=['manual-fixture'], target=None, skip_target=[], check=False, uv=False, pnpm=True, termux=False)
        assert cli.cmd_update(args) == 0, capsys.readouterr().out
        assert 'aging-fixture@1.0.0' in (repo / 'pnpm-lock.yaml').read_text(encoding='utf-8')
        # The unlocked entry leaves with its own reason; the locked one keeps its.
        assert (repo / 'pnpm-workspace.yaml').read_text(encoding='utf-8') == (
            'minimumReleaseAge: 20160\nminimumReleaseAgeExclude:\n  - aging-fixture@1.0.0  # reason for aging\n')
        capsys.readouterr()

        # A range only a too-young release satisfies: the hint names the entry to add, and the
        # listing run leaves the project's files as they were.
        manifest['dependencies']['aging-fixture'] = '^1.1.0'
        (repo / 'package.json').write_text(json.dumps(manifest), encoding='utf-8')
        before = {name: (repo / name).read_bytes() for name in ('package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml')}
        assert cli.cmd_update(args) == 1
        out = capsys.readouterr().out
        assert '- aging-fixture@1.1.0  # ' in out, out
        assert {name: (repo / name).read_bytes() for name in before} == before
