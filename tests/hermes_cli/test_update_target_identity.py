"""Stable updates consume one remote commit across Git and archive transports."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit
import urllib.request

import pytest

from hermes_cli import main as cli_main, update_cmd, update_cmd_zip, update_receipt


class DependencyBoundary(Exception):
    pass


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding='utf-8').stdout.strip()


@pytest.fixture
def update_tree(tmp_path, monkeypatch):
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(tmp_path / 'git-config'))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('GIT_ALLOW_PROTOCOL', 'file')
    monkeypatch.delenv('HERMES_UPDATE_HANDOFF_PID', raising=False)
    monkeypatch.delenv('HERMES_UPDATE_REEXEC', raising=False)
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '-q', '-b', 'main')
    git(origin, 'config', 'user.name', 'Fixture')
    git(origin, 'config', 'user.email', 'fixture@example.invalid')
    (origin / 'content.txt').write_text('base\n', encoding='utf-8')
    (origin / '.gitignore').write_text('.bytecode-fingerprint\n', encoding='utf-8')
    git(origin, 'add', 'content.txt', '.gitignore')
    git(origin, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base')
    git(origin, 'tag', 'v1.0.0')
    base = git(origin, 'rev-parse', 'HEAD')
    clone = tmp_path / 'install'
    git(tmp_path, 'clone', '-q', str(origin), str(clone))
    git(clone, 'config', 'user.name', 'Fixture')
    git(clone, 'config', 'user.email', 'fixture@example.invalid')
    git(clone, 'checkout', '-qb', 'retained-branch')
    git(clone, 'tag', 'v1.1.0')  # A stale local tag must never choose the release.
    (origin / 'content.txt').write_text('release\n', encoding='utf-8')
    git(origin, '-c', 'commit.gpgsign=false', 'commit', '-qam', 'release')
    wanted = git(origin, 'rev-parse', 'HEAD')
    git(origin, '-c', 'tag.gpgSign=false', 'tag', '-a', 'v1.1.0', '-m', 'release')
    (origin / 'content.txt').write_text('unreleased-main\n', encoding='utf-8')
    git(origin, '-c', 'commit.gpgsign=false', 'commit', '-qam', 'unreleased')
    newer = git(origin, 'rev-parse', 'HEAD')

    monkeypatch.setattr(cli_main, 'PROJECT_ROOT', clone)
    monkeypatch.setattr(update_receipt, '_code_identity', lambda **_: {'commit': base})
    monkeypatch.setattr(cli_main, '_capture_active_lazy_features', lambda: [])
    monkeypatch.setattr(cli_main, '_run_pre_update_backup', lambda *_: None)
    monkeypatch.setattr(cli_main, '_pause_windows_gateways_for_update', lambda: None)
    resumed = []
    monkeypatch.setattr(cli_main, '_resume_windows_gateways_after_update', lambda state: resumed.append(state))
    monkeypatch.setattr(cli_main, '_install_hangup_protection', lambda **_: {'installed': False})
    monkeypatch.setattr(cli_main, '_finalize_update_output', lambda *_: None)
    monkeypatch.setattr(update_cmd, '_begin_update_receipt_and_plan', lambda *_: None)
    monkeypatch.setattr(cli_main, '_sync_with_upstream_if_needed',
                        lambda *_a, **_k: pytest.fail('stable update reached upstream branch sync'))

    def stop_at_dependencies(*_args, **_kwargs):
        raise DependencyBoundary()

    monkeypatch.setattr(update_cmd, '_sync_python_dependencies_after_pull', stop_at_dependencies)
    monkeypatch.setattr(update_cmd_zip, '_reinstall_python_deps_after_zip', stop_at_dependencies)
    repaired = []
    monkeypatch.setattr(update_cmd, '_repair_current_checkout', lambda **_: repaired.append(True) or True)
    monkeypatch.setattr(update_cmd, '_apply_pending_fleet_restart_catchup', lambda: None)
    args = SimpleNamespace(branch=None, channel='stable', yes=True, force=True, force_venv=True,
                           check=False, plan=False, gateway=False, install_id=False, set_channel=None)
    return SimpleNamespace(origin=origin, clone=clone, base=base, wanted=wanted, newer=newer,
                           args=args, resumed=resumed, repaired=repaired)


@pytest.mark.parametrize('server', ['sha', 'tag-fallback', 'moved-sha', 'moved-fallback',
                                  'at-release', 'ahead-release', 'explicit-branch'])
def test_stable_git_uses_remote_identity_without_moving_local_tags(update_tree, monkeypatch, server):
    t = update_tree
    expected = t.wanted
    if server in {'at-release', 'ahead-release'}:
        git(t.clone, 'fetch', '--no-tags', 'origin', t.wanted)
        git(t.clone, 'merge', '--ff-only', t.wanted)
        if server == 'ahead-release':
            (t.clone / 'local.txt').write_text('local commit\n', encoding='utf-8')
            git(t.clone, 'add', 'local.txt')
            git(t.clone, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'local')
        expected = git(t.clone, 'rev-parse', 'HEAD')
    if server == 'explicit-branch':
        git(t.origin, 'branch', 'retained-branch', t.newer)
        t.args.branch = 'retained-branch'
        expected = t.newer
    run = subprocess.run
    calls = []
    resolved = False

    def guarded_run(command, *args, **kwargs):
        nonlocal resolved
        command = list(map(str, command))
        assert Path(command[0]).name.lower() in {'git', 'git.exe'}, command
        cwd = Path(kwargs.get('cwd', os.getcwd())).resolve()
        assert cwd in {t.clone, t.origin}, (command, cwd)
        assert 'push' not in command and 'pull' not in command, command
        calls.append(command)
        if 'fetch' in command and t.wanted in command and server.endswith('fallback'):
            return subprocess.CompletedProcess(command, 128, stdout='', stderr='fixture: raw SHA wants disabled')
        result = run(command, *args, **kwargs)
        if 'ls-remote' in command and not resolved:
            resolved = True
            if server.startswith('moved'):
                run(['git', '-c', 'tag.gpgSign=false', 'tag', '-fa', 'v1.1.0', t.newer, '-m', 'moved'],
                    cwd=t.origin, check=True, capture_output=True)
        return result

    monkeypatch.setattr(subprocess, 'run', guarded_run)
    if server == 'moved-fallback':
        with pytest.raises(SystemExit) as error:
            cli_main.cmd_update(t.args)
        assert error.value.code == 1
        assert git(t.clone, 'rev-parse', 'HEAD') == t.base
        assert t.resumed
    elif server in {'at-release', 'ahead-release'}:
        cli_main.cmd_update(t.args)
        assert t.repaired == [True]
        assert git(t.clone, 'rev-parse', 'HEAD') == expected
    else:
        with pytest.raises(DependencyBoundary):
            cli_main.cmd_update(t.args)
        assert git(t.clone, 'rev-parse', 'HEAD') == expected
        content = 'unreleased-main\n' if server == 'explicit-branch' else 'release\n'
        assert (t.clone / 'content.txt').read_text(encoding='utf-8') == content
        assert git(t.clone, 'branch', '--show-current') == 'retained-branch'
    assert resolved == (server != 'explicit-branch')
    assert git(t.clone, 'rev-parse', 'v1.1.0') == t.base
    assert not git(t.clone, 'status', '--porcelain')
    assert len([cmd for cmd in calls if 'ls-remote' in cmd]) == (0 if server == 'explicit-branch' else 1)


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('transport', ['gitless', 'no-git', 'api-tags', 'missing-sha',
                                     'git-error', 'moved-git-error', 'dirty'])
def test_stable_zip_consumes_the_same_commit_through_the_real_swap(update_tree, monkeypatch, tmp_path, transport):
    t = update_tree
    archive = tmp_path / 'source.zip'
    git(t.origin, 'archive', '--format=zip', '--prefix=hermes-agent-source/', f'--output={archive}', t.wanted)
    archive_bytes = archive.read_bytes()
    latest = {'tag_name': 'v1.1.0'}
    routes = {
        '/repos/NousResearch/hermes-agent/releases/latest': latest,
        '/repos/NousResearch/hermes-agent/commits/v1.1.0': {'sha': t.wanted},
        '/repos/NousResearch/hermes-agent/tags?per_page=100': [{'name': 'v1.1.0', 'commit': {'sha': t.wanted}}],
        f'/NousResearch/hermes-agent/archive/{t.wanted}.zip': archive_bytes,
    }
    if transport == 'api-tags':
        routes.pop('/repos/NousResearch/hermes-agent/releases/latest')
    if transport == 'missing-sha':
        routes['/repos/NousResearch/hermes-agent/commits/v1.1.0'] = {'sha': 'not-a-commit'}
        routes.pop('/repos/NousResearch/hermes-agent/tags?per_page=100')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            value = routes.get(self.path)
            if value is None:
                self.send_error(404)
                return
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    urls, git_calls = [], []
    real_open, real_run = urllib.request.urlopen, subprocess.run
    fetched, failed = False, False

    def local_open(request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        parsed = urlsplit(url)
        assert parsed.scheme == 'https' and parsed.netloc in {'api.github.com', 'github.com'}, url
        urls.append(url)
        local = f'http://127.0.0.1:{server.server_port}{parsed.path}'
        if parsed.query:
            local += '?' + parsed.query
        return real_open(local, *args, **kwargs)

    def guarded_run(command, *args, **kwargs):
        nonlocal fetched, failed
        command = list(map(str, command))
        assert Path(command[0]).name.lower() in {'git', 'git.exe'}, command
        cwd = Path(kwargs.get('cwd', os.getcwd())).resolve()
        assert cwd in {t.clone, t.origin}, (command, cwd)
        assert 'push' not in command and 'pull' not in command, command
        git_calls.append(command)
        if transport == 'no-git':
            raise FileNotFoundError('fixture: no Git executable')
        if fetched and not failed and '--abbrev-ref' in command and kwargs.get('check'):
            failed = True
            raise subprocess.CalledProcessError(128, command, '', 'fixture: Git file I/O failed')
        result = real_run(command, *args, **kwargs)
        if 'ls-remote' in command and transport == 'moved-git-error':
            real_run(['git', '-c', 'tag.gpgSign=false', 'tag', '-fa', 'v1.1.0', t.newer, '-m', 'moved'],
                     cwd=t.origin, check=True, capture_output=True)
        if 'fetch' in command:
            fetched = True
        return result

    if transport in {'gitless', 'no-git', 'api-tags', 'missing-sha'}:
        (t.clone / '.git').rename(tmp_path / 'git-state')
    if transport == 'dirty':
        (t.clone / 'content.txt').write_text('local work\n', encoding='utf-8')
    before = (t.clone / 'content.txt').read_bytes()
    monkeypatch.setattr(urllib.request, 'urlopen', local_open)
    monkeypatch.setattr(subprocess, 'run', guarded_run)
    try:
        if transport in {'missing-sha', 'dirty'}:
            with pytest.raises(SystemExit) as error:
                cli_main.cmd_update(t.args)
            assert error.value.code == 1
            assert (t.clone / 'content.txt').read_bytes() == before
            assert not any('/archive/' in url for url in urls)
        else:
            with pytest.raises(DependencyBoundary):
                cli_main.cmd_update(t.args)
            assert (t.clone / 'content.txt').read_text(encoding='utf-8') == 'release\n'
            assert [url for url in urls if '/archive/' in url] == [
                f'https://github.com/NousResearch/hermes-agent/archive/{t.wanted}.zip']
        assert t.resumed
        if transport in {'git-error', 'moved-git-error', 'dirty'}:
            assert failed and fetched
            assert len([cmd for cmd in git_calls if 'ls-remote' in cmd]) == 1
            assert not any('api.github.com' in url for url in urls)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
