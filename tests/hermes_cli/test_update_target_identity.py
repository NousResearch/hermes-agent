"""Stable updates consume one remote commit across Git and archive transports."""
from copy import deepcopy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit
import urllib.request

import pytest

from hermes_cli import main as cli_main, update_cmd, update_receipt
from hermes_cli.update_inventory import UpdatePlan
from hermes_cli.update_cmd import _sync_with_upstream_if_needed


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
    monkeypatch.setattr(cli_main, '_run_pre_update_backup', lambda *_: 'release-snapshot')
    monkeypatch.setattr(cli_main, '_pause_windows_gateways_for_update', lambda: None)
    resumed = []
    monkeypatch.setattr(cli_main, '_resume_windows_gateways_after_update', lambda state: resumed.append(state))
    monkeypatch.setattr(cli_main, '_install_hangup_protection', lambda **_: {'installed': False})
    monkeypatch.setattr(cli_main, '_finalize_update_output', lambda *_: None)
    plans = []

    def inventory():
        sha = git(clone, 'rev-parse', 'HEAD') if (clone / '.git').exists() else base
        plan = UpdatePlan(install_method='git', expected_sha=sha, profiles=['default'])
        plans.append(plan)
        return plan

    monkeypatch.setattr('hermes_cli.update_inventory.collect_runtime_inventory', inventory)
    monkeypatch.setattr(cli_main, '_sync_with_upstream_if_needed',
                        lambda *_a, **_k: pytest.fail('stable update reached upstream branch sync'))

    requests = []

    def capture(request):
        requests.append(deepcopy(request))
        return {'exit_code': 0, 'receipt': None}

    monkeypatch.setattr(update_cmd, 'run_completion', capture)
    args = SimpleNamespace(branch=None, channel='stable', yes=True, force=True, force_venv=True,
                           check=False, plan=False, gateway=False, install_id=False, set_channel=None)
    return SimpleNamespace(origin=origin, clone=clone, base=base, wanted=wanted, newer=newer,
                           args=args, resumed=resumed, requests=requests, plans=plans)


@pytest.mark.parametrize('case', ['main', 'explicit', 'missing', 'no-move', 'wrong-branch',
                                'fork-no-upstream', 'fork-upstream', 'fork-upstream-push-ok',
                                'fork-upstream-wrong-branch', 'fork-upstream-reverted',
                                'fork-late', 'fork-late-push-ok', 'fork-late-wrong-branch',
                                'fork-late-reverted', 'check-main',
                                'check-explicit', 'check-missing', 'check-upstream'])
def test_branch_update_uses_real_refs_and_completion_request(update_tree, monkeypatch, case, capsys):
    t = update_tree
    git(t.clone, 'checkout', '-q', 'main')
    t.args.channel = 'main'
    t.args.gateway = True
    t.args.check = case.startswith('check-')
    monkeypatch.setattr(cli_main, '_sync_with_upstream_if_needed', _sync_with_upstream_if_needed)
    if case in {'explicit', 'check-explicit'}:
        git(t.origin, 'branch', 'chosen', t.wanted)
        t.args.branch = 'chosen'
    if case in {'missing', 'check-missing'}:
        t.args.branch = 'absent'
    if case.startswith('fork-') or case == 'check-upstream':
        # Early sync starts level with origin; late sync first consumes origin.
        git(t.origin, 'reset', '--hard', t.wanted if case.startswith('fork-late') else t.base)
        if case.startswith(('fork-upstream', 'fork-late')) or case == 'check-upstream':
            upstream = t.origin.parent / 'upstream'
            git(t.origin.parent, 'clone', '-q', str(t.origin), str(upstream))
            git(upstream, 'reset', '--hard', t.newer)
            git(t.clone, 'remote', 'add', 'upstream', str(upstream))
            if case.endswith('push-ok'):
                git(t.origin, 'config', 'receive.denyCurrentBranch', 'updateInstead')
    local = t.clone / '.gitignore'
    local_work = local.read_bytes() + b'# local work\n'
    if case.startswith('fork-late'):
        local.write_bytes(local_work)
    run = subprocess.run
    pushes = []

    def fault(command, *args, **kwargs):
        assert Path(command[0]).name.lower() in {'git', 'git.exe'} or command[0] == sys.executable, command
        assert Path(kwargs['cwd']).resolve() in {t.clone, t.origin}, command
        if 'merge' in command and '--ff-only' in command:
            if case == 'no-move':
                return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
            result = run(command, *args, **kwargs)
            if case in {'wrong-branch', 'fork-upstream-wrong-branch'}:
                run(['git', 'checkout', '-qb', 'wrong'], cwd=t.clone, check=True, capture_output=True)
            if case == 'fork-upstream-reverted':
                run(['git', 'reset', '--hard', t.base], cwd=t.clone, check=True, capture_output=True)
            return result
        result = run(command, *args, **kwargs)
        if 'pull' in command and case.startswith('fork-late'):
            if case.endswith('wrong-branch'):
                run(['git', 'checkout', '-qb', 'wrong'], cwd=t.clone, check=True, capture_output=True)
            if case.endswith('reverted'):
                run(['git', 'reset', '--hard', t.base], cwd=t.clone, check=True, capture_output=True)
        if 'push' in command and 'origin' in command:
            pushes.append(result.returncode)
        return result

    monkeypatch.setattr(subprocess, 'run', fault)
    fails = case in {'missing', 'check-missing', 'no-move', 'wrong-branch',
                     'fork-upstream-wrong-branch', 'fork-upstream-reverted',
                     'fork-late-wrong-branch', 'fork-late-reverted'}
    if fails:
        with pytest.raises(SystemExit) as error:
            cli_main.cmd_update(t.args)
        assert error.value.code == 1
        assert t.requests == []
        if case.startswith('fork-late'):
            assert git(t.clone, 'show', 'stash@{0}:.gitignore') == local_work.decode().strip()
            assert local.read_bytes() != local_work
    else:
        cli_main.cmd_update(t.args)
        if t.args.check:
            assert t.requests == []
            assert git(t.clone, 'rev-parse', 'HEAD') == t.base
            output = capsys.readouterr().out
            assert ('absent' if case == 'check-missing' else 'update') in output.lower()
            if case == 'check-upstream':
                assert git(t.clone, 'rev-parse', 'upstream/main') == t.newer
        else:
            request, = t.requests
            assert request['assume_yes'] is True
            assert request['gateway_mode'] is True
            expected = t.base if case == 'fork-no-upstream' else t.wanted if case == 'explicit' else t.newer
            assert git(t.clone, 'rev-parse', 'HEAD') == expected
            assert request['source'] == str(t.clone)
            if case.startswith(('fork-upstream', 'fork-late')):
                assert request['expected_sha'] == expected
                output = capsys.readouterr().out
                assert 'Code did not move' not in output
                assert 'Already up to date' not in output
                origin_before = t.wanted if case.startswith('fork-late') else t.base
                assert git(t.origin, 'rev-parse', 'HEAD') == (t.newer if case.endswith('push-ok') else origin_before)
                assert pushes and all((rc == 0) is case.endswith('push-ok') for rc in pushes)
                if case.startswith('fork-late'):
                    assert local.read_bytes() == local_work
                    assert not git(t.clone, 'stash', 'list')
            if case == 'fork-no-upstream':
                assert 'official repo not checked' in request['completion_message']
                assert git(t.clone, 'remote') == 'origin'


@pytest.mark.parametrize('server', ['sha', 'fetch-refused', 'at-release', 'ahead-release', 'explicit-branch'])
def test_stable_git_uses_remote_identity_without_moving_local_tags(update_tree, monkeypatch, server):
    """A stable update is pinned to the channel's exact commit: no tag lookup on
    origin, the stale local ``v1.1.0`` never moves, and an explicit --branch
    bypasses the channel."""
    from hermes_cli import source_releases
    from hermes_cli.release_channels import ChannelResolution

    t = update_tree
    # The stable channel is an R2 record whose published build pins t.wanted
    # (the documented reader seam; see test_source_channel_integration).
    record = {"schema": 1, "name": "stable", "repository": "NousResearch/hermes-agent",
              "policy": "stable-release", "state": "active", "identity": {}, "nextSequence": 2,
              "head": {"buildId": "build-fixture", "sequence": 1}}
    manifest = {"schema": 1, "request": {"buildId": "build-fixture", "channel": "stable", "sequence": 1,
                "repository": "NousResearch/hermes-agent", "commit": t.wanted, "sourceVersion": "1.1.0",
                "version": "0.0.1", "identity": {}, "bundleEnv": {}}, "packages": []}
    monkeypatch.setattr(source_releases, '_resolve_channel',
                        lambda name, repository: ChannelResolution(record, record, manifest))
    expected = t.wanted
    if server in {'at-release', 'ahead-release'}:
        git(t.clone, 'fetch', '--no-tags', 'origin', t.wanted)
        git(t.clone, 'merge', '--ff-only', t.wanted)
        if server == 'ahead-release':
            (t.clone / 'local.txt').write_text('local commit\n', encoding='utf-8')
            git(t.clone, 'add', 'local.txt')
            git(t.clone, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'local')
    if server == 'explicit-branch':
        git(t.origin, 'branch', 'retained-branch', t.newer)
        t.args.branch = 'retained-branch'
        expected = t.newer
    run = subprocess.run
    calls = []

    def guarded_run(command, *args, **kwargs):
        command = list(map(str, command))
        assert Path(command[0]).name.lower() in {'git', 'git.exe'}, command
        cwd = Path(kwargs.get('cwd', os.getcwd())).resolve()
        assert cwd in {t.clone, t.origin}, (command, cwd)
        assert 'push' not in command and 'pull' not in command and 'ls-remote' not in command, command
        calls.append(command)
        if 'fetch' in command and t.wanted in command and server == 'fetch-refused':
            return subprocess.CompletedProcess(command, 128, stdout='', stderr='fixture: raw SHA wants disabled')
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', guarded_run)
    if server == 'fetch-refused':
        with pytest.raises(SystemExit) as error:
            cli_main.cmd_update(t.args)
        assert error.value.code == 1
        assert git(t.clone, 'rev-parse', 'HEAD') == t.base
        assert t.resumed
        assert t.requests == []
    else:
        cli_main.cmd_update(t.args)
        request, = t.requests
        plan, = t.plans
        assert request['source'] == str(t.clone.resolve())
        assert request['branch'] == (t.args.branch or 'main')
        assert request['plan'] == plan.to_dict()
        assert request['receipt']['plan'] == plan.to_dict()
        assert request['snapshot_id'] == 'release-snapshot'
        if server == 'at-release':
            assert request['completion_message'] == '✓ Already up to date!'
            assert request['plan']['expected_sha'] == expected
        else:
            assert request['expected_sha'] == expected
        assert git(t.clone, 'rev-parse', 'HEAD') == expected
        content = 'unreleased-main\n' if server == 'explicit-branch' else 'release\n'
        assert (t.clone / 'content.txt').read_text(encoding='utf-8-sig') == content
        branch = 'retained-branch' if server in {'at-release', 'explicit-branch'} else ''
        assert git(t.clone, 'branch', '--show-current') == branch
    assert git(t.clone, 'rev-parse', 'v1.1.0') == t.base
    assert not git(t.clone, 'status', '--porcelain')


def test_update_brings_release_tags_without_moving_local_ones(update_tree, monkeypatch):
    """The checkout's version identity is its nearest reachable release tag
    (hermes_cli.version_info), so a fetch that never brings new tags keeps it a
    release behind and fails the plugin ``requires_hermes`` gates the new tree
    itself advertises (#122054). New remote release tags travel with the update;
    a tag the checkout already has never moves."""
    t = update_tree
    monkeypatch.setattr(cli_main, '_sync_with_upstream_if_needed', _sync_with_upstream_if_needed)
    git(t.origin, 'tag', 'v1.2.0', t.wanted)  # published after this checkout was made
    git(t.clone, 'checkout', '-q', 'main')
    t.args.channel = 'main'
    t.args.gateway = True
    cli_main.cmd_update(t.args)
    assert git(t.clone, 'rev-parse', 'HEAD') == t.newer
    assert git(t.clone, 'rev-parse', 'v1.2.0') == t.wanted  # the new release tag arrived
    assert git(t.clone, 'rev-parse', 'v1.1.0') == t.base    # the local tag stayed put


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('transport', ['gitless', 'no-git', 'git-error', 'dirty'])
def test_stable_zip_consumes_the_same_commit_through_the_real_swap(update_tree, monkeypatch, tmp_path, transport):
    from hermes_cli import source_releases
    from hermes_cli.release_channels import ChannelResolution

    t = update_tree
    monkeypatch.setattr(cli_main, '_pause_windows_gateways_for_update',
                        lambda: {"resume_needed": True})
    archive = tmp_path / 'source.zip'
    git(t.origin, 'archive', '--format=zip', '--prefix=hermes-agent-source/', f'--output={archive}', t.wanted)
    archive_bytes = archive.read_bytes()
    record = {"schema": 1, "name": "stable", "repository": "NousResearch/hermes-agent",
              "policy": "stable-release", "state": "active", "identity": {}, "nextSequence": 2,
              "head": {"buildId": "build-fixture", "sequence": 1}}
    manifest = {"schema": 1, "request": {"buildId": "build-fixture", "channel": "stable", "sequence": 1,
                "repository": "NousResearch/hermes-agent", "commit": t.wanted, "sourceVersion": "1.1.0",
                "version": "0.0.1", "identity": {}, "bundleEnv": {}}, "packages": []}
    monkeypatch.setattr(source_releases, '_resolve_channel',
                        lambda name, repository: ChannelResolution(record, record, manifest))
    routes = {
        f'/NousResearch/hermes-agent/archive/{t.wanted}.zip': archive_bytes,
    }

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
        assert parsed.scheme == 'https' and parsed.netloc in {'api.github.com', 'github.com', 'hermes-assets.nousresearch.com'}, url
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
        if 'fetch' in command:
            fetched = True
        return result

    if transport in {'gitless', 'no-git'}:
        (t.clone / '.git').rename(tmp_path / 'git-state')
    if transport == 'dirty':
        (t.clone / 'content.txt').write_text('local work\n', encoding='utf-8')
    before = (t.clone / 'content.txt').read_bytes()
    monkeypatch.setattr(urllib.request, 'urlopen', local_open)
    monkeypatch.setattr(subprocess, 'run', guarded_run)
    try:
        if transport == 'dirty':
            with pytest.raises(SystemExit) as error:
                cli_main.cmd_update(t.args)
            assert error.value.code == 1
            assert (t.clone / 'content.txt').read_bytes() == before
            assert not any('/archive/' in url for url in urls)
            assert t.requests == []
        else:
            cli_main.cmd_update(t.args)
            request, = t.requests
            plan, = t.plans
            assert request['source'] == str(t.clone.resolve())
            assert request['expected_sha'] == t.wanted
            assert request['plan'] == plan.to_dict()
            assert request['receipt']['plan'] == plan.to_dict()
            assert request['snapshot_id'] == 'release-snapshot'
            assert (t.clone / 'content.txt').read_text(encoding='utf-8-sig') == 'release\n'
            assert [url for url in urls if '/archive/' in url] == [
                f'https://github.com/NousResearch/hermes-agent/archive/{t.wanted}.zip']
        assert t.resumed
        if transport in {'git-error', 'dirty'}:
            assert failed and fetched
            assert not any('ls-remote' in cmd for cmd in git_calls)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize('sync_phase', ['origin', 'early', 'late', 'late-other-branch'])
@pytest.mark.parametrize('dirty', [False, True])
def test_update_syntax_failure_restores_pre_update_head(update_tree, monkeypatch, capsys, sync_phase, dirty):
    t = update_tree
    git(t.clone, 'checkout', '-q', 'main')
    t.args.channel = 'main'
    monkeypatch.setattr(cli_main, '_sync_with_upstream_if_needed', _sync_with_upstream_if_needed)
    if sync_phase != 'origin':
        upstream = t.origin.parent / 'upstream'
        git(t.origin.parent, 'clone', '-q', str(t.origin), str(upstream))
        git(t.clone, 'remote', 'add', 'upstream', str(upstream))
        if sync_phase == 'early':
            git(t.origin, 'reset', '--hard', t.base)
        remote = upstream
    else:
        remote = t.origin
    bad = remote / 'hermes_cli' / 'config.py'
    bad.parent.mkdir()
    bad.write_text('def broken(:\n', encoding='utf-8')
    git(remote, 'add', 'hermes_cli/config.py')
    git(remote, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'invalid syntax')
    unexpected_head = git(remote, 'rev-parse', 'HEAD')
    if sync_phase == 'late-other-branch':
        def switch_after_sync(*args, **kwargs):
            result = _sync_with_upstream_if_needed(*args, **kwargs)
            git(t.clone, 'checkout', '-qb', 'unexpected')
            return result

        monkeypatch.setattr(cli_main, '_sync_with_upstream_if_needed', switch_after_sync)
    local = t.clone / '.gitignore'
    staged = local.read_bytes() + b'# staged local work\n'
    unstaged = staged + b'# unstaged local work\n'
    if dirty:
        local.write_bytes(staged)
        git(t.clone, 'add', '.gitignore')
        local.write_bytes(unstaged)
        (t.clone / 'notes.txt').write_bytes(b'untracked local work\n')
    with pytest.raises(SystemExit) as error:
        cli_main.cmd_update(t.args)
    assert error.value.code == 1
    assert not t.requests
    output = capsys.readouterr().out
    if sync_phase == 'late-other-branch':
        assert git(t.clone, 'rev-parse', 'unexpected') == unexpected_head
        assert git(t.clone, 'branch', '--show-current') == 'unexpected'
        assert (t.clone / 'hermes_cli/config.py').read_bytes() == bad.read_bytes()
        assert "checkout is on 'unexpected'" in output
        assert 'Rolling back' not in output
    else:
        assert 'Pulled code has a syntax error' in output
        assert git(t.clone, 'rev-parse', 'HEAD') == t.base
        assert not (t.clone / 'hermes_cli' / 'config.py').exists()
    assert not git(t.clone, 'status', '--porcelain')
    assert bool(git(t.clone, 'stash', 'list')) is dirty
    if dirty:
        assert git(t.clone, 'show', 'stash@{0}^2:.gitignore') == staged.decode().strip()
        assert git(t.clone, 'show', 'stash@{0}:.gitignore') == unstaged.decode().strip()
        assert git(t.clone, 'show', 'stash@{0}^3:notes.txt') == 'untracked local work'
        git(t.clone, 'stash', 'apply', '--index')
        assert local.read_bytes() == unstaged
        assert git(t.clone, 'show', ':.gitignore') == staged.decode().strip()
        assert (t.clone / 'notes.txt').read_bytes() == b'untracked local work\n'

