"""Canary policy on real Git refs; GitHub publication stays an inert boundary."""
from datetime import datetime, timezone
from types import SimpleNamespace
import subprocess

import pytest

from tests.scripts.test_release_tags import release, release_repo  # noqa: F401


@pytest.fixture
def canary_repo(tmp_path, monkeypatch, release_repo):
    git = release_repo
    remote = tmp_path / 'remote.git'
    git('init', '--bare', '-q', str(remote))
    git('remote', 'add', 'origin', 'https://github.com/fixture/release')

    calls = []
    actual_run = subprocess.run

    def run(argv, *args, **kwargs):
        if argv[0] != 'gh':
            if argv[:2] == ['git', 'push']:
                argv = ['git', '-c', f'url.{remote.as_uri()}.insteadOf=https://github.com/fixture/release', *argv[1:]]
            return actual_run(argv, *args, **kwargs)
        calls.append(argv)
        if argv[1:3] == ['release', 'create']:
            assert git('--git-dir', str(remote), 'rev-parse', argv[3] + '^{commit}') == git('rev-parse', 'HEAD')
        elif argv[1:3] == ['workflow', 'run']:
            assert any(c[1:3] == ['release', 'create'] for c in calls)
        else:
            assert argv[1:3] == ['repo', 'view']
        return subprocess.CompletedProcess(argv, 0, stdout='main\n', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    which = release.shutil.which
    monkeypatch.setattr(release.shutil, 'which', lambda name: 'gh' if name == 'gh' else which(name))
    monkeypatch.setattr(release, 'generate_changelog', lambda *a, **kw: 'fixture notes')
    monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
    return git, remote, calls


@pytest.mark.parametrize('previous,stable_date,admitted', [
    ('v1.2.4-canary.20260818103045', '2026-08-01T00:00:00Z', False),
    ('v1.2.3-canary.20260818103045', '2026-08-01T00:00:00Z', True),
    (None, '2026-06-01T00:00:00Z', False),
    (None, '2026-08-01T00:00:00Z', True),
])
def test_canary_cut_binds_pushed_head_before_draft_and_dispatch(canary_repo, monkeypatch, previous, stable_date, admitted):
    git, remote, calls = canary_repo
    monkeypatch.setenv('GIT_COMMITTER_DATE', stable_date)
    git('commit', '--allow-empty', '-qm', 'stable')
    git('tag', 'v1.2.3')
    if previous:
        git('tag', previous)
    git('commit', '--allow-empty', '-qm', 'feat: next')
    tag = 'v1.2.4-canary.20260818103000'
    args = SimpleNamespace(date='20260818103000', publish=True, no_changelog=True, remote='origin')
    if stable_date.startswith('2026-06'):
        with pytest.raises(SystemExit) as error:
            release.cmd_canary(args)
        assert error.value.code == 1
    else:
        release.cmd_canary(args)
    if not admitted:
        assert calls == []
        assert tag not in git('tag', '--list').splitlines()
        return
    create = next(c for c in calls if c[1:3] == ['release', 'create'])
    assert create[3] == tag and {'--draft', '--prerelease', '--verify-tag'} <= set(create)
    dispatch = next(c for c in calls if c[1:3] == ['workflow', 'run'])
    assert dispatch == ['gh', 'workflow', 'run', 'desktop-bundled-release.yml', '--ref', 'main',
                        '-f', f'tag={tag}', '-f', 'upload_release=true', '--repo', 'fixture/release']
    assert git('--git-dir', str(remote), 'rev-parse', tag + '^{commit}') == git('rev-parse', 'HEAD')
    calls.clear()
    release.cmd_canary(args)  # existing tag
    args.date = '20260818103100'
    release.cmd_canary(args)  # unchanged HEAD
    assert calls == []


def test_tag_shapes_and_real_prune_cutoff(canary_repo, monkeypatch, capsys):
    git, _, calls = canary_repo
    assert release.canary_tag_for_date('0.27.4', '20260818103000') == 'v0.27.5-canary.20260818103000'
    assert release.canary_tag_for_date('v1.4.0', '20261231235959') == 'v1.4.1-canary.20261231235959'
    tags = ['v0.27.1-canary.20260801', 'v0.27.1-canary.20260801235959',
            'v0.27.1-canary.20260804', 'v0.27.1-canary.20260804235959',
            'v0.27.1-canary.invalid', 'v0.27.1', 'v2026.7.20']
    for tag in tags:
        git('tag', tag)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 18, tzinfo=timezone.utc)
    monkeypatch.setattr(release, 'datetime', Clock)
    release.prune_old_canaries(SimpleNamespace(remote='origin', publish=False))
    assert {line.removeprefix('Would delete ') for line in capsys.readouterr().out.splitlines()
            if line.startswith('Would delete ')} == set(tags[:2])
    assert calls == [] and set(git('tag', '--list').splitlines()) == set(tags)


def test_stable_dispatch_and_failure(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(release, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(release.shutil, 'which', lambda _: 'gh')
    monkeypatch.setattr(release.subprocess, 'run', lambda cmd, **kw: (
        calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, '', '')))
    assert release.dispatch_desktop_build('v1.2.3', 'owner/repo')
    assert calls == [['gh', 'workflow', 'run', 'stable-release.yml', '--ref', 'v1.2.3',
                      '-f', 'tag=v1.2.3', '--repo', 'owner/repo']]
    with pytest.raises(ValueError):
        release.dispatch_desktop_build('v1.2.3/other', 'owner/repo')
    monkeypatch.setattr(release.subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, '', 'refused'))
    assert release.dispatch_desktop_build('v1.2.3', 'owner/repo') is False
    monkeypatch.setattr(release.shutil, 'which', lambda _: None)
    assert release.dispatch_desktop_build('v1.2.4-canary.20260818103000', 'owner/repo') is False
