"""Commit-build entry points resolve real Git refs without publishing releases."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def git(repo, *args):
    return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding='utf-8', timeout=30).stdout.strip()


@pytest.fixture
def fixture_repo(tmp_path):
    upstream = tmp_path / 'upstream.git'
    subprocess.run(['git', 'init', '--bare', '-q', '-b', 'main', str(upstream)], check=True)
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'clone', '-q', str(upstream), str(repo)], check=True)
    git(repo, 'config', 'user.name', 'Fixture')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    (repo / 'pyproject.toml').write_text('[project]\nname="fixture"\nversion="1.2.3"\n', encoding='utf-8')
    git(repo, 'add', 'pyproject.toml')
    git(repo, 'commit', '-qm', 'fixture')
    git(repo, 'push', '-q', 'origin', 'main')
    git(repo, 'remote', 'set-url', 'origin', 'https://github.com/fixture-owner/fixture-repo.git')
    shutil.copytree(ROOT / 'scripts/releases', repo / 'scripts/releases')
    for relative in ('scripts/release.py', 'scripts/release-content-types.json',
                     'hermes_cli/__init__.py', 'hermes_cli/update_channel.py',
                     'hermes_cli/release_channels.py',
                     'pm/paths.py', 'pm/environments.py', 'hermes_constants.py'):
        dest = repo / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    driver = tmp_path / 'drive.py'
    driver.write_text('''import json, os, pathlib, runpy, subprocess, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
actual = subprocess.run
calls = pathlib.Path(os.environ['PROBE_CALLS'])
def run(argv, *args, **kwargs):
    if argv and argv[0] == 'gh':
        with calls.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(argv) + '\\n')
        if os.environ.get('PROBE_GH_MISSING'):
            raise FileNotFoundError('fixture gh is absent')
        if argv[1:3] == ['repo', 'view']:
            assert '--repo' not in argv, 'gh repo view requires a positional repository'
            assert argv[-1].count('/') == 1, argv
            value = 'main\\n'
        elif argv[1:3] == ['workflow', 'run']:
            value = 'fixture dispatch accepted\\n'
        elif argv[1] == 'api':
            value = os.environ.get('PROBE_PERMISSION', 'write') + '\\n'
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=value, stderr='')
    assert argv and argv[0] == 'git', f'unexpected external command: {argv}'
    assert pathlib.Path(kwargs.get('cwd') or pathlib.Path.cwd()).resolve() == root.resolve()
    if argv[1] in ('fetch', 'ls-remote'):
        # Only the transport points at a local fixture. Identity reads stay real.
        rewrites = ['-c', 'url.' + os.environ['PROBE_REMOTE'] + '.insteadOf=https://github.com/fixture-owner/fixture-repo.git']
        if os.environ.get('PROBE_UPSTREAM_URL'):
            rewrites += ['-c', 'url.' + os.environ['PROBE_REMOTE'] + '.insteadOf=' + os.environ['PROBE_UPSTREAM_URL']]
        argv = ['git', *rewrites, *argv[1:]]
    return actual(argv, *args, **kwargs)
subprocess.run = run
if sys.argv[2] == 'admit':
    sys.argv = ['commit_build.py', 'admit']
    runpy.run_module('scripts.releases.commit_build', run_name='__main__')
else:
    sys.argv = ['release.py', *sys.argv[2:]]
    runpy.run_path(str(root / 'scripts/release.py'), run_name='__main__')
''', encoding='utf-8')
    calls = tmp_path / 'gh.jsonl'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GITHUB_', 'GH_'))}
    env.update({'PROBE_CALLS': str(calls), 'PROBE_REMOTE': upstream.as_uri(), 'GIT_ALLOW_PROTOCOL': 'file',
                'GIT_TERMINAL_PROMPT': '0', 'HERMES_HOME': str(tmp_path / 'home')})

    def invoke(*args, extra=None):
        calls.unlink(missing_ok=True)
        result = subprocess.run([sys.executable, '-I', '-S', str(driver), str(repo), *args],
                                cwd=repo, env={**env, **(extra or {})}, capture_output=True,
                                text=True, encoding='utf-8', timeout=45)
        recorded = [json.loads(line) for line in calls.read_text(encoding='utf-8').splitlines()] if calls.exists() else []
        return result, recorded

    return repo, upstream, invoke


def test_commit_build_cli_dispatches_only_the_resolved_remote_commit(fixture_repo):
    repo, upstream, invoke = fixture_repo
    tip = git(repo, 'rev-parse', 'HEAD')
    before = git(repo, 'show-ref', '--heads', '--tags')
    for revision in (tip[:8], 'main'):
        result, calls = invoke('--build-commit', revision)
        assert result.returncode == 0, result.stderr
        assert tip in result.stdout
        assert f'https://hermes-assets.nousresearch.com/releases/commit/{tip}/index.html' in result.stdout
        assert not any(call[1:3] == ['workflow', 'run'] for call in calls)
    result, calls = invoke('--build-commit', tip, '--publish')
    assert result.returncode == 0, result.stderr
    # Repository identity no longer selects the dispatch: every commit build
    # is the same direct dispatch (disposable mode is an explicit opt-in).
    dispatches = [call for call in calls if call[1:3] == ['workflow', 'run']]
    assert len(dispatches) == 1
    assert f'build_commit={tip}' in dispatches[0]
    assert not any(field.startswith('disposable_channel=') for field in dispatches[0])
    assert git(repo, 'show-ref', '--heads', '--tags') == before
    assert git(upstream, 'rev-parse', 'refs/heads/main') == tip


def test_commit_build_dispatch_is_repository_independent(fixture_repo):
    repo, _, invoke = fixture_repo
    tip = git(repo, 'rev-parse', 'HEAD')
    expected = ['gh', 'workflow', 'run', 'desktop-bundled-release.yml',
                '--ref', 'main', '--repo', 'fixture-owner/fixture-repo',
                '-f', f'build_commit={tip}', '-f', 'tag=', '-f', 'upload_release=false',
                '-f', 'termux_upgrade_from_tag=']
    result, calls = invoke('--build-commit', tip, '--publish')
    assert result.returncode == 0, result.stderr
    dispatches = [call for call in calls if call[1:3] == ['workflow', 'run']]
    assert dispatches == [expected]
    assert 'disposable' not in result.stdout.lower()
    # The upstream URL used to select a different command shape; it no longer does.
    git(repo, 'remote', 'set-url', 'origin', 'https://github.com/NousResearch/hermes-agent.git')
    result, calls = invoke('--build-commit', tip, '--publish',
                           extra={'PROBE_UPSTREAM_URL': 'https://github.com/NousResearch/hermes-agent.git'})
    assert result.returncode == 0, result.stderr
    dispatches = [call for call in calls if call[1:3] == ['workflow', 'run']]
    assert dispatches == [['gh', 'workflow', 'run', 'desktop-bundled-release.yml',
                           '--ref', 'main', '--repo', 'NousResearch/hermes-agent',
                           '-f', f'build_commit={tip}', '-f', 'tag=', '-f', 'upload_release=false',
                           '-f', 'termux_upgrade_from_tag=']]


# (Fork-conditional removal) The disposable routing test retired with it: a
# commit build dispatch does not branch on repository identity. Disposable
# channels remain reachable via explicit --channel-request / CI allocation.


def test_commit_bundle_environment_is_literal_and_validated(fixture_repo):
    repo, _, invoke = fixture_repo
    tip = git(repo, 'rev-parse', 'HEAD')
    values = {'HERMES_GUEST_ONBOARDING': '1', 'HERMES_DATA_DIR_SUFFIX': 'magic-test',
              'HERMES_SKIP_INTRO': '', 'HERMES_SHARED_AUTH_DIR': 'a=b "quote"\n$(not-a-command)'}
    flags = [part for key, value in values.items() for part in ('--bundle-env', f'{key}={value}')]
    result, calls = invoke('--build-commit', tip, '--publish', *flags)
    assert result.returncode == 0, result.stderr
    dispatch = next(call for call in calls if call[1:3] == ['workflow', 'run'])
    assert json.loads(next(field.split('=', 1)[1] for field in dispatch if field.startswith('bundle_env='))) == values
    for invalid in (['--bundle-env', 'MISSING'], ['--bundle-env', 'BAD-NAME=x'],
                    ['--bundle-env', 'NODE_OPTIONS=--require=evil'], ['--bundle-unset', 'PATH'],
                    ['--bundle-env', 'HERMES_HOME=x', '--bundle-env', 'HERMES_HOME=y']):
        result, calls = invoke('--build-commit', tip, '--publish', *invalid)
        assert result.returncode != 0 and not calls
    result, calls = invoke('--bundle-env', 'NAME=value')
    assert result.returncode == 2 and not calls

    result, calls = invoke('--build-commit', tip, '--publish', *flags,
                           '--bundle-unset', 'HERMES_HOME')
    assert result.returncode == 0, result.stderr
    dispatch = next(call for call in calls if call[1:3] == ['workflow', 'run'])
    assert json.loads(next(field.split('=', 1)[1] for field in dispatch if field.startswith('bundle_env='))) == {
        **values, 'HERMES_HOME': None}
    for invalid in (['--bundle-unset', 'BAD-NAME'],
                    ['--bundle-env', 'HERMES_HOME=x', '--bundle-unset', 'HERMES_HOME'],
                    ['--bundle-unset', 'HERMES_HOME', '--bundle-unset', 'HERMES_HOME']):
        result, calls = invoke('--build-commit', tip, '--publish', *invalid)
        assert result.returncode != 0 and not calls
    result, calls = invoke('--bundle-unset', 'HERMES_HOME')
    assert result.returncode == 2 and not calls

def test_commit_build_requires_pushed_refs_and_github_origin(tmp_path, fixture_repo):
    repo, upstream, invoke = fixture_repo
    tip = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'checkout', '-qb', 'feature')
    (repo / 'feature').write_text('pushed feature', encoding='utf-8')
    git(repo, 'add', 'feature')
    git(repo, 'commit', '-qm', 'feature')
    feature = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'tag', '-a', 'fixture-tag', '-m', 'fixture')
    git(repo, 'push', '-q', str(upstream), 'feature', 'refs/tags/fixture-tag')
    git(repo, 'checkout', '-q', 'main')
    for revision in (feature[:8], 'origin/feature', 'fixture-tag'):
        result, calls = invoke('--build-commit', revision)
        assert result.returncode == 0, result.stderr
        assert feature in result.stdout
        assert git(repo, 'rev-parse', 'HEAD') == tip
        assert not any(call[1:3] == ['workflow', 'run'] for call in calls)

    for flags in (['--bump', 'patch'], ['--canary'], ['--first-release'], ['--prune-canaries'],
                  ['--date', '2026.1.1'], ['--output', str(tmp_path / 'no-output')], ['--no-changelog']):
        result, calls = invoke('--build-commit', tip, *flags)
        assert result.returncode == 2 and not calls, result.stderr
    assert not (tmp_path / 'no-output').exists()
    for revision in ('', '--all', 'not-a-revision'):
        result, calls = invoke('--build-commit=' + revision, '--publish')
        assert result.returncode != 0 and not calls
        assert 'Traceback' not in result.stderr
    result, calls = invoke('--build-commit', tip, extra={'PROBE_GH_MISSING': '1'})
    assert result.returncode != 0 and 'Traceback' not in result.stderr
    (repo / 'local-only').write_text('not pushed', encoding='utf-8')
    git(repo, 'add', 'local-only')
    git(repo, 'commit', '-qm', 'local only')
    result, calls = invoke('--build-commit', 'HEAD', '--publish')
    assert result.returncode != 0 and 'not reachable' in result.stderr and not calls
    for url in (str(tmp_path / 'github.com/owner/repo'), 'https://notgithub.com/owner/repo',
                'https://github.com/owner/repo/extra'):
        git(repo, 'remote', 'set-url', 'origin', url)
        result, calls = invoke('--build-commit', tip)
        assert result.returncode != 0 and 'GitHub remote' in result.stderr and not calls


def test_workflow_admission_checks_trust_before_publishing_outputs(tmp_path, fixture_repo):
    repo, _, invoke = fixture_repo
    tip = git(repo, 'rev-parse', 'HEAD')
    output = tmp_path / 'output'
    env = {'BUILD_COMMIT': tip, 'DEFAULT_BRANCH': 'main',
           'GITHUB_REPOSITORY': 'fixture-owner/fixture-repo', 'GITHUB_EVENT_NAME': 'workflow_dispatch',
           'GITHUB_REF': 'refs/heads/main',
           'GITHUB_WORKFLOW_REF': 'fixture-owner/fixture-repo/.github/workflows/desktop-bundled-release.yml@refs/heads/main',
           'GITHUB_ACTOR': 'maintainer', 'GITHUB_TRIGGERING_ACTOR': 'maintainer',
           'GITHUB_OUTPUT': str(output), 'UPLOAD_RELEASE': 'false',
           'BUNDLE_ENV_JSON': '{"HERMES_GUEST_ONBOARDING":"1","HERMES_HOME":null}'}
    result, calls = invoke('admit', extra=env)
    assert result.returncode == 0, result.stderr
    assert dict(line.split('=', 1) for line in output.read_text(encoding='utf-8').splitlines()) == {
        'sha': tip, 'channel': 'commit', 'payload-version': '1.2.3'}
    assert calls and all(call[1] == 'api' for call in calls)
    original = output.read_bytes()
    for key, value in [('BUILD_COMMIT', tip[:8]), ('TAG', 'v1.2.3'), ('UPLOAD_RELEASE', 'true'),
                       ('RELEASE_PHASE', 'candidate'), ('TERMUX_UPGRADE_FROM_TAG', 'v1.0.0'),
                       ('GITHUB_EVENT_NAME', 'workflow_call'), ('GITHUB_REF', 'refs/heads/other'),
                       ('GITHUB_WORKFLOW_REF', 'fixture-owner/fixture-repo/.github/workflows/other.yml@refs/heads/main'),
                       ('PROBE_PERMISSION', 'read'), ('BUNDLE_ENV_JSON', '[]'),
                       ('BUNDLE_ENV_JSON', '{"X":1}')]:
        result, _ = invoke('admit', extra={**env, key: value})
        assert result.returncode != 0
        assert output.read_bytes() == original
