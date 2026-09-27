"""The Python stamp writer binds tagless identity to the built checkout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_stamp_uses_built_commit_even_with_dispatch_sha_and_refuses_mismatch(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    for relative in ('scripts/write_install_stamp.py',
                     'hermes_cli/__init__.py', 'hermes_cli/update_channel.py', 'hermes_cli/release_channels.py',
                     'pm/paths.py', 'pm/environments.py', 'hermes_cli/steward.py', 'hermes_constants.py'):
        dest = repo / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    # commit_build/distance -> versioning -> semver import each other; copy the
    # whole package so a new sibling import cannot break the fixture.
    shutil.copytree(ROOT / 'scripts/releases', repo / 'scripts/releases',
                    ignore=shutil.ignore_patterns('__pycache__'))

    def git(*args):
        return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True,
                              text=True, encoding='utf-8', timeout=30).stdout.strip()

    git('init', '-q', '-b', 'main')
    git('config', 'user.name', 'Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('add', '.')
    git('commit', '-qm', 'main')
    main = git('rev-parse', 'HEAD')
    git('checkout', '-qb', 'feature')
    (repo / 'feature').write_text('feature\n', encoding='utf-8')
    git('add', 'feature')
    git('commit', '-qm', 'feature')
    feature = git('rev-parse', 'HEAD')
    out = tmp_path / 'stamp.json'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GITHUB_', 'HERMES_BUILD_', 'HERMES_PAYLOAD_'))}
    env.update({'GITHUB_SHA': main, 'GITHUB_REF_NAME': 'main',
                'HERMES_BUILD_COMMIT': feature, 'HERMES_DESKTOP_VARIANT': 'bundled',
                'HERMES_HOME': str(tmp_path / 'home')})
    command = [sys.executable, '-I', '-S', str(repo / 'scripts/write_install_stamp.py'),
               '--output', str(out), '--base-version', '0.28.0', '--distance', '0',
               '--update-mechanism', 'app-installer']

    def run(*args, override=None):
        return subprocess.run([*command, *args], cwd=tmp_path, env={**env, **(override or {})},
                              capture_output=True, text=True, encoding='utf-8', timeout=30)

    result = run()
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding='utf-8'))
    assert data['commit'] == feature and data['source'] == 'commit-build'
    assert data['baseVersion'] == data['displayVersion'] == '0.28.0'
    assert data['branch'] is None and data['tag'] is None and data['updateMechanism'] == 'external'
    result = run('--commit', feature)
    assert result.returncode == 0, result.stderr
    before = out.read_bytes()
    git('checkout', '-q', 'main')
    for args in ((), ('--commit', feature)):
        result = run(*args)
        assert result.returncode != 0 and 'checkout' in result.stderr.lower()
        assert out.read_bytes() == before
    git('checkout', '-q', 'feature')
    for extra in ({'HERMES_BUILD_COMMIT': feature[:8]}, {'HERMES_PAYLOAD_TAG': 'v1.2.3'},
                  {'HERMES_BUILD_COMMIT': ' ' + feature}):
        assert run(override=extra).returncode != 0
        assert out.read_bytes() == before
    result = run('--commit', feature, override={'HERMES_BUILD_COMMIT': '', 'HERMES_PAYLOAD_TAG': 'v1.2.3'})
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text(encoding='utf-8'))['tag'] == 'v1.2.3'
