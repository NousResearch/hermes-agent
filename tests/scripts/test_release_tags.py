"""Release-tag policy: new releases use semver, old CalVer tags remain readable."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


_RELEASE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "release.py"
_SPEC = importlib.util.spec_from_file_location("hermes_release", _RELEASE_PATH)
assert _SPEC and _SPEC.loader
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


def test_release_tag_uses_the_semver_version():
    assert release.release_tag_for_version("0.20.0") == "v0.20.0"


@pytest.fixture
def release_repo(tmp_path, monkeypatch):
    from tests.scripts.test_release_build_commit import git

    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(tmp_path / 'absent-config'))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    git(tmp_path, 'init', '-q', '-b', 'main')
    git(tmp_path, 'config', 'user.name', 'Fixture')
    git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    git(tmp_path, 'commit', '--allow-empty', '-qm', 'fixture')
    monkeypatch.setattr(release, 'REPO_ROOT', tmp_path)
    return lambda *args: git(tmp_path, *args)


def test_real_tag_order_and_remote_selection(release_repo):
    git = release_repo
    assert release.get_last_tag() is None and release.get_last_canary_tag() is None
    for tag in ('v2026.7.7', 'v2026.7.20'):
        git('tag', tag)
    assert release.get_last_tag() == 'v2026.7.20'
    for tag in ('v0.9.0', 'v0.20.0', 'v0.19.0', 'v0.21.0-canary.20260818090000',
                'v0.21.0-canary.20260818171500', 'v0.21.0-canary.20260818'):
        git('tag', tag)
    assert release.get_last_tag() == 'v0.20.0'
    assert release.get_last_canary_tag() == 'v0.21.0-canary.20260818171500'
    with pytest.raises(SystemExit, match='no git remotes'):
        release.resolve_push_remote(None)
    git('remote', 'add', 'origin', 'https://github.com/o/r')
    assert release.resolve_push_remote(None) == 'origin'
    git('remote', 'add', 'fork', 'https://github.com/f/r')
    with pytest.raises(SystemExit, match='pass --remote'):
        release.resolve_push_remote(None)
    assert release.resolve_push_remote('fork') == 'fork'
    with pytest.raises(SystemExit, match='not configured'):
        release.resolve_push_remote('upstream')


def test_github_repo_parsed_from_ssh_and_https_urls(tmp_path, release_repo):
    urls = {
        "fork": "git@github.com:ethernet8023/hermes-agent.git",
        "origin": "https://github.com/NousResearch/hermes-agent",
        "gitlab": "git@gitlab.com:someone/elsewhere.git",
    }
    for name, url in urls.items():
        subprocess.run(['git', 'config', f'remote.{name}.url', url], cwd=tmp_path, check=True)

    assert release.remote_github_repo("fork") == "ethernet8023/hermes-agent"
    assert release.remote_github_repo("origin") == "NousResearch/hermes-agent"
    assert release.remote_github_repo("gitlab") is None
    subprocess.run(['git', 'config', 'url.https://github.com/fork/.pushInsteadOf',
                    'https://github.com/NousResearch/'], cwd=tmp_path, check=True)
    assert release.remote_github_repo('origin') == 'fork/hermes-agent'
    subprocess.run(['git', 'config', 'remote.origin.pushurl', 'ssh://git@github.com:22/other/repo.git'],
                   cwd=tmp_path, check=True)
    assert release.remote_github_repo('origin') == 'other/repo'
