"""Release-tag policy accepts only the current stable and canary grammars."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


_RELEASE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "release.py"
_SPEC = importlib.util.spec_from_file_location("hermes_release", _RELEASE_PATH)
assert _SPEC and _SPEC.loader
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


def test_every_stable_selector_rejects_legacy_calver_tags():
    """One shared stable grammar: a CalVer tag (v2026.9.21) must be refused by
    every stable admission path, or a workflow_call carrying the old GitHub
    'latest' tag would be admitted for docker/stable publication."""
    from hermes_cli.source_releases import _valid_tag
    from scripts.releases.docker import DockerReleaseError, require_stable_tag
    from scripts.releases.semver import compare

    for tag in ("v2026.9.21", "v1000.0.0", "v1.01.0", "v1.0.0-canary.20260921000000"):
        assert not _valid_tag(tag, "stable")
        with pytest.raises(DockerReleaseError):
            require_stable_tag(tag)
    assert _valid_tag("v1.0.0", "stable") and require_stable_tag("v999.0.0") == "v999.0.0"
    with pytest.raises(ValueError):
        compare("1.0.0", "2026.9.21")


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


def test_canary_tag_order_and_remote_selection(release_repo):
    git = release_repo
    assert release.get_last_canary_tag() is None
    for tag in ('v2026.7.7', 'v2026.7.20'):
        git('tag', tag)
    for tag in ('v0.9.0', 'v0.20.0', 'v0.19.0', 'v0.20.0+canary.20260818T090000Z',
                'v0.20.0+canary.20260818T171500Z'):
        git('tag', tag)
    assert release.get_last_canary_tag() == 'v0.20.0+canary.20260818T171500Z'
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
